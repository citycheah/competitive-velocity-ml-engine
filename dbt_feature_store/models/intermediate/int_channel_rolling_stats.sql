-- dbt_feature_store/models/intermediate/int_channel_rolling_stats.sql
{{ config(materialized='view') }}

WITH raw_engagements AS (
    SELECT
        video_id,
        channel_id,
        published_at,
        published_epoch,
        COALESCE(target_views_30d, target_views_7d, 0) AS target_views_clean,
        COALESCE(target_likes_30d, target_likes_7d, 0) AS target_likes_clean,
        COALESCE(target_comments_30d, target_comments_7d, 0) AS target_comments_clean,

        -- 1. Linear weighted engagement composite score: E = V + 10*L + 20*C
        -- Note: For historical prior uploads in the feature store, realized metrics serve as
        -- an offline proxy for channel baseline capacity strictly prior to the current upload t0.
        COALESCE(
            CAST(target_weighted_engagement AS DOUBLE),
            CAST(
                COALESCE(target_views_30d, target_views_7d, 0)
                + 10.0 * COALESCE(target_likes_30d, target_likes_7d, 0)
                + 20.0 * COALESCE(target_comments_30d, target_comments_7d, 0) AS DOUBLE
            )
        ) AS weighted_engagement,

        -- 2. Log-weighted composite engagement: ln(1 + E)
        LN(
            1.0
            + GREATEST(
                0.0,
                COALESCE(
                    CAST(target_weighted_engagement AS DOUBLE),
                    CAST(
                        COALESCE(target_views_30d, target_views_7d, 0)
                        + 10.0 * COALESCE(target_likes_30d, target_likes_7d, 0)
                        + 20.0 * COALESCE(target_comments_30d, target_comments_7d, 0) AS DOUBLE
                    )
                )
            )
        ) AS log_weighted_engagement

    FROM {{ ref('stg_youtube_videos') }}
),

windowed_channel_stats AS (
    SELECT
        video_id,
        channel_id,
        published_at,
        published_epoch,
        target_views_clean,
        target_likes_clean,
        target_comments_clean,
        weighted_engagement,
        log_weighted_engagement,

        -- 1. Preceding 30-day upload count (2,592,000s) strictly excluding t0
        COUNT(*) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
        ) AS channel_upload_count_30d,

        -- 2. Trailing 30-day rolling mean of log-weighted engagement strictly excluding t0
        AVG(log_weighted_engagement) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
        ) AS channel_rolling_avg_log_engagement_30d,

        -- 3. Trailing 30-day sample standard deviation strictly excluding t0
        STDDEV_SAMP(log_weighted_engagement) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
        ) AS channel_rolling_std_log_engagement_30d,

        -- 4. Trailing 30-day average raw views strictly excluding t0
        AVG(target_views_clean) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
        ) AS channel_historical_avg_views_30d,

        -- 5. Trailing 30-day average raw weighted engagement strictly excluding t0
        AVG(weighted_engagement) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
        ) AS channel_historical_avg_engagement_raw_30d,

        -- 6. Lifetime channel upload count strictly excluding t0
        COUNT(*) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS channel_lifetime_upload_count,

        -- 7. Lifetime channel baseline mean strictly excluding t0 (fallback when 30d < 3)
        AVG(log_weighted_engagement) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS channel_lifetime_avg_log_engagement,

        -- 8. Lifetime channel baseline sample stddev strictly excluding t0 (fallback when 30d < 3)
        STDDEV_SAMP(log_weighted_engagement) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS channel_lifetime_std_log_engagement

    FROM raw_engagements
),

computed_effective AS (
    SELECT
        video_id,
        channel_id,
        published_at,
        published_epoch,
        weighted_engagement,
        log_weighted_engagement,

        -- Upload count in preceding 30 days
        COALESCE(channel_upload_count_30d, 0) AS channel_upload_count_30d,
        COALESCE(channel_upload_count_30d, 0) AS rolling_uploads_count_30d,

        -- Rolling 30-day log-engagement mean (mu_{c, 30d})
        channel_rolling_avg_log_engagement_30d,
        channel_rolling_avg_log_engagement_30d AS channel_rolling_avg_engagement_30d,

        -- Rolling 30-day log-engagement sample standard deviation (sigma_{c, 30d})
        channel_rolling_std_log_engagement_30d,
        channel_rolling_std_log_engagement_30d AS channel_rolling_std_engagement_30d,

        -- Trailing 30-day raw views (defaults to 0.0 on cold start)
        COALESCE(ROUND(channel_historical_avg_views_30d, 2), 0.0) AS channel_historical_avg_views_30d,
        COALESCE(ROUND(channel_historical_avg_engagement_raw_30d, 2), 0.0) AS channel_historical_avg_engagement_raw_30d,

        -- Lifetime channel fallback baselines
        channel_lifetime_upload_count,
        channel_lifetime_avg_log_engagement,
        channel_lifetime_std_log_engagement,

        -- Cold-start target indicator: 1 if trailing 30d uploads < 3, else 0
        CASE
            WHEN COALESCE(channel_upload_count_30d, 0) < 3 THEN 1
            ELSE 0
        END AS is_cold_start_target,

        -- Effective baseline mean (mu) with 3-tier fallback hierarchy:
        -- 1) 30d mean (if count >= 3) -> 2) lifetime mean (if lifetime count >= 3) -> 3) global default (10.50)
        COALESCE(
            CASE WHEN channel_upload_count_30d >= 3 THEN channel_rolling_avg_log_engagement_30d END,
            CASE WHEN channel_lifetime_upload_count >= 3 THEN channel_lifetime_avg_log_engagement END,
            channel_rolling_avg_log_engagement_30d,
            channel_lifetime_avg_log_engagement,
            10.50
        ) AS effective_baseline_log_mean_30d,

        -- Effective baseline stddev (sigma) with 3-tier fallback hierarchy:
        -- 1) 30d stddev (if count >= 3 and > 0) -> 2) lifetime stddev (if lifetime count >= 3 and > 0) -> 3) default (1.00)
        COALESCE(
            CASE WHEN channel_upload_count_30d >= 3 THEN NULLIF(channel_rolling_std_log_engagement_30d, 0.0) END,
            CASE WHEN channel_lifetime_upload_count >= 3 THEN NULLIF(channel_lifetime_std_log_engagement, 0.0) END,
            NULLIF(channel_rolling_std_log_engagement_30d, 0.0),
            NULLIF(channel_lifetime_std_log_engagement, 0.0),
            1.00
        ) AS effective_baseline_log_std_30d

    FROM windowed_channel_stats
)

SELECT
    *,
    effective_baseline_log_mean_30d AS channel_historical_avg_engagement_30d,
    effective_baseline_log_std_30d AS channel_historical_std_engagement_30d
FROM computed_effective
