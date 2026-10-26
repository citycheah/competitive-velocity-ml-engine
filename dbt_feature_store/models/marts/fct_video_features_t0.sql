-- dbt_feature_store/models/marts/fct_video_features_t0.sql
{{ config(
    materialized='table'
) }}

WITH videos AS (
    SELECT * FROM {{ ref('stg_youtube_videos') }}
),

channels AS (
    SELECT * FROM {{ ref('stg_youtube_channels') }}
),

cadence AS (
    SELECT * FROM {{ ref('int_video_cadence') }}
),

stats AS (
    SELECT * FROM {{ ref('int_channel_rolling_stats') }}
),

base_joined AS (
    SELECT
        -- Primary Identifiers & Temporal Anchor (t0)
        v.video_id,
        v.channel_id,
        v.published_at,
        v.published_epoch,

        -- Channel Context Observable at t0
        COALESCE(c.channel_title, 'Unknown Channel') AS channel_title,
        COALESCE(c.subscriber_count, 0) AS channel_subscriber_count,
        COALESCE(c.topic_category, 'General') AS topic_category,

        -- Intrinsic Video Features Observable at t0
        COALESCE(v.title, '') AS title,
        COALESCE(v.description, '') AS description,
        COALESCE(v.duration_seconds, 0) AS duration_seconds,
        COALESCE(
            v.is_short, 
            CASE WHEN COALESCE(v.duration_seconds, 0) <= 60 THEN true ELSE false END
        ) AS is_short,
        COALESCE(v.tags_count, 0) AS tags_count,
        COALESCE(v.publish_hour_of_day, 0) AS publish_hour_of_day,
        COALESCE(v.publish_day_of_week, 0) AS publish_day_of_week,
        COALESCE(v.is_weekend, 0) AS is_weekend,
        COALESCE(v.title_length_chars, LENGTH(COALESCE(v.title, ''))) AS title_length_chars,
        COALESCE(
            v.title_word_count, 
            LENGTH(COALESCE(v.title, '')) - LENGTH(REPLACE(COALESCE(v.title, ''), ' ', '')) + 1
        ) AS title_word_count,

        -- Cadence & Velocity Features Strictly Prior to t0
        COALESCE(cad.channel_upload_blitz_24h, 0) AS channel_upload_blitz_24h,
        COALESCE(cad.channel_upload_blitz_7d, 0) AS channel_upload_blitz_7d,
        COALESCE(cad.hours_since_last_upload, 720.0) AS hours_since_last_upload,
        COALESCE(cad.channel_format_volatility_30d, 0.0) AS channel_format_volatility_30d,

        -- Channel Historical Baseline Prior to t0 (from int_channel_rolling_stats)
        COALESCE(stats.channel_historical_avg_views_30d, 0.0) AS channel_historical_avg_views_30d,
        COALESCE(stats.effective_baseline_log_mean_30d, 10.50) AS channel_historical_avg_engagement_30d,
        COALESCE(stats.effective_baseline_log_std_30d, 1.0) AS channel_historical_std_engagement_30d,

        -- Quarantined Realized Targets (Ground Truth Only, Never Used as Features)
        COALESCE(v.target_views_7d, 0) AS target_views_7d,
        COALESCE(v.target_likes_7d, 0) AS target_likes_7d,
        COALESCE(v.target_comments_7d, 0) AS target_comments_7d,
        COALESCE(v.target_views_30d, v.target_views_7d, 0) AS target_views_30d,
        COALESCE(v.target_likes_30d, v.target_likes_7d, 0) AS target_likes_30d,
        COALESCE(v.target_comments_30d, v.target_comments_7d, 0) AS target_comments_30d,
        COALESCE(v.target_weighted_engagement, 0.0) AS target_weighted_engagement,
        COALESCE(v.target_log_engagement, 0.0) AS target_log_engagement

    FROM videos v
    LEFT JOIN channels c
        ON v.channel_id = c.channel_id
    LEFT JOIN cadence cad
        ON v.video_id = cad.video_id
    LEFT JOIN stats
        ON v.video_id = stats.video_id
),

calculated_target AS (
    SELECT
        *,
        -- Effective standard deviation with guardrail: if <= 0 or null, impute 1.0
        CASE 
            WHEN channel_historical_std_engagement_30d IS NULL OR channel_historical_std_engagement_30d <= 0.0 
            THEN 1.0 
            ELSE channel_historical_std_engagement_30d 
        END AS effective_sigma
    FROM base_joined
)

SELECT
    -- --- t0 Observable Feature Attributes (Features X) ---
    video_id,
    channel_id,
    published_at,
    published_epoch,
    channel_title,
    channel_subscriber_count,
    topic_category,
    title,
    description,
    duration_seconds,
    is_short,
    tags_count,
    publish_hour_of_day,
    publish_day_of_week,
    is_weekend,
    title_length_chars,
    title_word_count,
    channel_upload_blitz_24h,
    channel_upload_blitz_7d,
    hours_since_last_upload,
    channel_format_volatility_30d,
    channel_historical_avg_views_30d,
    channel_historical_avg_engagement_30d,
    channel_historical_std_engagement_30d,

    -- --- Quarantined Mature Realizations (Ground Truth Only) ---
    target_views_7d,
    target_likes_7d,
    target_comments_7d,
    target_views_30d,
    target_likes_30d,
    target_comments_30d,
    target_weighted_engagement,

    -- --- Supervised Target Formulation ---
    -- Standardized Z-Score: Z = (E_tilde - mu) / (sigma + 1e-4)
    ROUND(
        (target_log_engagement - channel_historical_avg_engagement_30d) / (effective_sigma + 0.0001),
        6
    ) AS engagement_zscore_30d,

    -- Binary Breakout Label: is_breakout = 1 if Z >= 1.0 else 0
    CASE 
        WHEN ((target_log_engagement - channel_historical_avg_engagement_30d) / (effective_sigma + 0.0001)) >= 1.0 
        THEN 1 
        ELSE 0 
    END AS is_breakout
FROM calculated_target
