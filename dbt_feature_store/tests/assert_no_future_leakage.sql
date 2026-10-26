-- dbt_feature_store/tests/assert_no_future_leakage.sql
-- Assert that point-in-time features strictly exclude records published at or after current t0.
-- Fails if any video record has rolling metrics that include records published at or after current t0.

WITH staging_videos AS (
    SELECT
        video_id,
        channel_id,
        published_epoch,
        LN(1.0 + GREATEST(0.0, COALESCE(
            target_weighted_engagement, 
            CAST(target_views_30d + 10.0 * target_likes_30d + 20.0 * target_comments_30d AS DOUBLE), 
            0.0
        ))) AS curr_log_eng
    FROM {{ ref('stg_youtube_videos') }}
),

ground_truth_windows AS (
    SELECT
        curr.video_id,
        curr.channel_id,
        curr.published_epoch,
        curr.curr_log_eng,
        -- Ground-truth count in [t0 - 86400, t0) strictly excluding curr.published_epoch (DISTINCT prevents M x N fanout)
        COUNT(DISTINCT prior_24h.video_id) AS expected_blitz_24h,
        -- Ground-truth count in [t0 - 2592000, t0) strictly excluding curr.published_epoch (DISTINCT prevents M x N fanout)
        COUNT(DISTINCT prior_30d.video_id) AS expected_30d_count
    FROM staging_videos curr
    LEFT JOIN staging_videos prior_24h
        ON curr.channel_id = prior_24h.channel_id
        AND prior_24h.published_epoch >= curr.published_epoch - 86400
        AND prior_24h.published_epoch < curr.published_epoch
    LEFT JOIN staging_videos prior_30d
        ON curr.channel_id = prior_30d.channel_id
        AND prior_30d.published_epoch >= curr.published_epoch - 2592000
        AND prior_30d.published_epoch < curr.published_epoch
    GROUP BY curr.video_id, curr.channel_id, curr.published_epoch, curr.curr_log_eng
),

mart_features AS (
    SELECT
        video_id,
        channel_id,
        published_epoch,
        channel_upload_blitz_24h,
        hours_since_last_upload,
        channel_historical_avg_engagement_30d
    FROM {{ ref('fct_video_features_t0') }}
)

SELECT
    m.video_id,
    m.channel_id,
    m.published_epoch,
    m.channel_upload_blitz_24h,
    gt.expected_blitz_24h,
    m.hours_since_last_upload,
    m.channel_historical_avg_engagement_30d
FROM mart_features m
JOIN ground_truth_windows gt
    ON m.video_id = gt.video_id
WHERE 
    -- 1. Upload blitz count must strictly equal ground truth prior upload count
    m.channel_upload_blitz_24h != gt.expected_blitz_24h

    -- 2. Elapsed hours since last upload cannot be negative (indicates future look-ahead)
    OR m.hours_since_last_upload < 0.0

    -- 3. Self-leakage check: If no prior videos existed in the 30d window,
    -- the baseline average cannot equal the current video's own engagement
    OR (
        gt.expected_30d_count = 0 
        AND m.channel_historical_avg_engagement_30d != 0.0 
        AND m.channel_historical_avg_engagement_30d = gt.curr_log_eng
    )
