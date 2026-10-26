-- dbt_feature_store/models/staging/stg_youtube_videos.sql
{{ config(materialized='view') }}

WITH raw_source AS (
    SELECT * FROM {{ source('raw_youtube', 'youtube_videos') }}
),

standardized AS (
    SELECT
        -- Primary Key & Foreign Keys
        CAST(video_id AS VARCHAR) AS video_id,
        CAST(channel_id AS VARCHAR) AS channel_id,

        -- Publication Timestamps (Point-in-Time t0 Anchor)
        CAST(published_at AS TIMESTAMP) AS published_at,
        COALESCE(
            CAST(published_epoch AS BIGINT),
            {{ epoch_seconds('published_at') }}
        ) AS published_epoch,

        -- Intrinsic Video Metadata Knowable at t0
        TRIM(CAST(title AS VARCHAR)) AS title,
        COALESCE(TRIM(CAST(description AS VARCHAR)), '') AS description,
        CAST(duration_seconds AS INTEGER) AS duration_seconds,
        CASE
            WHEN CAST(duration_seconds AS INTEGER) <= 60 THEN true
            ELSE false
        END AS is_short,
        COALESCE(CAST(tags_count AS INTEGER), 0) AS tags_count,

        -- Content & Text Derived Features at t0
        LENGTH(TRIM(CAST(title AS VARCHAR))) AS title_length_chars,
        LENGTH(TRIM(CAST(title AS VARCHAR))) AS title_char_length,
        CASE
            WHEN LENGTH(TRIM(CAST(title AS VARCHAR))) > 0
            THEN LENGTH(TRIM(CAST(title AS VARCHAR))) - LENGTH(REPLACE(TRIM(CAST(title AS VARCHAR)), ' ', '')) + 1
            ELSE 0
        END AS title_word_count,

        -- Calendar & Temporal Context at t0
        {{ extract_hour('published_at') }} AS publish_hour_of_day,
        {{ extract_day_of_week('published_at') }} AS publish_day_of_week,
        {{ is_weekend('published_at') }} AS is_weekend,

        -- Quarantined Target Ground Truth Metrics (STRICTLY FOR SUPERVISED TARGET FORMULATION)
        COALESCE(CAST(target_views_7d AS BIGINT), 0) AS target_views_7d,
        COALESCE(CAST(target_likes_7d AS BIGINT), 0) AS target_likes_7d,
        COALESCE(CAST(target_comments_7d AS BIGINT), 0) AS target_comments_7d,
        COALESCE(CAST(target_views_30d AS BIGINT), CAST(target_views_7d AS BIGINT), 0) AS target_views_30d,
        COALESCE(CAST(target_likes_30d AS BIGINT), CAST(target_likes_7d AS BIGINT), 0) AS target_likes_30d,
        COALESCE(CAST(target_comments_30d AS BIGINT), CAST(target_comments_7d AS BIGINT), 0) AS target_comments_30d,
        COALESCE(CAST(target_views_30d AS BIGINT), CAST(target_views_7d AS BIGINT), 0) AS target_views_mature,
        COALESCE(CAST(target_likes_30d AS BIGINT), CAST(target_likes_7d AS BIGINT), 0) AS target_likes_mature,
        COALESCE(CAST(target_comments_30d AS BIGINT), CAST(target_comments_7d AS BIGINT), 0) AS target_comments_mature,

        -- Pre-calculated Weighted Engagement (V + 10L + 20C)
        COALESCE(
            CAST(target_weighted_engagement AS DOUBLE),
            CAST(
                COALESCE(target_views_30d, target_views_7d, 0) +
                10.0 * COALESCE(target_likes_30d, target_likes_7d, 0) +
                20.0 * COALESCE(target_comments_30d, target_comments_7d, 0) AS DOUBLE
            )
        ) AS target_weighted_engagement,

        -- Pre-calculated Log Weighted Engagement (ln(1 + V + 10L + 20C))
        LN(
            1.0 +
            GREATEST(
                0.0,
                COALESCE(
                    CAST(target_weighted_engagement AS DOUBLE),
                    CAST(
                        COALESCE(target_views_30d, target_views_7d, 0) +
                        10.0 * COALESCE(target_likes_30d, target_likes_7d, 0) +
                        20.0 * COALESCE(target_comments_30d, target_comments_7d, 0) AS DOUBLE
                    )
                )
            )
        ) AS target_log_engagement

    FROM raw_source
    WHERE video_id IS NOT NULL
)

SELECT * FROM standardized
