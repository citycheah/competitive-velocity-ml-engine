-- dbt_feature_store/models/intermediate/int_video_cadence.sql
{{ config(materialized='view') }}

WITH staging_videos AS (
    SELECT
        video_id,
        channel_id,
        published_at,
        published_epoch,
        duration_seconds,
        is_short
    FROM {{ ref('stg_youtube_videos') }}
),

windowed_cadence AS (
    SELECT
        video_id,
        channel_id,
        published_at,
        published_epoch,
        duration_seconds,
        is_short,

        -- 1. Prior upload epoch for interval computation (tie-breaker on video_id)
        LAG(published_epoch, 1) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch, video_id
        ) AS prev_published_epoch,

        -- 2. 24-hour upload cadence blitz (86,400s) strictly excluding t0
        COUNT(*) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
        ) AS upload_count_preceding_24h,

        -- 3. 7-day upload cadence blitz (604,800s) strictly excluding t0
        COUNT(*) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING
        ) AS upload_count_preceding_7d,

        -- 4. Trailing 30-day shorts proportion (2,592,000s) strictly excluding t0
        AVG(CASE WHEN is_short THEN 1.0 ELSE 0.0 END) OVER (
            PARTITION BY channel_id
            ORDER BY published_epoch
            RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING
        ) AS trailing_shorts_ratio_30d

    FROM staging_videos
)

SELECT
    video_id,
    channel_id,
    published_at,
    published_epoch,

    -- Bounded 24-hour upload blitz count (zero-null guaranteed)
    COALESCE(upload_count_preceding_24h, 0) AS channel_upload_blitz_24h,

    -- Bounded 7-day upload blitz count (zero-null guaranteed)
    COALESCE(upload_count_preceding_7d, 0) AS channel_upload_blitz_7d,

    -- Hours elapsed since last channel upload (cold-start imputed to 720.0 hours / 30 days)
    CASE
        WHEN prev_published_epoch IS NULL THEN 720.0
        ELSE ROUND(CAST((published_epoch - prev_published_epoch) AS DOUBLE) / 3600.0, 4)
    END AS hours_since_last_upload,

    -- Cold-start flag for first channel upload
    CASE
        WHEN prev_published_epoch IS NULL THEN 1
        ELSE 0
    END AS is_first_channel_upload,

    -- Trailing 30-day shorts proportion (nullable if 0 prior uploads)
    trailing_shorts_ratio_30d,

    -- Normalized format volatility: 4 * p * (1 - p), bounded in [0.0, 1.0], 0.0 on cold start
    COALESCE(
        ROUND(
            4.0 * trailing_shorts_ratio_30d * (1.0 - trailing_shorts_ratio_30d),
            4
        ),
        0.0
    ) AS channel_format_volatility_30d

FROM windowed_cadence
