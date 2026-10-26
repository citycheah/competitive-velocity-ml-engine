-- dbt_feature_store/tests/assert_blitz_monotonicity.sql
-- Assert that channel_upload_blitz_24h is strictly non-negative (>= 0) and non-null for all records.
-- Also asserts window monotonicity: blitz over trailing 7 days must be >= blitz over trailing 24h.
-- In dbt, test fails if any violating rows are returned.

SELECT
    video_id,
    channel_id,
    published_at,
    channel_upload_blitz_24h,
    channel_upload_blitz_7d
FROM {{ ref('fct_video_features_t0') }}
WHERE channel_upload_blitz_24h < 0
   OR channel_upload_blitz_24h IS NULL
   OR channel_upload_blitz_7d < 0
   OR channel_upload_blitz_7d IS NULL
   OR channel_upload_blitz_7d < channel_upload_blitz_24h
