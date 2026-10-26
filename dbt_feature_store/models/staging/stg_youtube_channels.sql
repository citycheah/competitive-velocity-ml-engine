-- dbt_feature_store/models/staging/stg_youtube_channels.sql
{{ config(materialized='view') }}

WITH raw_source AS (
    SELECT * FROM {{ source('raw_youtube', 'youtube_channels') }}
),

standardized AS (
    SELECT
        -- Primary Identifier
        CAST(channel_id AS VARCHAR) AS channel_id,
        TRIM(CAST(channel_title AS VARCHAR)) AS channel_title,

        -- Baseline Scale & Categorization
        COALESCE(CAST(subscriber_count AS BIGINT), 0) AS subscriber_count,
        COALESCE(TRIM(CAST(topic_category AS VARCHAR)), 'General') AS topic_category,
        COALESCE(TRIM(CAST(tier AS VARCHAR)), 'Tier 2') AS tier,
        COALESCE(TRIM(CAST(country AS VARCHAR)), 'US') AS country,

        -- Account History
        CAST(created_at AS TIMESTAMP) AS created_at

    FROM raw_source
    WHERE channel_id IS NOT NULL
)

SELECT * FROM standardized
