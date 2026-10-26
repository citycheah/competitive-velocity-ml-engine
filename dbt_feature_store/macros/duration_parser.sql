{# dbt_feature_store/macros/duration_parser.sql #}

{% macro parse_iso_duration(iso_col) -%}
    {{ adapter.dispatch('parse_iso_duration', 'dbt_feature_store')(iso_col) }}
{%- endmacro %}

{% macro default__parse_iso_duration(iso_col) -%}
    (
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)D', 1), '') AS INTEGER), 0) * 86400 +
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)H', 1), '') AS INTEGER), 0) * 3600 +
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)M', 1), '') AS INTEGER), 0) * 60 +
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)S', 1), '') AS INTEGER), 0)
    )
{%- endmacro %}

{% macro duckdb__parse_iso_duration(iso_col) -%}
    (
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)D', 1), '') AS INTEGER), 0) * 86400 +
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)H', 1), '') AS INTEGER), 0) * 3600 +
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)M', 1), '') AS INTEGER), 0) * 60 +
        COALESCE(TRY_CAST(NULLIF(regexp_extract({{ iso_col }}, '([0-9]+)S', 1), '') AS INTEGER), 0)
    )
{%- endmacro %}

{% macro bigquery__parse_iso_duration(iso_col) -%}
    (
        COALESCE(SAFE_CAST(REGEXP_EXTRACT({{ iso_col }}, r'(\d+)D') AS INT64), 0) * 86400 +
        COALESCE(SAFE_CAST(REGEXP_EXTRACT({{ iso_col }}, r'(\d+)H') AS INT64), 0) * 3600 +
        COALESCE(SAFE_CAST(REGEXP_EXTRACT({{ iso_col }}, r'(\d+)M') AS INT64), 0) * 60 +
        COALESCE(SAFE_CAST(REGEXP_EXTRACT({{ iso_col }}, r'(\d+)S') AS INT64), 0)
    )
{%- endmacro %}
