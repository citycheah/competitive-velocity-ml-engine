{# dbt_feature_store/macros/time_helpers.sql #}

{# 1. Extract Epoch Seconds from Timestamp #}
{% macro epoch_seconds(column_name) -%}
    {{ adapter.dispatch('epoch_seconds', 'dbt_feature_store')(column_name) }}
{%- endmacro %}

{% macro default__epoch_seconds(column_name) -%}
    CAST(date_part('epoch', {{ column_name }}) AS BIGINT)
{%- endmacro %}

{% macro duckdb__epoch_seconds(column_name) -%}
    CAST(date_part('epoch', {{ column_name }}) AS BIGINT)
{%- endmacro %}

{% macro bigquery__epoch_seconds(column_name) -%}
    UNIX_SECONDS({{ column_name }})
{%- endmacro %}


{# 2. Convert Epoch Seconds to Timestamp #}
{% macro timestamp_from_epoch(epoch_col) -%}
    {{ adapter.dispatch('timestamp_from_epoch', 'dbt_feature_store')(epoch_col) }}
{%- endmacro %}

{% macro default__timestamp_from_epoch(epoch_col) -%}
    to_timestamp({{ epoch_col }})
{%- endmacro %}

{% macro duckdb__timestamp_from_epoch(epoch_col) -%}
    to_timestamp({{ epoch_col }})
{%- endmacro %}

{% macro bigquery__timestamp_from_epoch(epoch_col) -%}
    TIMESTAMP_SECONDS({{ epoch_col }})
{%- endmacro %}


{# 3. Difference in Hours between Two Timestamps #}
{% macro timestamp_diff_hours(start_col, end_col) -%}
    {{ adapter.dispatch('timestamp_diff_hours', 'dbt_feature_store')(start_col, end_col) }}
{%- endmacro %}

{% macro default__timestamp_diff_hours(start_col, end_col) -%}
    (date_diff('second', {{ start_col }}, {{ end_col }}) / 3600.0)
{%- endmacro %}

{% macro duckdb__timestamp_diff_hours(start_col, end_col) -%}
    (date_diff('second', {{ start_col }}, {{ end_col }}) / 3600.0)
{%- endmacro %}

{% macro bigquery__timestamp_diff_hours(start_col, end_col) -%}
    (TIMESTAMP_DIFF({{ end_col }}, {{ start_col }}, SECOND) / 3600.0)
{%- endmacro %}


{# 4. Subtract Time Interval from Timestamp #}
{% macro timestamp_sub_interval(timestamp_col, count, unit) -%}
    {{ adapter.dispatch('timestamp_sub_interval', 'dbt_feature_store')(timestamp_col, count, unit) }}
{%- endmacro %}

{% macro default__timestamp_sub_interval(timestamp_col, count, unit) -%}
    ({{ timestamp_col }} - INTERVAL '{{ count }} {{ unit }}')
{%- endmacro %}

{% macro duckdb__timestamp_sub_interval(timestamp_col, count, unit) -%}
    ({{ timestamp_col }} - INTERVAL '{{ count }} {{ unit }}')
{%- endmacro %}

{% macro bigquery__timestamp_sub_interval(timestamp_col, count, unit) -%}
    TIMESTAMP_SUB({{ timestamp_col }}, INTERVAL {{ count }} {{ unit }})
{%- endmacro %}


{# 5. Extract Hour of Day (0-23) #}
{% macro extract_hour(timestamp_col) -%}
    CAST(EXTRACT(HOUR FROM {{ timestamp_col }}) AS INTEGER)
{%- endmacro %}


{# 6. Extract Day of Week (0 = Sunday, 6 = Saturday) #}
{% macro extract_day_of_week(timestamp_col) -%}
    {{ adapter.dispatch('extract_day_of_week', 'dbt_feature_store')(timestamp_col) }}
{%- endmacro %}

{% macro default__extract_day_of_week(timestamp_col) -%}
    CAST(date_part('dow', {{ timestamp_col }}) AS INTEGER)
{%- endmacro %}

{% macro duckdb__extract_day_of_week(timestamp_col) -%}
    CAST(date_part('dow', {{ timestamp_col }}) AS INTEGER)
{%- endmacro %}

{% macro bigquery__extract_day_of_week(timestamp_col) -%}
    CAST(EXTRACT(DAYOFWEEK FROM {{ timestamp_col }}) - 1 AS INT64)
{%- endmacro %}


{# 7. Weekend Indicator (Saturday=6, Sunday=0) as Integer 1 or 0 #}
{% macro is_weekend(timestamp_col) -%}
    {{ adapter.dispatch('is_weekend', 'dbt_feature_store')(timestamp_col) }}
{%- endmacro %}

{% macro default__is_weekend(timestamp_col) -%}
    CASE WHEN date_part('dow', {{ timestamp_col }}) IN (0, 6) THEN 1 ELSE 0 END
{%- endmacro %}

{% macro duckdb__is_weekend(timestamp_col) -%}
    CASE WHEN date_part('dow', {{ timestamp_col }}) IN (0, 6) THEN 1 ELSE 0 END
{%- endmacro %}

{% macro bigquery__is_weekend(timestamp_col) -%}
    CASE WHEN EXTRACT(DAYOFWEEK FROM {{ timestamp_col }}) IN (1, 7) THEN 1 ELSE 0 END
{%- endmacro %}
