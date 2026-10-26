{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if target.type == 'duckdb' -%}
        {# In DuckDB, keep all models in the target schema (default 'main') #}
        {{ default_schema }}
    {%- else -%}
        {# In BigQuery, prefix custom schema to default dataset if custom exists #}
        {%- if custom_schema_name is none -%}
            {{ default_schema }}
        {%- else -%}
            {{ default_schema }}_{{ custom_schema_name | trim }}
        {%- endif -%}
    {%- endif -%}
{%- endmacro %}
