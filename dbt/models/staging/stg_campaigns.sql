SELECT
    CAST(campaign AS INTEGER) AS campaign,
    UPPER(TRIM(description)) AS campaign_type,
    CAST(start_day AS INTEGER) AS start_day,
    CAST(end_day AS INTEGER) AS end_day,
    CAST(FLOOR((start_day - 1) / 28) + 1 AS INTEGER) AS start_period,
    CAST(FLOOR((end_day - 1) / 28) + 1 AS INTEGER) AS end_period
FROM {{ source('raw', 'campaign_desc') }}
