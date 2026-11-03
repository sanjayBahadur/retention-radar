SELECT
    CAST(product_id AS INTEGER) AS product_id,
    CAST(store_id AS INTEGER) AS store_id,
    CAST(week_no AS INTEGER) AS week_no,
    TRIM(CAST(display AS VARCHAR)) AS display,
    TRIM(CAST(mailer AS VARCHAR)) AS mailer,
    CASE WHEN display IS NOT NULL AND TRIM(CAST(display AS VARCHAR)) != '0' THEN 1 ELSE 0 END AS has_display,
    CASE WHEN mailer IS NOT NULL AND TRIM(CAST(mailer AS VARCHAR)) != '0' THEN 1 ELSE 0 END AS has_mailer
FROM {{ source('raw', 'causal_data') }}
WHERE week_no <= 100
