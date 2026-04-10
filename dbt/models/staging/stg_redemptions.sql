SELECT
    CAST(household_key AS INTEGER) AS household_key,
    CAST(day AS INTEGER) AS day,
    CAST(FLOOR((day - 1) / 28) + 1 AS INTEGER) AS period_id,
    CAST(coupon_upc AS BIGINT) AS coupon_upc,
    CAST(campaign AS INTEGER) AS campaign
FROM {{ source('raw', 'coupon_redempt') }}
WHERE day <= 700
