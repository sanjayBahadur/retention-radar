SELECT
    CAST(coupon_upc AS BIGINT) AS coupon_upc,
    CAST(product_id AS INTEGER) AS product_id,
    CAST(campaign AS INTEGER) AS campaign
FROM {{ source('raw', 'coupon') }}
