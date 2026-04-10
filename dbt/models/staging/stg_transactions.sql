-- Staging model for transactions
-- Drops days 701-711 as partial period (giving 25 full 28-day periods)
-- Gross sales definition: sales_value - retail_disc - coupon_match_disc (discounts are negative)

SELECT
    CAST(household_key AS INTEGER) AS household_key,
    CAST(basket_id AS BIGINT) AS basket_id,
    CAST(day AS INTEGER) AS day,
    CAST(FLOOR((day - 1) / 28) + 1 AS INTEGER) AS period_id,
    CAST(product_id AS INTEGER) AS product_id,
    CAST(quantity AS INTEGER) AS quantity,
    CAST(sales_value AS DOUBLE) AS sales_value,
    CAST(store_id AS INTEGER) AS store_id,
    CAST(retail_disc AS DOUBLE) AS retail_disc,
    CAST(trans_time AS INTEGER) AS trans_time,
    CAST(week_no AS INTEGER) AS week_no,
    CAST(coupon_disc AS DOUBLE) AS coupon_disc,
    CAST(coupon_match_disc AS DOUBLE) AS coupon_match_disc,
    CAST(sales_value - retail_disc - coupon_match_disc AS DOUBLE) AS gross_sales,
    CASE WHEN coupon_disc < 0 THEN 1 ELSE 0 END AS has_coupon
FROM {{ source('raw', 'transaction_data') }}
WHERE day <= 700
