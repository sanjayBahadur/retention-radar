-- Basket grain: one row per basket occasion
SELECT
    t.basket_id,
    t.household_key,
    t.day,
    t.period_id,
    t.store_id,
    ROUND(SUM(t.sales_value), 2) AS spend,
    ROUND(SUM(t.gross_sales), 2) AS gross_spend,
    SUM(t.quantity) AS items,
    COUNT(DISTINCT t.product_id) AS distinct_products,
    COUNT(DISTINCT p.department) AS distinct_departments,
    MAX(t.has_coupon) AS coupon_used,
    ROUND(SUM(ABS(t.retail_disc)), 2) AS total_retail_disc,
    ROUND(SUM(ABS(t.coupon_disc)), 2) AS total_coupon_disc,
    MAX(t.trans_time) AS trans_time,
    CASE 
        WHEN MAX(t.trans_time) < 1200 THEN 'Morning (before 12pm)'
        WHEN MAX(t.trans_time) < 1700 THEN 'Afternoon (12pm-5pm)'
        ELSE 'Evening (after 5pm)'
    END AS time_of_day_bucket
FROM {{ ref('stg_transactions') }} t
LEFT JOIN {{ ref('stg_products') }} p ON t.product_id = p.product_id
GROUP BY 
    t.basket_id,
    t.household_key,
    t.day,
    t.period_id,
    t.store_id
