-- Business Question: Which store departments drive the highest item volume, sales value, and coupon discount dependency?
-- Method: Aggregate transaction metrics grouped by department hierarchy, calculating share of total revenue and coupon rate.

WITH dept_metrics AS (
    SELECT
        p.department,
        COUNT(DISTINCT t.basket_id) AS basket_penetration_count,
        SUM(t.quantity) AS total_units_sold,
        ROUND(SUM(t.sales_value), 2) AS total_sales_value,
        ROUND(SUM(t.gross_sales), 2) AS total_gross_sales,
        ROUND(SUM(ABS(t.retail_disc)), 2) AS total_retail_discounts,
        ROUND(SUM(ABS(t.coupon_disc)), 2) AS total_coupon_discounts,
        COUNT(DISTINCT CASE WHEN t.coupon_disc < 0 THEN t.basket_id END) AS baskets_with_coupon
    FROM {{ ref('stg_transactions') }} t
    LEFT JOIN {{ ref('stg_products') }} p ON t.product_id = p.product_id
    GROUP BY p.department
),

totals AS (
    SELECT SUM(total_sales_value) AS store_wide_sales FROM dept_metrics
)

SELECT
    d.department,
    d.basket_penetration_count,
    d.total_units_sold,
    d.total_sales_value,
    ROUND(d.total_sales_value * 100.0 / t.store_wide_sales, 2) AS pct_store_sales,
    d.total_retail_discounts,
    d.total_coupon_discounts,
    ROUND(d.total_coupon_discounts * 100.0 / NULLIF(d.total_sales_value, 0), 2) AS coupon_discount_pct_of_sales,
    ROUND(d.baskets_with_coupon * 100.0 / NULLIF(d.basket_penetration_count, 0), 2) AS coupon_basket_penetration_pct
FROM dept_metrics d
CROSS JOIN totals t
ORDER BY d.total_sales_value DESC;
