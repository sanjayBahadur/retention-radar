-- Business Question: How is total grocery revenue trending across 28-day periods, and what is the period-over-period growth rate in margin dollars?
-- Method: Window functions comparing current period revenue and estimated margin to prior period.

WITH period_revenue AS (
    SELECT
        b.period_id,
        COUNT(DISTINCT b.household_key) AS active_households,
        COUNT(DISTINCT b.basket_id) AS total_baskets,
        ROUND(SUM(b.spend), 2) AS total_revenue,
        ROUND(SUM(b.spend * 0.25), 2) AS gross_margin_dollars  -- 0.25 assumption
    FROM {{ ref('fct_baskets') }} b
    GROUP BY b.period_id
)

SELECT
    period_id,
    active_households,
    total_baskets,
    total_revenue,
    gross_margin_dollars,
    LAG(total_revenue) OVER (ORDER BY period_id) AS prior_period_revenue,
    ROUND(
        (total_revenue - LAG(total_revenue) OVER (ORDER BY period_id)) * 100.0 / 
        NULLIF(LAG(total_revenue) OVER (ORDER BY period_id), 0), 
        2
    ) AS pct_revenue_change,
    ROUND(
        gross_margin_dollars - LAG(gross_margin_dollars) OVER (ORDER BY period_id),
        2
    ) AS delta_margin_dollars
FROM period_revenue
ORDER BY period_id;
