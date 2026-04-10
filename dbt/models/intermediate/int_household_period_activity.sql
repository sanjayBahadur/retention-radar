-- Dense household x period activity grid from first observed purchase period through period 25

WITH household_first_period AS (
    SELECT
        household_key,
        MIN(period_id) AS first_period_id
    FROM {{ ref('stg_transactions') }}
    GROUP BY household_key
),

periods AS (
    SELECT UNNEST(GENERATE_SERIES(1, 25)) AS period_id
),

dense_grid AS (
    SELECT
        h.household_key,
        p.period_id,
        h.first_period_id
    FROM household_first_period h
    CROSS JOIN periods p
    WHERE p.period_id >= h.first_period_id
),

period_aggregates AS (
    SELECT
        household_key,
        period_id,
        COUNT(DISTINCT basket_id) AS trips,
        ROUND(SUM(spend), 2) AS spend,
        ROUND(SUM(gross_spend), 2) AS gross_spend,
        SUM(items) AS items,
        MAX(coupon_used) AS coupon_used
    FROM {{ ref('int_baskets') }}
    GROUP BY household_key, period_id
),

joined_grid AS (
    SELECT
        g.household_key,
        g.period_id,
        g.first_period_id,
        COALESCE(a.trips, 0) AS trips,
        COALESCE(a.spend, 0.0) AS spend,
        COALESCE(a.gross_spend, 0.0) AS gross_spend,
        COALESCE(a.items, 0) AS items,
        COALESCE(a.coupon_used, 0) AS coupon_used
    FROM dense_grid g
    LEFT JOIN period_aggregates a
        ON g.household_key = a.household_key 
        AND g.period_id = a.period_id
),

with_grouping AS (
    SELECT
        household_key,
        period_id,
        first_period_id,
        trips,
        spend,
        gross_spend,
        items,
        coupon_used,
        SUM(CASE WHEN trips > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY household_key 
            ORDER BY period_id
        ) AS purchase_epoch
    FROM joined_grid
)

SELECT
    household_key,
    period_id,
    first_period_id,
    trips,
    spend,
    gross_spend,
    items,
    coupon_used,
    CASE 
        WHEN trips > 0 THEN 0
        ELSE ROW_NUMBER() OVER (
            PARTITION BY household_key, purchase_epoch 
            ORDER BY period_id
        ) - 1
    END AS consecutive_missed_periods,
    ROUND(SUM(spend) OVER (
        PARTITION BY household_key 
        ORDER BY period_id 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS trailing_3period_spend
FROM with_grouping
