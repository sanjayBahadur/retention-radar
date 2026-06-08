-- Business Question: Which households experienced a severe (>50%) quarter-over-quarter shopping trip velocity drop?
-- Method: Group 28-day periods into retail quarters (3 periods per quarter = 84 days) and calculate quarter-over-quarter drop.

WITH quarterly_activity AS (
    SELECT
        household_key,
        CAST(FLOOR((period_id - 1) / 3) + 1 AS INTEGER) AS retail_quarter,
        SUM(trips) AS quarter_trips,
        ROUND(SUM(spend), 2) AS quarter_spend
    FROM {{ ref('fct_household_period_state') }}
    WHERE period_id <= 24  -- 8 full 3-period quarters
    GROUP BY household_key, CAST(FLOOR((period_id - 1) / 3) + 1 AS INTEGER)
),

velocity_trends AS (
    SELECT
        household_key,
        retail_quarter,
        quarter_trips,
        quarter_spend,
        LAG(quarter_trips) OVER (PARTITION BY household_key ORDER BY retail_quarter) AS prior_quarter_trips,
        LAG(quarter_spend) OVER (PARTITION BY household_key ORDER BY retail_quarter) AS prior_quarter_spend
    FROM quarterly_activity
)

SELECT
    household_key,
    retail_quarter,
    prior_quarter_trips,
    quarter_trips,
    ROUND((quarter_trips - prior_quarter_trips) * 100.0 / NULLIF(prior_quarter_trips, 0), 2) AS pct_trip_change,
    prior_quarter_spend,
    quarter_spend,
    ROUND(quarter_spend - prior_quarter_spend, 2) AS delta_spend
FROM velocity_trends
WHERE prior_quarter_trips >= 5  -- filter to historically established shoppers
  AND (quarter_trips - prior_quarter_trips) * 1.0 / prior_quarter_trips <= -0.50  -- fell > 50%
ORDER BY retail_quarter, pct_trip_change ASC;
