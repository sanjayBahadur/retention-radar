-- Lifecycle state assignment for each household at the end of each 28-day period (1 to 25)
-- Champion threshold uses only the current period, available at each scoring cutoff.

WITH champion_cutoff AS (
    SELECT
        period_id,
        QUANTILE_CONT(trailing_3period_spend, 0.75) AS spend_p75
    FROM {{ ref('int_household_period_activity') }}
    WHERE trips > 0
    GROUP BY period_id
),

assigned_states AS (
    SELECT
        a.household_key,
        a.period_id,
        a.trips,
        a.spend,
        a.gross_spend,
        a.items,
        a.coupon_used,
        a.consecutive_missed_periods,
        a.trailing_3period_spend,
        a.first_period_id,
        c.spend_p75 AS champion_threshold,
        CASE
            -- Active in period t (trips > 0)
            WHEN a.trips > 0 THEN
                CASE
                    -- New: First-ever purchase falls in period t, and t > warm-up (periods 1-3)
                    WHEN a.period_id = a.first_period_id AND a.period_id > 3 THEN 'New'
                    -- Champion: Top 25% trailing 3-period spend
                    WHEN a.trailing_3period_spend >= c.spend_p75 THEN 'Champion'
                    -- Regular: Active, neither New nor Champion
                    ELSE 'Regular'
                END
            -- Inactive in period t (trips = 0)
            ELSE
                CASE
                    -- At-Risk: 1 missed period (purchased in t-1)
                    WHEN a.consecutive_missed_periods = 1 THEN 'At-Risk'
                    -- Dormant: 2 to 3 consecutive missed periods
                    WHEN a.consecutive_missed_periods BETWEEN 2 AND 3 THEN 'Dormant'
                    -- Lapsed: 4 or more consecutive missed periods
                    ELSE 'Lapsed'
                END
        END AS state
    FROM {{ ref('int_household_period_activity') }} a
    JOIN champion_cutoff c ON a.period_id = c.period_id
)

SELECT
    household_key,
    period_id,
    trips,
    spend,
    gross_spend,
    items,
    coupon_used,
    consecutive_missed_periods,
    trailing_3period_spend,
    first_period_id,
    champion_threshold,
    state,
    LAG(state) OVER (PARTITION BY household_key ORDER BY period_id) AS prior_state,
    LEAD(state) OVER (PARTITION BY household_key ORDER BY period_id) AS next_state
FROM assigned_states
