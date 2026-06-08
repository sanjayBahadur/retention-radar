-- Business Question: How well do new customer acquisition cohorts retain over successive 28-day periods?
-- Method: Cohort analysis tracking period-by-period active rates for households acquiring after warm-up (period > 3).

WITH household_cohorts AS (
    SELECT
        household_key,
        first_period_id AS cohort_period
    FROM {{ ref('fct_household_period_state') }}
    WHERE period_id = first_period_id
      AND first_period_id > 3  -- post-warmup cohorts
),

cohort_sizes AS (
    SELECT
        cohort_period,
        COUNT(DISTINCT household_key) AS cohort_size
    FROM household_cohorts
    GROUP BY cohort_period
),

cohort_activity AS (
    SELECT
        c.cohort_period,
        s.period_id - c.cohort_period AS periods_since_acquisition,
        COUNT(DISTINCT CASE WHEN s.trips > 0 THEN s.household_key END) AS active_households,
        ROUND(SUM(s.spend), 2) AS cohort_period_spend
    FROM household_cohorts c
    INNER JOIN {{ ref('fct_household_period_state') }} s ON c.household_key = s.household_key
    WHERE s.period_id >= c.cohort_period
    GROUP BY c.cohort_period, s.period_id - c.cohort_period
)

SELECT
    a.cohort_period,
    cs.cohort_size,
    a.periods_since_acquisition,
    a.active_households,
    ROUND(a.active_households * 100.0 / cs.cohort_size, 2) AS retention_rate_pct,
    a.cohort_period_spend,
    ROUND(a.cohort_period_spend / NULLIF(a.active_households, 0), 2) AS spend_per_active_household
FROM cohort_activity a
INNER JOIN cohort_sizes cs ON a.cohort_period = cs.cohort_period
ORDER BY a.cohort_period, a.periods_since_acquisition;
