-- Singular test: assert lifecycle state is completely consistent with missed-period counts and activity
-- Returns rows that violate consistency rules (failing if any row returned)

SELECT *
FROM {{ ref('fct_household_period_state') }}
WHERE 
    -- Active states must have trips > 0 and missed periods = 0
    (state IN ('New', 'Champion', 'Regular') AND (trips <= 0 OR consecutive_missed_periods != 0))
    -- At-Risk must have exactly 1 missed period
    OR (state = 'At-Risk' AND consecutive_missed_periods != 1)
    -- Dormant must have 2 or 3 missed periods
    OR (state = 'Dormant' AND consecutive_missed_periods NOT IN (2, 3))
    -- Lapsed must have 4 or more missed periods
    OR (state = 'Lapsed' AND consecutive_missed_periods < 4)
