WITH tx_summary AS (
    SELECT
        household_key,
        MIN(day) AS first_purchase_day,
        MAX(day) AS last_purchase_day,
        MAX(day) - MIN(day) + 1 AS tenure_days,
        COUNT(DISTINCT basket_id) AS total_baskets,
        ROUND(SUM(sales_value), 2) AS lifetime_spend,
        SUM(quantity) AS lifetime_items
    FROM {{ ref('stg_transactions') }}
    GROUP BY household_key
)

SELECT
    t.household_key,
    t.first_purchase_day,
    t.last_purchase_day,
    t.tenure_days,
    t.total_baskets,
    t.lifetime_spend,
    t.lifetime_items,
    CASE WHEN d.household_key IS NOT NULL THEN 1 ELSE 0 END AS has_demographics,
    d.age_desc,
    d.marital_status_code,
    d.income_desc,
    d.homeowner_desc,
    d.hh_comp_desc,
    d.household_size_desc,
    d.kid_category_desc
FROM tx_summary t
LEFT JOIN {{ ref('stg_households') }} d ON t.household_key = d.household_key
