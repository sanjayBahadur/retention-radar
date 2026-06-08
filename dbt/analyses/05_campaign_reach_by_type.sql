-- Business Question: How many unique households are reached by each direct marketing campaign type, and what is their redemption rate?
-- Method: Join campaign exposures to redeemed coupons to calculate conversion reach by strategy (Type A, B, C).

WITH campaign_targets AS (
    SELECT
        e.campaign_type,
        COUNT(DISTINCT e.campaign) AS distinct_campaigns,
        COUNT(DISTINCT e.household_key) AS targeted_households,
        COUNT(*) AS total_targeting_instances
    FROM {{ ref('fct_campaign_exposure') }} e
    GROUP BY e.campaign_type
),

campaign_redemptions AS (
    SELECT
        c.campaign_type,
        COUNT(DISTINCT c.household_key) AS converting_households,
        COUNT(DISTINCT c.basket_id) AS conversion_baskets,
        ROUND(SUM(c.sales_value), 2) AS attributed_sales,
        ROUND(SUM(ABS(c.coupon_disc)), 2) AS total_coupon_cost
    FROM {{ ref('int_coupon_transactions') }} c
    GROUP BY c.campaign_type
)

SELECT
    t.campaign_type,
    t.distinct_campaigns,
    t.targeted_households,
    t.total_targeting_instances,
    COALESCE(r.converting_households, 0) AS converting_households,
    ROUND(COALESCE(r.converting_households, 0) * 100.0 / NULLIF(t.targeted_households, 0), 2) AS household_conversion_rate_pct,
    COALESCE(r.conversion_baskets, 0) AS conversion_baskets,
    COALESCE(r.attributed_sales, 0.0) AS attributed_sales,
    COALESCE(r.total_coupon_cost, 0.0) AS total_coupon_cost
FROM campaign_targets t
LEFT JOIN campaign_redemptions r ON t.campaign_type = r.campaign_type
ORDER BY t.targeted_households DESC;
