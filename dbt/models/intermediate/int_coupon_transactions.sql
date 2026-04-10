-- Complete coupon redemption source:
-- transaction lines with coupon_disc < 0 joined to coupons and campaign windows
SELECT
    t.household_key,
    t.basket_id,
    t.day,
    t.period_id,
    t.product_id,
    t.sales_value,
    t.coupon_disc,
    c.coupon_upc,
    c.campaign,
    cd.campaign_type
FROM {{ ref('stg_transactions') }} t
INNER JOIN {{ ref('stg_coupons') }} c ON t.product_id = c.product_id
INNER JOIN {{ ref('stg_campaigns') }} cd 
    ON c.campaign = cd.campaign
    AND t.day BETWEEN cd.start_day AND cd.end_day
WHERE t.coupon_disc < 0
