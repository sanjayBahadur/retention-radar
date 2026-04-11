SELECT
    basket_id,
    household_key,
    day,
    period_id,
    store_id,
    spend,
    gross_spend,
    items,
    distinct_products,
    distinct_departments,
    coupon_used,
    total_retail_disc,
    total_coupon_disc,
    trans_time,
    time_of_day_bucket
FROM {{ ref('int_baskets') }}
