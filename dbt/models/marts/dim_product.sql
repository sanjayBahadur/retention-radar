SELECT
    product_id,
    manufacturer,
    department,
    brand,
    commodity_desc,
    sub_commodity_desc,
    curr_size_of_product,
    is_private_label
FROM {{ ref('stg_products') }}
