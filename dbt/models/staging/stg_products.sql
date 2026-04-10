SELECT
    CAST(product_id AS INTEGER) AS product_id,
    CAST(manufacturer AS INTEGER) AS manufacturer,
    UPPER(TRIM(department)) AS department,
    UPPER(TRIM(brand)) AS brand,
    UPPER(TRIM(commodity_desc)) AS commodity_desc,
    UPPER(TRIM(sub_commodity_desc)) AS sub_commodity_desc,
    TRIM(curr_size_of_product) AS curr_size_of_product,
    CASE WHEN UPPER(TRIM(brand)) = 'PRIVATE' THEN 1 ELSE 0 END AS is_private_label
FROM {{ source('raw', 'product') }}
