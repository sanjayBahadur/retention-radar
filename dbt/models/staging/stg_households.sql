SELECT
    CAST(household_key AS INTEGER) AS household_key,
    TRIM(age_desc) AS age_desc,
    TRIM(marital_status_code) AS marital_status_code,
    TRIM(income_desc) AS income_desc,
    TRIM(homeowner_desc) AS homeowner_desc,
    TRIM(hh_comp_desc) AS hh_comp_desc,
    TRIM(household_size_desc) AS household_size_desc,
    TRIM(kid_category_desc) AS kid_category_desc
FROM {{ source('raw', 'hh_demographic') }}
