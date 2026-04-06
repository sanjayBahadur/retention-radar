# Retention Radar Data Dictionary

Profiling summary across all raw ingested retail datasets.

### `raw.campaign_desc` (30 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `description` | VARCHAR | 3 | 0.0% | TypeA | TypeC | Campaign strategy classification (Type A, Type B, Type C) |
| `campaign` | BIGINT | 30 | 0.0% | 1 | 9 | Numeric identifier for the direct mail campaign |
| `start_day` | BIGINT | 27 | 0.0% | 224 | 659 | Calendar day number when the campaign mailings began |
| `end_day` | BIGINT | 28 | 0.0% | 264 | 719 | Calendar day number when the campaign mailings concluded |

### `raw.campaign_table` (7,208 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `description` | VARCHAR | 3 | 0.0% | TypeA | TypeC | Campaign strategy type |
| `household_key` | BIGINT | 1,584 | 0.0% | 1 | 999 | Household targeted for campaign communication |
| `campaign` | BIGINT | 30 | 0.0% | 1 | 9 | Numeric identifier for the campaign |

### `raw.causal_data` (36,786,524 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `product_id` | BIGINT | 68,377 | 0.0% | 1000050 | 999992 | Product identifier |
| `store_id` | BIGINT | 115 | 0.0% | 286 | 450 | Store location identifier |
| `week_no` | BIGINT | 93 | 0.0% | 10 | 99 | Week number of promotional activity |
| `display` | VARCHAR | 10 | 0.0% | 0 | A | In-store display location placement flag (0, 1, 2, etc.) |
| `mailer` | VARCHAR | 11 | 0.0% | 0 | Z | Circular / weekly mailer feature page placement flag (0, A, B, etc.) |

### `raw.coupon` (124,548 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `coupon_upc` | BIGINT | 1,135 | 0.0% | 10000085189 | 59986600074 | Standard universal product code representing the coupon |
| `product_id` | BIGINT | 44,133 | 0.0% | 1000050 | 999975 | Product ID for which coupon is valid |
| `campaign` | BIGINT | 30 | 0.0% | 1 | 9 | Campaign under which coupon was distributed |

### `raw.coupon_redempt` (2,318 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `household_key` | BIGINT | 434 | 0.0% | 1 | 998 | Household that redeemed the coupon |
| `day` | BIGINT | 328 | 0.0% | 225 | 704 | Day number when the coupon was redeemed |
| `coupon_upc` | BIGINT | 556 | 0.0% | 10000085320 | 58978500076 | UPC code of the redeemed coupon |
| `campaign` | BIGINT | 30 | 0.0% | 1 | 9 | Campaign under which redeemed coupon was issued |

### `raw.hh_demographic` (801 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `age_desc` | VARCHAR | 6 | 0.0% | 19-24 | 65+ | Estimated age band of primary shopper (e.g. 35-44, 45-54) |
| `marital_status_code` | VARCHAR | 3 | 0.0% | A | U | Marital status code (A = Married, B = Single, U = Unknown) |
| `income_desc` | VARCHAR | 12 | 0.0% | 100-124K | Under 15K | Annual household income bracket (e.g. 35-49K, 50-74K) |
| `homeowner_desc` | VARCHAR | 5 | 0.0% | Homeowner | Unknown | Home ownership status (Homeowner, Renter, Probable) |
| `hh_comp_desc` | VARCHAR | 6 | 0.0% | 1 Adult Kids | Unknown | Household composition classification |
| `household_size_desc` | VARCHAR | 5 | 0.0% | 1 | 5+ | Number of individuals residing in household (1, 2, 3, 4, 5+) |
| `kid_category_desc` | VARCHAR | 4 | 0.0% | 1 | None/Unknown | Presence and number of children in household |
| `household_key` | BIGINT | 801 | 0.0% | 1 | 997 | Household identifier linking to transactions |

### `raw.product` (92,353 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `product_id` | BIGINT | 92,353 | 0.0% | 1000002 | 999992 | Uniquely identifies each product item |
| `manufacturer` | BIGINT | 6,476 | 0.0% | 1 | 999 | Numeric ID representing the manufacturing brand owner |
| `department` | VARCHAR | 44 | 0.0% |   | VIDEO RENTAL | High-level retail category department (e.g. GROCERY, PASTRIES) |
| `brand` | VARCHAR | 2 | 0.0% | National | Private | Indicator whether brand is Private Label or National |
| `commodity_desc` | VARCHAR | 308 | 0.0% |   | YOGURT | Commodity category group (e.g. MILK, SOFT DRINKS, BREAD) |
| `sub_commodity_desc` | VARCHAR | 2,383 | 0.0% |   | YOGURT NOT MULTI-PAC... | Granular category classification |
| `curr_size_of_product` | VARCHAR | 4,345 | 0.0% |   | XLRG | Package size string (e.g. 16 OZ, 1 GAL) |

### `raw.transaction_data` (2,595,732 rows)

| Column | Type | Distinct | Null % | Min | Max | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `household_key` | BIGINT | 2,500 | 0.0% | 1 | 999 | Uniquely identifies each frequent-shopper household |
| `basket_id` | BIGINT | 276,484 | 0.0% | 26984851472 | 42305362535 | Uniquely identifies each shopping trip / transaction occasion |
| `day` | BIGINT | 711 | 0.0% | 1 | 99 | Day number of the transaction (day 1 to 711 across 102 weeks) |
| `product_id` | BIGINT | 92,339 | 0.0% | 1000002 | 999992 | Uniquely identifies each purchased retail product UPC |
| `quantity` | BIGINT | 11,746 | 0.0% | 0 | 9997 | Number of units purchased in this transaction line |
| `sales_value` | DOUBLE | 5,502 | 0.0% | 0.0 | 99.99 | Amount of dollars paid by customer after all discounts applied |
| `store_id` | BIGINT | 582 | 0.0% | 1 | 984 | Uniquely identifies the physical grocery retail store location |
| `retail_disc` | DOUBLE | 2,414 | 0.0% | -0.01 | 6.938894e-18 | Loyalty card / price reduction discount applied (negative float) |
| `trans_time` | VARCHAR | 1,440 | 0.0% | 0000 | 2359 | Time of day of the purchase (HHMM format) |
| `week_no` | BIGINT | 102 | 0.0% | 1 | 99 | Week number of the transaction (week 1 to 102) |
| `coupon_disc` | DOUBLE | 415 | 0.0% | -0.08 | 0.0 | Manufacturer coupon discount applied (negative float) |
| `coupon_match_disc` | DOUBLE | 80 | 0.0% | -0.01 | 0.0 | Retailer match on manufacturer coupon discount (negative float) |
