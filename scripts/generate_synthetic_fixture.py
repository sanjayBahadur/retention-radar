"""Generate synthetic test fixtures mirroring the schema of Dunnhumby and Hillstrom datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_fixtures(output_dir: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Products (100 synthetic products)
    n_products = 100
    departments = ["GROCERY", "PASTRIES", "PRODUCE", "MEAT", "DRUG GM", "DELI", "SEAFOOD"]
    commodities = [
        "BAKED BREAD/BUNS/ROLLS",
        "FLUID MILK PRODUCTS",
        "CHEESE",
        "BEEF",
        "CHICKEN/POULTRY",
        "DIAPERS & DISPOSABLES",
        "DOG FOODS",
        "SOFT DRINKS",
        "DRY NOODLES/PASTA",
        "PASTA SAUCE",
    ]
    product_ids = [1000 + i for i in range(n_products)]
    products_df = pd.DataFrame(
        {
            "product_id": product_ids,
            "manufacturer": rng.integers(1, 50, size=n_products),
            "department": rng.choice(departments, size=n_products),
            "brand": rng.choice(["National", "Private"], size=n_products),
            "commodity_desc": rng.choice(commodities, size=n_products),
            "sub_commodity_desc": "STANDARD MERCHANDISE",
            "curr_size_of_product": "16 OZ",
        }
    )
    products_df.to_csv(output_dir / "product.csv", index=False)

    # 2. Households & Demographics (50 synthetic households, 20 with demographics)
    n_hhs = 50
    hh_keys = list(range(1, n_hhs + 1))
    demo_keys = hh_keys[:20]
    demo_df = pd.DataFrame(
        {
            "age_desc": rng.choice(
                ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"], size=len(demo_keys)
            ),
            "marital_status_code": rng.choice(["A", "B", "U"], size=len(demo_keys)),
            "income_desc": rng.choice(
                ["Under 15K", "15-24K", "35-49K", "50-74K", "75-99K", "100-124K"],
                size=len(demo_keys),
            ),
            "homeowner_desc": rng.choice(
                ["Homeowner", "Renter", "Probable Renter"], size=len(demo_keys)
            ),
            "hh_comp_desc": rng.choice(
                ["Single Female", "Single Male", "2 Adults", "Family"], size=len(demo_keys)
            ),
            "household_size_desc": rng.choice(["1", "2", "3", "4+"], size=len(demo_keys)),
            "kid_category_desc": rng.choice(["None/Unknown", "1", "2+"], size=len(demo_keys)),
            "household_key": demo_keys,
        }
    )
    demo_df.to_csv(output_dir / "hh_demographic.csv", index=False)

    # 3. Transactions (1,500 transactions across 700 days / 102 weeks)
    n_trans = 1500
    days = rng.integers(1, 701, size=n_trans)
    days.sort()
    trans_df = pd.DataFrame(
        {
            "household_key": rng.choice(hh_keys, size=n_trans),
            "basket_id": np.arange(100000, 100000 + n_trans),
            "day": days,
            "product_id": rng.choice(product_ids, size=n_trans),
            "quantity": rng.integers(1, 5, size=n_trans),
            "sales_value": np.round(rng.uniform(1.50, 45.00, size=n_trans), 2),
            "store_id": rng.choice([101, 102, 103], size=n_trans),
            "retail_disc": np.round(rng.choice([0.0, -0.50, -1.00], size=n_trans), 2),
            "trans_time": rng.integers(800, 2200, size=n_trans),
            "week_no": np.minimum(102, (days // 7) + 1),
            "coupon_disc": 0.0,
            "coupon_match_disc": 0.0,
        }
    )
    trans_df.to_csv(output_dir / "transaction_data.csv", index=False)

    # 4. Campaigns (10 campaigns)
    campaign_ids = list(range(1, 11))
    c_desc_df = pd.DataFrame(
        {
            "description": ["TypeA", "TypeB", "TypeC"] * 3 + ["TypeA"],
            "campaign": campaign_ids,
            "start_day": [i * 60 + 1 for i in range(10)],
            "end_day": [i * 60 + 40 for i in range(10)],
        }
    )
    c_desc_df.to_csv(output_dir / "campaign_desc.csv", index=False)

    # Campaign Table (targeting exposure)
    c_records = []
    for c in campaign_ids:
        targeted = rng.choice(hh_keys, size=rng.integers(10, 30), replace=False)
        for h in targeted:
            c_records.append(
                {
                    "description": c_desc_df.loc[c_desc_df["campaign"] == c, "description"].values[
                        0
                    ],
                    "household_key": h,
                    "campaign": c,
                }
            )
    pd.DataFrame(c_records).to_csv(output_dir / "campaign_table.csv", index=False)

    # 5. Coupons & Redemptions
    coupon_records = []
    for c in campaign_ids:
        c_prods = rng.choice(product_ids, size=5, replace=False)
        for p in c_prods:
            coupon_records.append(
                {"coupon_upc": 10000000000 + c * 1000 + p, "product_id": p, "campaign": c}
            )
    coupon_df = pd.DataFrame(coupon_records)
    coupon_df.to_csv(output_dir / "coupon.csv", index=False)

    # Redemptions
    redempt_df = pd.DataFrame(
        {
            "household_key": rng.choice(hh_keys, size=40),
            "day": rng.integers(50, 650, size=40),
            "coupon_upc": rng.choice(coupon_df["coupon_upc"].values, size=40),
            "campaign": rng.choice(campaign_ids, size=40),
        }
    )
    redempt_df.to_csv(output_dir / "coupon_redempt.csv", index=False)

    # 6. Causal promotions (store display / mailer)
    causal_df = pd.DataFrame(
        {
            "product_id": rng.choice(product_ids, size=200),
            "store_id": rng.choice([101, 102, 103], size=200),
            "week_no": rng.integers(1, 103, size=200),
            "display": rng.choice([0, 1, 2], size=200),
            "mailer": rng.choice([0, 1], size=200),
        }
    )
    causal_df.to_csv(output_dir / "causal_data.csv", index=False)

    # 7. Hillstrom Benchmark (500 synthetic customers)
    n_hill = 500
    hill_df = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, size=n_hill),
            "history_segment": rng.choice(
                ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350+"], size=n_hill
            ),
            "history": np.round(rng.uniform(25.0, 500.0, size=n_hill), 2),
            "mens": rng.choice([0, 1], size=n_hill),
            "womens": rng.choice([0, 1], size=n_hill),
            "zip_code": rng.choice(["Urban", "Suburban", "Rural"], size=n_hill),
            "newbie": rng.choice([0, 1], size=n_hill),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], size=n_hill),
            "segment": rng.choice(["Womens E-Mail", "Mens E-Mail", "No E-Mail"], size=n_hill),
            "visit": rng.choice([0, 1], size=n_hill, p=[0.85, 0.15]),
            "conversion": rng.choice([0, 1], size=n_hill, p=[0.98, 0.02]),
            "spend": 0.0,
        }
    )
    hill_df.loc[hill_df["conversion"] == 1, "spend"] = np.round(
        rng.uniform(20.0, 150.0, size=(hill_df["conversion"] == 1).sum()), 2
    )
    hill_df.to_csv(output_dir / "hillstrom.csv", index=False)


if __name__ == "__main__":
    out = Path("tests/fixtures/sample")
    generate_synthetic_fixtures(out)
    print(f"Synthetic fixtures successfully generated in {out}")
