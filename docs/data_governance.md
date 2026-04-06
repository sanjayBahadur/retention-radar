# Data Governance & Privacy Note

## 1. Data Sources and Licensing Terms

| Source | Role | Licensing & Redistribution Restrictions |
| :--- | :--- | :--- |
| **Dunnhumby The Complete Journey** | Core household transactions, promotions, demographics | Non-commercial, research, and educational use only. **Strictly forbids redistribution of raw transaction or customer-level records.** |
| **X5 RetailHero Uplift** | Randomized campaign ground truth (Part A & C uplift) | Released for competition and academic research; cite X5 Retail Group; no redistribution of raw customer records. |
| **Hillstrom MineThatData** | Randomized 3-arm campaign test fixture | Public benchmark data; cite Kevin Hillstrom / MineThatData. |

## 2. Anonymization & Aggregation Policy
- **Never commit `data/`:** The `.gitignore` excludes `data/raw/`, `data/warehouse.duckdb`, and large Parquet tables.
- **Aggregated Outputs:** Any public portfolio artifacts (such as Streamlit dashboards or Power BI workbooks) present only aggregated statistics, cohort curves, and anonymized/re-keyed IDs where necessary.
- **No PII Exposure:** No customer names, phone numbers, or addresses exist in these datasets. Household identifiers are synthetic numerical keys (`household_key`).

## 3. Production & Enterprise Governance Considerations
In an enterprise retail deployment with real consumer data:
1. **PII and PCI-DSS:** Transaction records containing payment details must be tokenized; customer identities must reside in restricted Customer Data Platforms (CDP) governed by RBAC.
2. **Consent & Marketing Opt-Outs:** Customers opting out under GDPR/CCPA/state privacy legislation must be excluded dynamically from downstream campaign targeting tables.
3. **Retention & Deletion Policy:** Transaction data aged beyond statutory audit requirements must be anonymized or purged.
4. **Algorithmic Fairness:** Predictive churn and uplift models must be monitored continuously for disparate impact across demographic segments (age, income, household composition).
