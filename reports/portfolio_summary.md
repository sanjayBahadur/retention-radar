# Retention Radar: Portfolio Summary & Public Announcement

**Author:** Sanjay Bahadur (`mallasanjay2099@gmail.com`)  
**Project Repository:** `github.com/sanjayBahadur/retention-radar`  

---

## 1. Resume Bullet Points (Tailored for Staff / Senior Data Scientist Roles)

- **End-to-End Retention Analytics Architecture:** Architected and deployed **Retention Radar**, an enterprise customer retention platform analyzing 2.6M grocery transactions across 2,500 households using DuckDB, dbt-core, and Python 3.12, uncovering an 8-week precursor category departure signal (fresh meats -76%, baby care -66%) that predicts household lapse prior to basket cessation.
- **Causal Machine Learning & Uplift Modeling:** Developed a causal uplift pipeline using CausalML and doubly robust estimation (AIPW) benchmarked against randomized RCTs; identified 202 "Sleeping Dogs" (-$0.60/day negative uplift), designed a Top-30% targeting policy generating **3.61x ROI** ($20.4k net incremental profit), and parameterized an 8-week randomized A/B test design ($N = 454$).
- **Predictive Churn & Decision Optimization:** Built an Optuna-tuned LightGBM churn model with isotonic calibration achieving **0.3520 Test PR-AUC** (4.82x baseline lift, capturing 72.2% of churners in the top 20%) and an out-of-time Tweedie CLV regressor (0.9162 Spearman rank correlation); demonstrated via demographic ablation that pure behavioral modeling boosts PR-AUC by +0.0107 while eliminating proxy bias from 68% missing loyalty records.

---

## 2. LinkedIn Announcement Post

```markdown
🚀 Excited to open-source **Retention Radar**: an end-to-end customer retention and causal uplift intelligence system designed for retail and grocery enterprises!

Most customer retention strategies in retail suffer from two fundamental flaws:
1. They rely on blunt heuristic rules (e.g. "inactive for 30 days"), detecting churn only after the customer has already defected.
2. They blast mass discounts to everyone, eroding margins on customers who would shop anyway while irritating customers who dislike promotional spam.

To solve this, I built **Retention Radar** from scratch on 2.6 million Dunnhumby transactions across 2,500 households over two years.

Here are the key technical and business insights:

🔍 **1. Precursor Category Abandonment (The 8-Week Runway):**
Customers don't churn abruptly. Long before their basket size drops to zero, they strip high-involvement categories—fresh butcher meats (chicken -76%, beef -64%) and family essentials (diapers -66%, dog food -74%). They switch primary store loyalty for butcher cuts while continuing to buy milk and bread at their old store for 6–8 weeks. Tracking category share departure provides a leading indicator before traditional churn metrics even notice.

⚠️ **2. Causal Uplift & The "Sleeping Dogs" Guardrail:**
Using CausalML (S-Learner, X-Learner, AIPW) validated against randomized retail RCT benchmarks, we identified that 2.1% of households are "Sleeping Dogs"—customers who exhibit statistically negative treatment effects (-$0.60/day). Marketing contact causes friction and churn. By hard-suppressing Sleeping Dogs and focusing on Persuadables, our Top-30% targeting policy delivers **3.61x ROI** ($20,435 net incremental profit).

🎯 **3. Calibrated ML & Demographic Fairness:**
Our Optuna-tuned LightGBM model delivers a **0.3520 Test PR-AUC** (4.82x baseline lift) with an Expected Calibration Error under 1% (ECE = 0.0090), capturing 72.2% of all future churners in the top 20% outreach bracket. Furthermore, demographic ablation proved that removing loyalty card demographics improved PR-AUC by +0.0107 while eliminating proxy disparities from 68% missing demographic cards.

📊 **Production-Grade Stack:**
- **Data Engineering:** DuckDB + dbt-core staging, intermediate, and marts with 29 automated data tests.
- **Data Integrity & Drift:** 16 strict Pandera runtime schemas and Population Stability Index (PSI) drift monitoring.
- **Consumption Layer:** Interactive 6-page Streamlit Command Center + Tableau / Power BI starter package with 10 production DAX and Tableau calculated fields.
- **Testing & CI:** 57 unit and integration tests passing in under 30 seconds with 82% coverage.

Check out the full repository, technical report, model cards, and interactive dashboard:
👉 [https://github.com/sanjayBahadur/retention-radar]

#DataScience #MachineLearning #CausalInference #CustomerRetention #Python #dbt #DuckDB #Streamlit #PowerBI #RetailAnalytics
```
