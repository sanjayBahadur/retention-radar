# Retention Radar: findings and proposed test

Purchase history helped concentrate departures into a smaller review list. Across four later test periods, selecting the highest-risk 20% at each cutoff identified **534 of 712 departures (75%)**. A departure means no purchases in the following eight weeks. The evaluation covers 9,397 household observations; some households appear at several cutoffs.

Two other results helped shape the recommendation:

- **A simple value forecast held up well.** Reusing the previous 26 weeks of spending produced an average absolute error of **$339 per household**, compared with **$356 for LightGBM**, when forecasting the next 26 weeks of sales.
- **Shopping changes differed by category.** Around dormancy, diapers and poultry lost **3.61 and 3.33 percentage points** of spending share relative to the active comparison group. These observations give an analyst specific categories to investigate.

The priority lists also included **1,344 cases where shopping continued**, while **178 departures fell outside them**. That tradeoff matters when deciding how much outreach to fund. The [evaluation review](evaluation.md) breaks it down by period and customer state.

## Outreach decision

Use the ranking to focus a small offer experiment. The dashboard's estimated-margin scenario lets a reviewer vary the save rate while keeping contact costs visible. Historical campaign models offer a separate, exploratory response ranking for choosing a test population.

## Proposed experiment

Freeze the highest 30% of the response ranking and randomly assign each eligible household to receive one defined offer or no study offer. Keep that assignment for eight weeks and record other promotions in both groups.

Measure average daily spend per assigned household as the primary outcome. Also track contribution after offer costs, continued activity, complaints, and opt-outs. Set the minimum worthwhile improvement, analysis, and rollout criteria before starting.

For an assumed detectable increase of **$0.50 per day**, the current planning calculation requires **5,340 households**, compared with **739 available** in the selected pool. A larger population or a revised pilot design would be needed for that target.

[Explore the dashboard](https://sanjaybahadur.github.io/retention-radar/) for the charts and scenarios. [Technical notes](technical_report.md) explain the evaluation, financial assumptions, and interpretation of historical campaigns; [generated results](../data/outputs/results.json) contain the current values.
