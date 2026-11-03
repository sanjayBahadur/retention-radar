# Customer value: a 26-week spending forecast

This model estimates how much a household may spend over the next 26 weeks.
The dashboard converts that forecast into estimated margin using explicit
business assumptions. These are forecasts, not money earned by a campaign.

## What was compared?

- **Recent spending:** assume the next 26 weeks resemble the previous 26 weeks.
- **BG/NBD + Gamma-Gamma:** estimate shopping frequency and spending per visit.
- **LightGBM:** learn spending patterns from historical household features.

The recent-spending baseline had the lowest mean absolute error in this test,
so it supplies the final day-700 customer-value estimates. A simple baseline
is useful when added model complexity does not improve the decision.

The Markov lifecycle model remains a separate long-run analysis. Its
infinite-horizon values are not compared with a 26-week spending target.

## How was it evaluated?

All models forecast **182 days (26 weeks)**. Features use the previous 182
days, plus tenure and days since the last purchase. A purchase occasion is
a shopping day: item lines and baskets on the same day are added together.
Gamma-Gamma averages spending on repeat occasions, excluding the first one.

| Window | Feature cutoff | Future spending label |
|---|---:|---|
| LightGBM training | Day 336 | Days 337–518 |
| Evaluation | Day 518 | Days 519–700 |
| Latest forecast | Day 700 | Next 182 days; not observed here |

Training outcomes end before evaluation outcomes begin. The same households
can appear in both windows: this tests future forecasts for known customers,
not predictions for entirely new customers. Probabilistic model parameters
are fit using purchase history available by day 518.

## Results

Evaluation covers **2,498 households** that were already observed by day 518.
Errors below are in **sales dollars**, before any assumed margin or discount.

| Method | Rank correlation | Mean absolute error | Root mean squared error |
|---|---:|---:|---:|
| Recent 26-week spend | 0.8179 | $339.20 | $529.10 |
| BG/NBD + Gamma-Gamma | 0.8086 | $351.74 | $544.92 |
| LightGBM | 0.8112 | $355.99 | $549.55 |

Rank correlation measures whether households are ordered correctly by future
spending; a high rank score does not guarantee accurate individual amounts.
The evaluation selects the lowest-error method on one future window. This
is an exploratory comparison, without a second untouched selection holdout
or a claim that the small differences are statistically significant.

## How estimates enter the dashboard

Estimated margin = forecast sales × assumed 25% gross margin × (1.10)^−0.5.
The 10% annual discount rate and 25% margin are configurable assumptions,
not measured household profitability. The mean day-700 estimate is **$223.31**.

The historical churn roster uses features through day 644 and a prespecified
LightGBM model trained on outcomes ending at day 518. It does not use the
winner selected with later day-700 outcomes. The file records its cutoff.

## Limits and reproducibility

The retailer's data does not reveal competitor purchases, customer profit,
or whether outreach changes spending. BG/NBD assumes stable purchase and
dropout behavior; Gamma-Gamma assumes spending and frequency independence.
This implementation constrains BG/NBD a > 1 and Gamma-Gamma q > 1 for finite,
positive expectations. Returns are netted within each shopping day; only
positive-spend days enter the purchase-frequency features.

Run `python -m retention_radar.clv.compare` to regenerate the comparison,
per-household validation predictions, summary metadata, latest forecasts,
and historical forecasts. Artifacts are in `data/outputs/clv*`.

Formula references: Fader, Hardie & Lee (2005), [BG/NBD notes](http://brucehardie.com/notes/004/)
and [Gamma-Gamma notes](http://www.brucehardie.com/notes/025/).
