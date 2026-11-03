# Analytical decisions

These notes explain the implemented choices. They are not a record of external approvals or a claim of deployment.

| Choice | Reason and tradeoff |
| --- | --- |
| Keep one central question: prioritizing limited retention outreach | Gives the dashboard a practical thread while retaining supporting analyses |
| Aggregate transaction lines before measuring trips or monetary value | Purchase frequency and spend must describe the same purchase occasion |
| Use 28-day lifecycle periods | Gives consistent windows; does not follow calendar months |
| Label Lapsed after four missed periods | Matches the configured lifecycle table; observed returns remain possible |
| Compute the Champion threshold within each current period | Uses information available at the scoring cutoff rather than a threshold fitted on future periods |
| Evaluate churn on later cutoffs with nonoverlapping label windows | Separates model selection from later outcomes; repeated households still require clustered uncertainty |
| Compare simple models alongside LightGBM | Extra complexity needs practical benefit, not just a slightly better headline metric |
| Evaluate value forecasts on nonoverlapping 26-week outcome windows | Prevents evaluating a model on the labels used to fit it; errors remain in sales dollars |
| Use cutoff-correct value estimates in historical scoring | Prevents future spending features from influencing a past decision |
| Evaluate campaign models across held-out household groups | Keeps a household out of its own training fold, while retaining campaign observations for evaluation |
| Use unique households for a single-contact scenario | Repeated campaign observations must not become duplicate outreach contacts |
| Generate headline metrics from artifacts | Prevents drift between saved evaluations, dashboard copy, and financial scenarios |
| Separate observed, predicted, and modeled quantities | Readers can judge what the data establishes and what requires an experiment |
| Treat uplift segments as exploratory | Historical selection and unmeasured factors prevent automatic causal conclusions |
| Keep financial assumptions visible | Margin, cost, and save rate are inputs to a scenario, not observed campaign gains |
| Present methods beneath a short guided explanation | Newcomers can follow the question; technical readers can inspect the details |

A randomized outreach test is the proposed next evidence. It has not been conducted by this project; the [experiment brief](../reports/executive_summary.md) describes the design.
