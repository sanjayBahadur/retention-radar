# Retention Radar: findings and next decision

**Question:** How should a grocery retailer prioritize limited retention outreach?

The project follows purchase histories for 2,500 households. It combines a description of changing shopping behavior with predictions of future inactivity and an exploration of historical campaign response.

## What the evidence supports

**Shopping patterns change as households become inactive.** Category-share comparisons highlight where spending changes differ from an active comparison group. A share change is measured in percentage points; it is different from a percentage fall in dollars spent. The comparison does not reveal where shoppers went or why their habits changed.

**Risk models can help order a contact list.** The evaluation asks whether households make no purchases in the following two 28-day periods. Models are compared on later periods, with household-clustered confidence intervals because the same household can appear at several cutoffs. A small difference between model scores is not enough to claim statistically established superiority.

**Predicted risk and responsiveness are different.** A household likely to stop visiting may still be unaffected by an offer. Historical campaign models explore response differences, but campaign selection and unmeasured factors can bias those estimates.

See the [visual case study](https://sanjaybahadur.github.io/retention-radar/) and [generated results](../data/outputs/results.json) for current values. Forecasts and financial scenarios change when models or assumptions change; they are not realized returns.

## Recommended decision

Use risk rankings to propose a small, budget-limited experiment. Keep campaign-effect estimates exploratory. Do not automatically suppress households based on a negative estimated treatment effect.

## Experiment brief

- **Eligibility:** Unique households in the highest 30% of a frozen exploratory response ranking, subject to contact permissions. Freeze eligibility before assignment.
- **Assignment:** Randomize eligible households 1:1 to a defined offer or no additional offer. Stratify by prior spend and score band; keep assignment fixed throughout the experiment.
- **Primary outcome:** Average spend per assigned household per day over eight weeks, matching the included power calculation. Analyze everyone as assigned, including nonresponders.
- **Secondary outcomes:** Net contribution after measured offer and delivery costs, visits, and continued activity. Track opt-outs, complaints, and contact frequency as guardrails.
- **Sample size:** Set a minimum worthwhile effect and estimate outcome variability before launch. The included power calculation is a planning aid, not evidence that its assumed effect will occur.
- **Decision rule:** Agree on a meaningful effect threshold, uncertainty interval, budget limit, and guardrail criteria before seeing results. Report the primary outcome first; label subgroup analyses exploratory.

A measured trial would estimate whether an offer changes behavior in the selected pool and whether any improvement covers its cost. It would not by itself establish that this ranking beats random targeting. The historical analysis alone cannot settle that decision.
