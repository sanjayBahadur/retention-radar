# Where the churn ranking helps—and where it misses

This review ranks households separately at each test cutoff and selects the highest-risk 20%. Selection uses the raw model ranking; calibrated probabilities are used for financial scenarios. A departure means no purchases in the next two 28-day periods.

A false alarm is a contacted household that continued shopping. A missed departure is a household that stopped visiting but fell outside the contact budget. Neither proves that an offer would help.

| Group | Observations | Departures | Contacted | Captured | Missed | False alarms | Capture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall: All observations | 9,397 | 712 | 1,878 | 534 | 178 | 1,344 | 75.0% |
| cutoff_period: 20 | 2,348 | 171 | 469 | 127 | 44 | 342 | 74.3% |
| cutoff_period: 21 | 2,345 | 165 | 469 | 126 | 39 | 343 | 76.4% |
| cutoff_period: 22 | 2,353 | 185 | 470 | 139 | 46 | 331 | 75.1% |
| cutoff_period: 23 | 2,351 | 191 | 470 | 142 | 49 | 328 | 74.3% |
| current_state: At-Risk | 785 | 182 | 609 | 158 | 24 | 451 | 86.8% |
| current_state: Champion | 2,027 | 6 | 5 | 1 | 5 | 4 | 16.7% |
| current_state: Dormant | 512 | 210 | 510 | 209 | 1 | 301 | 99.5% |
| current_state: New | 1 | 0 | 1 | 0 | 0 | 1 | — |
| current_state: Regular | 6,072 | 314 | 753 | 166 | 148 | 587 | 52.9% |
| tenure_group: More than 6 periods | 9,393 | 712 | 1,875 | 534 | 178 | 1,341 | 75.0% |
| tenure_group: Up to 6 periods | 4 | 0 | 3 | 0 | 0 | 3 | — |

## How to interpret this

Households can appear at several cutoffs. These are descriptive counts, not independent samples or proof that a subgroup difference is meaningful. Small groups need particular care. The overall model comparison reports household-clustered bootstrap intervals; this table does not attach independent-row confidence intervals to repeated household observations.

Before deployment, investigate missed departures and false alarms with an analyst, review changes across time, and run a household-randomized offer experiment. These historical errors were used for review, not further model tuning.
