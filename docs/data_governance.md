# Data use and provenance

## Sources

Retention Radar uses Dunnhumby's *The Complete Journey* for retail purchase and campaign analysis, Kevin Hillstrom's email-marketing data for a randomized benchmark, and X5 RetailHero data for uplift evaluation and confounding experiments.

Code licensing and dataset permissions are separate. The repository's MIT license covers project code; it does not grant rights to the source datasets. Obtain data from its original provider or an authorized distribution and review that provider's current terms before use or redistribution. Do not infer permission from a mirror or from this repository.

## What outputs represent

Household identifiers are pseudonymous source keys. They are not proof that the records are anonymous or suitable for unrestricted sharing. The project does not need customer names, contact information, or payment details.

The public case study emphasizes aggregate analysis. Row-level exports exist to demonstrate local analytical workflows; review the source terms and intended audience before sharing those files. Synthetic test data should be clearly labeled and must not be described as real customer evidence.

Raw datasets and the local warehouse belong outside version control. Do not commit credentials, downloaded archives, or personal data. Review generated outputs before publication as well as source files.

## Boundaries for any future use

This repository does not implement a retailer's consent system, identity access controls, deletion process, or marketing operations. A real deployment would need those controls, documented data retention, restricted row-level access, and measured contact outcomes.

Demographic information is incomplete. Audit differences with sample sizes and uncertainty, and avoid inferring fairness from the removal of selected features. Historical response estimates should not trigger automatic punitive actions or permanent contact suppression.
