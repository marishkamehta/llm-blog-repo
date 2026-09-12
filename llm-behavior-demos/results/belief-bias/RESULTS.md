# NeuBAROCO two-premise temperature-series results

## Analysis set

This analysis excludes source rows 234–239 and 250–252. These nine records
contain a third premise that is present in the source data but omitted by the
accompanying published prompt formatter. The exclusion is based on prompt
completeness, not model performance. The collected responses remain in the raw
record.

The final analysis contains 366 items, each repeated three times at eleven
temperatures from 0.0 to 1.0: 12,078 responses.

## Results

- Overall accuracy ranged from 65.0% to 66.7%.
- Belief-consistent accuracy ranged from 67.1% to 70.3%.
- Belief-inconsistent accuracy ranged from 52.9% to 57.2%.
- Symbolic-problem accuracy ranged from 70.5% to 74.4%.
- Belief-inconsistent accuracy was lower than belief-consistent accuracy at
  every temperature. The difference ranged from 10.5 to 16.7 percentage points.
- Mean repetition stability declined from 99.7% at temperature 0.0 to 94.2% at
  temperature 1.0.

One response at temperature 0.9 was invalid. The published and strict parsers
produced identical scores at every temperature.

These are descriptive results for a fixed model, set of items, and generation
environment. The lower accuracy for belief-inconsistent content is consistent
with the qualitative pattern examined in NeuBAROCO; it does not establish that
the model and humans use the same cognitive process.
