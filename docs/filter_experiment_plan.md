# Filter Experiment Plan

This experiment is intentionally separate from the official raw baseline.

Its purpose is to answer:

- Does filtering improve segmentation stability?
- Does filtering reduce corner-to-corner scatter?
- Does filtering improve reference-map comparison readiness?

It is **not** meant to silently replace the official baseline.

## Principle

We keep the same:

- input run
- cleaned merged dataset
- merged Scz channels
- recalculated `scz_push_f_pitot`
- recalculated `scz_push_r_pitot`
- segmentation logic
- corner extraction logic

and vary only the filtering branch.

## Branches

The experiment runner creates 3 branches:

1. `raw`
   - no filtering

2. `despike`
   - despike only

3. `despike_butterworth`
   - despike first
   - Butterworth second

## Important Scope Decision

Filtering is applied **after** dataset build and **after** Scz recalculation.

That means this experiment currently tests:

- segmentation robustness
- corner extraction robustness
- comparison stability

and does **not** yet test:

- whether filtering should be part of the Scz formulas themselves

This is intentional because it keeps the first experiment lower-risk and easier to interpret.

## First Channels Tested

### Despike

- `rh_f`
- `rh_r`
- `pushavg_c`
- `pushavd_c`
- `pusharg_c`
- `pushard_c`

### Butterworth

- ride height:
  - `rh_f`
  - `rh_r`
- push load:
  - `pushavg_c`
  - `pushavd_c`
  - `pusharg_c`
  - `pushard_c`
- pitot/pAir:
  - `pitot_c`
  - `pair`
- damper:
  - `damper_fl_art`
  - `damper_fr_art`
  - `damper_rl_art`
  - `damper_rr_art`

## Segmentation Used

All branches use the same official baseline segmentation:

- `macroway`

with the same parameters already used in the project.

## What Gets Saved

For each branch:

- filtered dataset
- segmented dataset
- front push corner summary
- rear push corner summary
- front exact sparse table
- rear exact sparse table
- front binned table
- rear binned table

At experiment level:

- `filter_experiment_comparison.csv`
- `filter_experiment_report.json`

## Main Metrics To Compare

The comparison CSV reports:

- pit sample count
- straight sample count
- corner sample count
- transition sample count
- front corner segment count
- rear corner segment count
- corner RH spread
- corner `scz_push_f_pitot` spread
- corner `scz_push_r_pitot` spread
- deltas versus raw branch

## Run Command

```bash
.venv/bin/python -m src.data.filter_experiment_real
```

## Output Directory

```text
data/processed/Run_<run_number>/Filtering/
```

## How To Decide If Filtering Helps

Filtering is worth promoting only if it clearly does at least one of these:

- removes obvious spikes
- improves segmentation consistency
- reduces corner scatter
- makes repeated corners more stable
- improves reference-map comparison readiness

and does **not** flatten physically meaningful behavior.

If filtering only makes curves look smoother, that is not enough.
