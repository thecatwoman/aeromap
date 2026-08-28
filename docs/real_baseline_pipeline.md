# Real Baseline Pipeline

This file freezes one official real-data path for the project so later ML work uses the same upstream logic every time.

## Official Baseline

- Source: real telemetry run in `data/raw/Barcelona_2026/Test/Run_<run_number>/`
- Scz source: `Scz_RUN<run_number>/`
- Run selector: `CURRENT_RUN` in [src/run_paths.py](/Users/nini/Projects/aeromap/src/run_paths.py:6)
- Official segmentation for the baseline: `macroway`
- Official segment label for ML preparation: `corner`

## Baseline Stages

1. Raw run input
2. Cleaned + merged dataset
3. Scz merge
4. Recalculated `scz_push_f_pitot`
5. Recalculated `scz_push_r_pitot`
6. Official segmentation
7. Corner extraction
8. Reference-map comparison

## One-Command Runner

```bash
.venv/bin/python -m src.data.baseline_real_pipeline
```

This runs all official baseline steps for `CURRENT_RUN` and writes a manifest to:

```text
data/processed/Run_<run_number>/baseline_real_pipeline_manifest.json
```

## Equivalent Manual Commands

```bash
.venv/bin/python -m src.data.build_dataset --source real
.venv/bin/python -m src.segment_with_macroway --source real
.venv/bin/python -m src.data.build_push_corner_tables --segmenter macroway
.venv/bin/python -m src.data.plot_push_corner_3d --side both --segmenter macroway
```

## Official Baseline Outputs

### Dataset

- `data/processed/Run_<run_number>/Cleaned_Merged/barcelona_2026_merged_cleaned_full_run_<run_number>.csv`

This is the baseline real dataset after:
- cleaning/merge
- Scz merge
- recalculated push Scz channels

### Segmentation

- `data/processed/Run_<run_number>/Segmented/barcelona_2026_merged_cleaned_RH_run_<run_number>_segmented.csv`

This is the official segmented real dataset for the baseline path.

### Corner Tables

- `data/processed/tables/corner_maps/run_<run_number>_corner_scz_push_f_pitot_macroway_two_way_table_exact.csv`
- `data/processed/tables/corner_maps/run_<run_number>_corner_scz_push_r_pitot_macroway_two_way_table_exact.csv`

These are the official sparse corner tables extracted from the segmented real dataset.

### Reference Comparison Plots

- `data/processed/plots/reference_table/run_<run_number>_scz_push_f_pitot_macroway_overlay_shifted_axes.png`
- `data/processed/plots/reference_table/run_<run_number>_scz_push_f_pitot_macroway_overlay_shifted_axes.html`
- `data/processed/plots/reference_table/run_<run_number>_scz_push_r_pitot_macroway_overlay_shifted_axes.png`
- `data/processed/plots/reference_table/run_<run_number>_scz_push_r_pitot_macroway_overlay_shifted_axes.html`

## Why This Is Frozen

ML for offset decision should not learn from mixed upstream logic.

This baseline removes ambiguity about:
- which run selector is active
- which segmentation is used
- which Scz channels are the official recalculated ones
- which corner tables become ML inputs
- which plots are the official visual comparison against the reference map
