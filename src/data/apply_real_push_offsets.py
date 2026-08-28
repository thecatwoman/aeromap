import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data.calculate_scz_push import calculate_scz_push_f, calculate_scz_push_r


DEFAULT_PREDICTIONS = Path(
    "data/processed/ml_predictions/baseline_v1/baseline_v1_push_ml_predictions.csv"
)
DEFAULT_OUTPUT_DIRNAME = "Real_Offset_Applied_Baseline_v1"

CHANNEL_TO_PUSH_COLUMN = {
    "fl": "pushavg_c",
    "fr": "pushavd_c",
    "rl": "pusharg_c",
    "rr": "pushard_c",
}


@dataclass
class OffsetSelection:
    run: int
    channel_set: str
    pit_band: str
    push_column: str
    tree_push_action: str
    predicted_push_offset_required: int
    direct_real_push_offset_final: float | None
    predicted_real_push_offset_final: float | None
    amount_model_vs_direct_delta_abs_diff: float | None
    auto_apply_status: str
    auto_apply_reason: str
    final_recommendation: str


@dataclass
class OutputManifest:
    run: int
    amount_source: str
    source_full_run_dataset: str
    source_segmented_dataset: str
    output_full_run_dataset: str
    output_segmented_dataset: str
    offsets_csv: str
    offsets_json: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply real-data push offsets to full-run frozen baseline datasets. "
            "Uses the slow-band direct-delta amount or ML-predicted amount as the applied engineering offset, "
            "while retaining the other amount and confidence metadata for audit."
        )
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Band-level push ML predictions CSV.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        help="Optional subset of runs to export. Defaults to all runs present in the slow-band predictions.",
    )
    parser.add_argument(
        "--output-dirname",
        default=None,
        help="Name of the per-run output directory created under data/processed/Run_<N>/.",
    )
    parser.add_argument(
        "--amount-source",
        choices=["direct", "ml"],
        default="direct",
        help="Which real-offset amount to apply: direct delta or ML-predicted amount.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_optional_float(value: str) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def select_slow_band_offsets(rows: list[dict[str, str]]) -> list[OffsetSelection]:
    selections: list[OffsetSelection] = []
    for row in rows:
        if row["pit_band"] != "slow":
            continue
        channel_set = row["channel_set"]
        if channel_set not in CHANNEL_TO_PUSH_COLUMN:
            continue
        selections.append(
            OffsetSelection(
                run=int(row["run"]),
                channel_set=channel_set,
                pit_band=row["pit_band"],
                push_column=CHANNEL_TO_PUSH_COLUMN[channel_set],
                tree_push_action=row["tree_push_action"],
                predicted_push_offset_required=int(row["predicted_push_offset_required"]),
                direct_real_push_offset_final=parse_optional_float(row["direct_real_push_offset_final"]),
                predicted_real_push_offset_final=parse_optional_float(row["predicted_real_push_offset_final"]),
                amount_model_vs_direct_delta_abs_diff=parse_optional_float(
                    row["amount_model_vs_direct_delta_abs_diff"]
                ),
                auto_apply_status=row["auto_apply_status"],
                auto_apply_reason=row["auto_apply_reason"],
                final_recommendation=row["final_recommendation"],
            )
        )
    return selections


def find_dataset_path(run_dir: Path, filename_pattern: str) -> Path:
    matches = sorted((run_dir / "Frozen_Baseline" / "datasets").glob(filename_pattern))
    if not matches:
        raise FileNotFoundError(f"No dataset matched {filename_pattern} under {run_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple datasets matched {filename_pattern} under {run_dir}: {matches}")
    return matches[0]


def apply_offsets_and_recalculate_scz(
    source_path: Path,
    target_path: Path,
    column_offsets: dict[str, float],
) -> int:
    df = pd.read_csv(source_path, low_memory=False)
    for column_name, offset_value in column_offsets.items():
        if column_name not in df.columns:
            raise KeyError(f"Column {column_name!r} not found in {source_path}")
        numeric = pd.to_numeric(df[column_name], errors="coerce")
        df[column_name] = numeric + offset_value

    # Reuse the same trusted SCz recalculation functions as the baseline dataset build.
    df = calculate_scz_push_f(df)
    df = calculate_scz_push_r(df)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return int(len(df))


def write_offsets_table(path: Path, selections: list[OffsetSelection]) -> None:
    rows = [asdict(item) for item in selections]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_run(
    run: int,
    run_selections: list[OffsetSelection],
    output_dirname: str,
    amount_source: str,
) -> OutputManifest:
    run_dir = Path("data/processed") / f"Run_{run}"
    source_full = find_dataset_path(run_dir, "*merged_cleaned_full_run_*.csv")
    source_segmented = find_dataset_path(run_dir, "*merged_cleaned_RH_run_*_segmented.csv")

    output_dir = run_dir / output_dirname
    datasets_dir = output_dir / "datasets"
    metadata_dir = output_dir / "metadata"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_real_push_offset_applied_{amount_source}.csv"
    output_full = datasets_dir / source_full.name.replace(".csv", suffix)
    output_segmented = datasets_dir / source_segmented.name.replace(
        ".csv", suffix
    )

    column_offsets: dict[str, float] = {}
    for selection in run_selections:
        if selection.tree_push_action != "apply_offset":
            continue
        if amount_source == "direct":
            selected_amount = selection.direct_real_push_offset_final
            selected_field = "direct_real_push_offset_final"
        else:
            selected_amount = selection.predicted_real_push_offset_final
            selected_field = "predicted_real_push_offset_final"
        if selected_amount is None:
            raise ValueError(
                f"Run {run} channel {selection.channel_set} requires offset but has no {selected_field}."
            )
        column_offsets[selection.push_column] = selected_amount

    if not column_offsets:
        raise ValueError(
            f"Run {run} produced no applied push offsets from the slow-band official decisions."
        )

    apply_offsets_and_recalculate_scz(source_full, output_full, column_offsets)
    apply_offsets_and_recalculate_scz(source_segmented, output_segmented, column_offsets)

    offsets_csv = metadata_dir / f"run_{run}_applied_real_push_offsets_{amount_source}.csv"
    write_offsets_table(offsets_csv, run_selections)

    offsets_json = metadata_dir / f"run_{run}_applied_real_push_offsets_{amount_source}.json"
    offsets_payload = {
        "run": run,
        "policy": {
            "decision_band": "slow",
            "applied_amount_source": (
                "direct_real_push_offset_final"
                if amount_source == "direct"
                else "predicted_real_push_offset_final"
            ),
            "ml_role": "confidence_only" if amount_source == "direct" else "applied_amount_source",
            "channel_scope": "push_only",
            "data_scope": "full_run_and_segmented_exports",
            "original_datasets_preserved": True,
        },
        "applied_column_offsets": column_offsets,
        "selections": [asdict(item) for item in run_selections],
    }
    offsets_json.write_text(json.dumps(offsets_payload, indent=2) + "\n", encoding="utf-8")

    manifest = OutputManifest(
        run=run,
        amount_source=amount_source,
        source_full_run_dataset=str(source_full),
        source_segmented_dataset=str(source_segmented),
        output_full_run_dataset=str(output_full),
        output_segmented_dataset=str(output_segmented),
        offsets_csv=str(offsets_csv),
        offsets_json=str(offsets_json),
        notes=[
            "Original frozen baseline datasets were not modified.",
            "Only real push channels were adjusted in this export.",
            (
                "Applied amounts come from slow-band direct_real_push_offset_final."
                if amount_source == "direct"
                else "Applied amounts come from slow-band predicted_real_push_offset_final."
            ),
            "SCz front and rear were recalculated after the real push offsets were applied.",
            (
                "ML-predicted amount is retained only as confidence/audit metadata."
                if amount_source == "direct"
                else "Direct-delta amount is retained as comparison/audit metadata."
            ),
        ],
    )
    manifest_path = output_dir / f"offset_application_manifest_{amount_source}.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    output_dirname = args.output_dirname
    if output_dirname is None:
        output_dirname = (
            DEFAULT_OUTPUT_DIRNAME
            if args.amount_source == "direct"
            else "Real_Offset_Applied_Baseline_v1_ML"
        )
    rows = load_rows(args.predictions_csv)
    selections = select_slow_band_offsets(rows)
    if args.runs:
        wanted = set(args.runs)
        selections = [item for item in selections if item.run in wanted]

    if not selections:
        raise ValueError("No slow-band offset selections available for the requested runs.")

    manifests: list[OutputManifest] = []
    grouped: dict[int, list[OffsetSelection]] = {}
    for selection in selections:
        grouped.setdefault(selection.run, []).append(selection)

    for run in sorted(grouped):
        manifests.append(process_run(run, grouped[run], output_dirname, args.amount_source))

    summary_name = (
        "real_push_offset_application_summary.json"
        if args.amount_source == "direct"
        else "real_push_offset_application_summary_ml.json"
    )
    summary_path = Path("data/processed") / summary_name
    summary_path.write_text(
        json.dumps([asdict(item) for item in manifests], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved offset application summary: {summary_path}")
    for manifest in manifests:
        print(f"Run {manifest.run}: wrote {manifest.output_full_run_dataset}")


if __name__ == "__main__":
    main()
