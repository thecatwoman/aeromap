import argparse
import csv
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.build_pitlane_ml_dataset import CHANNEL_PUSH_SIGNAL

DEFAULT_INPUT_DATASET = Path(
    "data/processed/ml_datasets/baseline_v1/band_level/baseline_v1_band_level_push_training_dataset.csv"
)
DEFAULT_DECISION_MODEL = Path(
    "data/processed/ml_models/baseline_v1/push_band_offset_required_model.joblib"
)
DEFAULT_AMOUNT_MODEL = Path(
    "data/processed/ml_models/baseline_v1/push_band_offset_amount_model.joblib"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/ml_predictions/baseline_v1")
DEFAULT_OFFSET_CLAMP_ABS = 150.0
DEFAULT_OFFSET_STEP = 5.0
DEFAULT_DISAGREEMENT_THRESHOLD = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the push ML inference chain: first predict whether push offset is required, "
            "then predict the signed real-data push offset only for positive cases."
        )
    )
    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=DEFAULT_INPUT_DATASET,
        help="Band-level feature dataset to score.",
    )
    parser.add_argument(
        "--decision-model",
        type=Path,
        default=DEFAULT_DECISION_MODEL,
        help="Trained band-level push offset required classifier.",
    )
    parser.add_argument(
        "--amount-model",
        type=Path,
        default=DEFAULT_AMOUNT_MODEL,
        help="Trained band-level push offset amount regressor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where scored predictions and summary report will be written.",
    )
    parser.add_argument(
        "--offset-clamp-abs",
        type=float,
        default=DEFAULT_OFFSET_CLAMP_ABS,
        help="Maximum absolute real-data push offset allowed in the final recommendation.",
    )
    parser.add_argument(
        "--offset-step",
        type=float,
        default=DEFAULT_OFFSET_STEP,
        help="Rounding step for the final applied real-data push offset.",
    )
    parser.add_argument(
        "--disagreement-threshold",
        type=float,
        default=DEFAULT_DISAGREEMENT_THRESHOLD,
        help="Absolute difference threshold between direct-delta amount and ML amount above which review is required.",
    )
    return parser.parse_args()


def load_required_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def infer_feature_columns(model, frame: pd.DataFrame, excluded: set[str]) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [name for name in model.feature_names_in_ if name in frame.columns]
    return [name for name in frame.columns if name not in excluded]


def clamp_offset(value: float, clamp_abs: float) -> float:
    return max(-clamp_abs, min(clamp_abs, value))


def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def parse_float_cell(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def derive_direct_delta_amount(
    row: pd.Series,
    offset_clamp_abs: float,
    offset_step: float,
) -> dict[str, object]:
    signal = CHANNEL_PUSH_SIGNAL[str(row["channel_set"])]
    basis = str(row.get("push_band_basis", "unknown"))
    plateau_state = str(row.get(f"band_plateau_push_load_{signal}_state", "unknown"))
    plateau_delta = parse_float_cell(row.get(f"band_plateau_push_load_{signal}_rolling_median_delta"))
    median_state = str(row.get(f"band_median_{signal}_state", "unknown"))
    median_delta = parse_float_cell(row.get(f"band_median_{signal}_rolling_median_delta"))
    raw_delta = parse_float_cell(row.get("band_raw_primary_push_load_median_delta"))

    selected_delta = None
    source = "unavailable"
    if plateau_delta is not None and plateau_state == basis:
        selected_delta = plateau_delta
        source = "band_plateau_push_signal_rolling_median_delta"
    elif median_delta is not None and median_state == basis:
        selected_delta = median_delta
        source = "band_median_push_signal_rolling_median_delta"
    elif raw_delta is not None:
        selected_delta = raw_delta
        source = "band_raw_primary_push_load_median_delta"

    if selected_delta is None:
        return {
            "direct_real_push_offset_raw": "",
            "direct_real_push_offset_clamped": "",
            "direct_real_push_offset_final": "",
            "direct_real_push_offset_abs": "",
            "direct_real_push_offset_source": source,
        }

    direct_real_offset = -float(selected_delta)
    clamped = round(clamp_offset(direct_real_offset, offset_clamp_abs), 4)
    final = round(round_to_step(clamped, offset_step), 4)
    return {
        "direct_real_push_offset_raw": round(direct_real_offset, 4),
        "direct_real_push_offset_clamped": clamped,
        "direct_real_push_offset_final": final,
        "direct_real_push_offset_abs": round(abs(final), 4),
        "direct_real_push_offset_source": source,
    }


def build_predictions(
    frame: pd.DataFrame,
    decision_model,
    amount_model,
    offset_clamp_abs: float,
    offset_step: float,
    disagreement_threshold: float,
) -> pd.DataFrame:
    excluded = {
        "run",
        "channel_set",
        "segment_label",
        "pit_band",
        "push_band_offset_required",
    }
    decision_features = infer_feature_columns(decision_model, frame, excluded)
    amount_features = infer_feature_columns(amount_model, frame, excluded)

    X_decision = frame[decision_features].copy()
    decision_pred = decision_model.predict(X_decision)

    decision_prob = None
    if hasattr(decision_model, "predict_proba"):
        proba = decision_model.predict_proba(X_decision)
        positive_index = list(decision_model.classes_).index(1)
        decision_prob = proba[:, positive_index]

    results = frame[["run", "channel_set", "segment_label", "pit_band"]].copy()
    if "push_band_offset_required" in frame.columns:
        results["tree_push_offset_required"] = frame["push_band_offset_required"].astype(int)
        results["tree_push_action"] = frame["push_band_offset_required"].astype(int).map(
            {
                1: "apply_offset",
                0: "no_offset",
            }
        )
    else:
        results["tree_push_offset_required"] = ""
        results["tree_push_action"] = ""
    results["predicted_push_offset_required"] = decision_pred.astype(int)
    if decision_prob is not None:
        results["predicted_push_offset_probability"] = decision_prob.round(4)
    else:
        results["predicted_push_offset_probability"] = ""

    X_amount = frame[amount_features].copy()
    amount_pred = amount_model.predict(X_amount)
    results["predicted_real_push_offset_raw"] = ""
    results["predicted_real_push_offset_clamped"] = ""
    results["predicted_real_push_offset_final"] = ""
    results["predicted_real_push_offset_abs"] = ""
    results["direct_real_push_offset_raw"] = ""
    results["direct_real_push_offset_clamped"] = ""
    results["direct_real_push_offset_final"] = ""
    results["direct_real_push_offset_abs"] = ""
    results["direct_real_push_offset_source"] = ""
    positive_mask = results["predicted_push_offset_required"] == 1
    raw_values = pd.Series(amount_pred[positive_mask], index=results.index[positive_mask]).round(4)
    clamped_values = raw_values.apply(lambda value: round(clamp_offset(float(value), offset_clamp_abs), 4))
    final_values = clamped_values.apply(lambda value: round(round_to_step(float(value), offset_step), 4))
    results.loc[positive_mask, "predicted_real_push_offset_raw"] = raw_values
    results.loc[positive_mask, "predicted_real_push_offset_clamped"] = clamped_values
    results.loc[positive_mask, "predicted_real_push_offset_final"] = final_values
    results.loc[positive_mask, "predicted_real_push_offset_abs"] = (
        results.loc[positive_mask, "predicted_real_push_offset_final"].astype(float).abs().round(4)
    )

    results["ml_recommendation"] = results["predicted_push_offset_required"].map(
        {
            1: "Apply push offset",
            0: "No push offset",
        }
    )
    if "push_band_offset_required" in frame.columns:
        results["tree_ml_agreement"] = (
            results["tree_push_offset_required"].astype(int) == results["predicted_push_offset_required"].astype(int)
        ).map(
            {
                True: "agree",
                False: "disagree",
            }
        )
    else:
        results["tree_ml_agreement"] = "unavailable"
    results["final_push_action"] = results["predicted_push_offset_required"].map(
        {
            1: "apply_offset",
            0: "no_offset",
        }
    )

    direct_amount_rows = frame.apply(
        lambda row: derive_direct_delta_amount(row, offset_clamp_abs=offset_clamp_abs, offset_step=offset_step),
        axis=1,
    )
    direct_amount_df = pd.DataFrame(list(direct_amount_rows), index=frame.index)
    for column in direct_amount_df.columns:
        results[column] = direct_amount_df[column]

    results["amount_model_vs_direct_delta_abs_diff"] = ""
    results["amount_model_vs_direct_delta_pct_diff"] = ""
    comparable_mask = (
        positive_mask
        & (results["direct_real_push_offset_final"] != "")
        & (results["predicted_real_push_offset_final"] != "")
    )
    if comparable_mask.any():
        ml_final = results.loc[comparable_mask, "predicted_real_push_offset_final"].astype(float)
        direct_final = results.loc[comparable_mask, "direct_real_push_offset_final"].astype(float)
        abs_diff = (ml_final - direct_final).abs().round(4)
        results.loc[comparable_mask, "amount_model_vs_direct_delta_abs_diff"] = abs_diff
        pct_diff = pd.Series("", index=abs_diff.index, dtype=object)
        nonzero_direct = direct_final.abs() > 1e-9
        pct_diff.loc[nonzero_direct] = ((abs_diff.loc[nonzero_direct] / direct_final.loc[nonzero_direct].abs()) * 100.0).round(4)
        results.loc[comparable_mask, "amount_model_vs_direct_delta_pct_diff"] = pct_diff

    def build_review_status(row: pd.Series) -> tuple[str, str]:
        if row["final_push_action"] == "no_offset":
            return "safe_to_auto_apply", "ML recommends no push offset."
        if row["direct_real_push_offset_final"] == "":
            return "review_needed", "No direct delta amount available for comparison."
        abs_diff = parse_float_cell(row["amount_model_vs_direct_delta_abs_diff"])
        if abs_diff is None:
            return "review_needed", "Could not compute ML vs direct delta disagreement."
        if abs_diff <= disagreement_threshold:
            return "safe_to_auto_apply", "ML amount agrees with direct delta within threshold."
        return "review_needed", "ML amount and direct delta disagree beyond threshold."

    review_pairs = results.apply(build_review_status, axis=1)
    results["auto_apply_status"] = [item[0] for item in review_pairs]
    results["auto_apply_reason"] = [item[1] for item in review_pairs]
    results["final_recommendation"] = results.apply(
        lambda row: (
            f"Apply real push offset {row['predicted_real_push_offset_final']}"
            if row["final_push_action"] == "apply_offset"
            else "No push offset"
        ),
        axis=1,
    )
    return results


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows available to export.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    output_path: Path,
    input_dataset: Path,
    decision_model: Path,
    amount_model: Path,
    scored_df: pd.DataFrame,
    offset_clamp_abs: float,
    offset_step: float,
    disagreement_threshold: float,
) -> None:
    total_rows = len(scored_df)
    positive_rows = int((scored_df["predicted_push_offset_required"] == 1).sum())
    negative_rows = total_rows - positive_rows
    agreement_available = "tree_ml_agreement" in scored_df.columns
    agreement_counts = (
        scored_df["tree_ml_agreement"].value_counts(dropna=False).to_dict()
        if agreement_available
        else {}
    )
    auto_apply_counts = scored_df["auto_apply_status"].value_counts(dropna=False).to_dict()
    by_band = (
        scored_df.groupby(["pit_band", "predicted_push_offset_required"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    lines = [
        "# Push ML Inference Report: baseline_v1",
        "",
        f"- Input dataset: `{input_dataset}`",
        f"- Decision model: `{decision_model}`",
        f"- Amount model: `{amount_model}`",
        f"- Rows scored: `{total_rows}`",
        "",
        "## Prediction Summary",
        "",
        f"- Predicted `apply push offset`: `{positive_rows}`",
        f"- Predicted `no push offset`: `{negative_rows}`",
        f"- Final offset clamp: `+/-{offset_clamp_abs}`",
        f"- Final offset step: `{offset_step}`",
        f"- ML vs direct delta disagreement threshold: `{disagreement_threshold}`",
        "",
        "## By Band",
        "",
    ]

    for band in sorted(by_band.keys()):
        counts = by_band[band]
        zero_count = counts.get(0, 0)
        one_count = counts.get(1, 0)
        lines.append(f"- `{band}`: no_offset=`{zero_count}` apply_offset=`{one_count}`")

    if agreement_counts:
        lines.extend(
            [
                "",
                "## Tree vs ML",
                "",
                f"- agreement counts: `{agreement_counts}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Auto-Apply Status",
            "",
            f"- status counts: `{auto_apply_counts}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Inference Policy",
            "",
            "- First run the band-level push classifier.",
            "- Only if `predicted_push_offset_required = 1`, run the amount regressor.",
            "- The raw predicted amount is a signed real-data offset: positive means real should increase, negative means real should decrease.",
            "- A direct-delta amount is also computed from the band evidence using the same priority order as the training label builder.",
            "- ML and direct-delta amounts are compared before final recommendation is marked safe for auto-application.",
            "- Before final recommendation, clamp the raw amount to the configured absolute limit.",
            "- After clamping, round to the configured engineering step size.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_required_file(args.input_dataset, "Input dataset")
    load_required_file(args.decision_model, "Decision model")
    load_required_file(args.amount_model, "Amount model")

    frame = load_dataframe(args.input_dataset)
    decision_model = joblib.load(args.decision_model)
    amount_model = joblib.load(args.amount_model)

    scored_df = build_predictions(
        frame,
        decision_model,
        amount_model,
        offset_clamp_abs=args.offset_clamp_abs,
        offset_step=args.offset_step,
        disagreement_threshold=args.disagreement_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "baseline_v1_push_ml_predictions.csv"
    json_path = args.output_dir / "baseline_v1_push_ml_predictions_metadata.json"
    md_path = args.output_dir / "baseline_v1_push_ml_predictions_report.md"

    write_csv(scored_df.to_dict(orient="records"), csv_path)
    write_summary_markdown(
        md_path,
        args.input_dataset,
        args.decision_model,
        args.amount_model,
        scored_df,
        offset_clamp_abs=args.offset_clamp_abs,
        offset_step=args.offset_step,
        disagreement_threshold=args.disagreement_threshold,
    )

    metadata = {
        "input_dataset": str(args.input_dataset),
        "decision_model": str(args.decision_model),
        "amount_model": str(args.amount_model),
        "offset_clamp_abs": args.offset_clamp_abs,
        "offset_step": args.offset_step,
        "disagreement_threshold": args.disagreement_threshold,
        "output_csv": str(csv_path),
        "row_count": int(len(scored_df)),
        "predicted_apply_offset_count": int((scored_df["predicted_push_offset_required"] == 1).sum()),
        "predicted_no_offset_count": int((scored_df["predicted_push_offset_required"] == 0).sum()),
        "auto_apply_counts": scored_df["auto_apply_status"].value_counts(dropna=False).to_dict(),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved push ML predictions: {csv_path}")
    print(f"Saved push ML report: {md_path}")
    print(f"Saved push ML metadata: {json_path}")


if __name__ == "__main__":
    main()
