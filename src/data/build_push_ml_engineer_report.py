import csv
from pathlib import Path


DEFAULT_INPUT = Path("data/processed/ml_predictions/baseline_v1/baseline_v1_push_ml_predictions.csv")
DEFAULT_OUTPUT = Path("data/processed/ml_predictions/baseline_v1/baseline_v1_push_ml_engineer_report.md")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_amount(value: str) -> str:
    return value if value else "-"


def build_report(rows: list[dict[str, str]]) -> str:
    review_rows = [row for row in rows if row["auto_apply_status"] == "review_needed"]
    safe_rows = [row for row in rows if row["auto_apply_status"] == "safe_to_auto_apply"]

    lines: list[str] = [
        "# Push ML Engineer Report",
        "",
        f"- Total scored rows: `{len(rows)}`",
        f"- Review needed: `{len(review_rows)}`",
        f"- Safe to auto apply: `{len(safe_rows)}`",
        "",
        "## Review Needed First",
        "",
    ]

    if review_rows:
        lines.extend(
            [
                "| Run | Ch | Band | Tree | ML | ML Final | Direct Delta | Abs Diff | Reason |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in review_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["run"],
                        row["channel_set"],
                        row["pit_band"],
                        row["tree_push_action"],
                        row["ml_recommendation"],
                        fmt_amount(row["predicted_real_push_offset_final"]),
                        fmt_amount(row["direct_real_push_offset_final"]),
                        fmt_amount(row["amount_model_vs_direct_delta_abs_diff"]),
                        row["auto_apply_reason"],
                    ]
                )
                + " |"
            )
    else:
        lines.append("No review-needed cases.")

    lines.extend(["", "## Full Results By Run", ""])

    runs = sorted({int(row["run"]) for row in rows})
    for run in runs:
        run_rows = [row for row in rows if int(row["run"]) == run]
        lines.extend(
            [
                f"### Run {run}",
                "",
                "| Ch | Band | Tree | ML | Agreement | ML Final | Direct Delta | Abs Diff | Status |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in run_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["channel_set"],
                        row["pit_band"],
                        row["tree_push_action"],
                        row["ml_recommendation"],
                        row["tree_ml_agreement"],
                        fmt_amount(row["predicted_real_push_offset_final"]),
                        fmt_amount(row["direct_real_push_offset_final"]),
                        fmt_amount(row["amount_model_vs_direct_delta_abs_diff"]),
                        row["auto_apply_status"],
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows(DEFAULT_INPUT)
    report = build_report(rows)
    DEFAULT_OUTPUT.write_text(report, encoding="utf-8")
    print(f"Saved engineer report: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
