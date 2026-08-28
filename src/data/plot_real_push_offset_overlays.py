import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.pitlane_decision_tree import select_matched_pit_segments_for_analysis
from src.run_paths import processed_simudata_dir


DEFAULT_SUMMARY = Path("data/processed/real_push_offset_application_summary.json")
DEFAULT_ML_SUMMARY = Path("data/processed/real_push_offset_application_summary_ml.json")

CHANNEL_TO_PUSH_COLUMN = {
    "fl": "pushavg_c",
    "fr": "pushavd_c",
    "rl": "pusharg_c",
    "rr": "pushard_c",
}

CHANNEL_TO_SCZ_COLUMN = {
    "fl": "scz_push_f_pitot",
    "fr": "scz_push_f_pitot",
    "rl": "scz_push_r_pitot",
    "rr": "scz_push_r_pitot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pit-band push-channel overlays for original real, offset-applied real, and sim data."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Primary offset application summary JSON produced by apply_real_push_offsets.",
    )
    parser.add_argument(
        "--compare-summary-json",
        type=Path,
        default=None,
        help="Optional second offset application summary JSON to compare in the same plot.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        help="Optional subset of runs to plot.",
    )
    parser.add_argument(
        "--channel-set",
        choices=["fl", "fr", "rl", "rr"],
        default=None,
        help="Optional single corner/channel-set to plot instead of all four.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNG files instead of opening interactive plot windows.",
    )
    parser.add_argument(
        "--pit-speed-band",
        choices=["slow", "fast"],
        default="slow",
        help="Pit speed band to inspect. This uses the same band definitions as the decision tree.",
    )
    parser.add_argument(
        "--min-pit-band-samples",
        type=int,
        default=50,
        help="Minimum samples required for the selected pit band.",
    )
    parser.add_argument(
        "--show-delta",
        action="store_true",
        help="Also show a lower panel with real-sim deltas.",
    )
    parser.add_argument(
        "--signal-kind",
        choices=["push", "scz"],
        default="push",
        help="Which signal family to compare.",
    )
    parser.add_argument(
        "--scope",
        choices=["pit_band", "full_run"],
        default="pit_band",
        help="Plot either the selected pit band or the entire run.",
    )
    parser.add_argument(
        "--full-run-axis",
        choices=["global_samples", "distancelap"],
        default="distancelap",
        help="When --scope full_run is used, choose the shared x-axis style.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_summary_by_run(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(row["run"]): row for row in rows}


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalized_axis(length: int) -> np.ndarray:
    if length <= 1:
        return np.asarray([0.0], dtype=float)
    return np.linspace(0.0, 1.0, length)


def build_normalized_axis_from_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Missing shared axis column {column!r}")
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()
    if len(valid) < 2:
        raise ValueError(f"Not enough valid values in shared axis column {column!r}")
    value_min = float(valid.min())
    value_max = float(valid.max())
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max <= value_min:
        raise ValueError(f"Shared axis column {column!r} is not usable for normalization")
    normalized = ((values - value_min) / (value_max - value_min)).clip(0.0, 1.0)
    return normalized


def interpolate_on_common_axis(
    df: pd.DataFrame,
    axis_column: str,
    value_column: str,
    common_axis: np.ndarray,
) -> np.ndarray:
    axis = build_normalized_axis_from_column(df, axis_column)
    values = pd.to_numeric(df[value_column], errors="coerce")
    clean = pd.DataFrame({"axis": axis, "value": values}).dropna()
    clean = clean[np.isfinite(clean["axis"]) & np.isfinite(clean["value"])]
    clean = clean.sort_values("axis")
    clean = clean.drop_duplicates(subset="axis", keep="first")
    if len(clean) < 2:
        raise ValueError(f"Not enough valid {value_column!r} values for interpolation")
    return np.interp(
        common_axis,
        clean["axis"].to_numpy(dtype=float),
        clean["value"].to_numpy(dtype=float),
    )


def signal_column_map(signal_kind: str) -> dict[str, str]:
    if signal_kind == "scz":
        return CHANNEL_TO_SCZ_COLUMN
    return CHANNEL_TO_PUSH_COLUMN


def resolve_compare_summary_path(primary_summary: Path, requested_compare: Path | None) -> Path | None:
    if requested_compare is not None:
        return requested_compare
    primary_resolved = primary_summary.resolve()
    default_direct = DEFAULT_SUMMARY.resolve()
    default_ml = DEFAULT_ML_SUMMARY.resolve()
    if primary_resolved == default_direct and DEFAULT_ML_SUMMARY.exists():
        return DEFAULT_ML_SUMMARY
    if primary_resolved == default_ml and DEFAULT_SUMMARY.exists():
        return DEFAULT_SUMMARY
    return None


def plot_channel_overlay(
    run: int,
    channel_set: str,
    value_column: str,
    source_real_df: pd.DataFrame,
    offset_real_df: pd.DataFrame,
    offset_label: str,
    compare_offset_real_df: pd.DataFrame | None,
    sim_df: pd.DataFrame,
    output_path: Path | None,
    save: bool,
    pit_speed_band: str,
    min_pit_band_samples: int,
    show_delta: bool,
    compare_label: str | None,
    scope: str,
    full_run_axis: str,
    signal_kind: str,
) -> None:
    if scope == "pit_band":
        source_real_view, sim_view = select_matched_pit_segments_for_analysis(
            source_real_df,
            sim_df,
            pit_speed_band=pit_speed_band,
            min_pit_band_samples=min_pit_band_samples,
            quality_column=value_column,
        )
        offset_real_view = offset_real_df.loc[source_real_view.index].copy()
        compare_offset_real_view = (
            compare_offset_real_df.loc[source_real_view.index].copy()
            if compare_offset_real_df is not None
            else None
        )
        title_scope = f"{pit_speed_band} pit-band"
        x_label = "Normalized pit-band progress"
    else:
        source_real_view = source_real_df.copy()
        sim_view = sim_df.copy()
        offset_real_view = offset_real_df.copy()
        compare_offset_real_view = compare_offset_real_df.copy() if compare_offset_real_df is not None else None
        title_scope = "full-run"
        x_label = (
            "Normalized distancelap progress"
            if full_run_axis == "distancelap"
            else "Normalized global sample progress"
        )

    if scope == "full_run":
        if full_run_axis == "distancelap":
            common_axis = np.linspace(0.0, 1.0, 1500)
            source_real = interpolate_on_common_axis(source_real_view, "distancelap", value_column, common_axis)
            offset_real = interpolate_on_common_axis(offset_real_view, "distancelap", value_column, common_axis)
            compare_offset_real = (
                interpolate_on_common_axis(compare_offset_real_view, "distancelap", value_column, common_axis)
                if compare_offset_real_view is not None
                else None
            )
            sim = interpolate_on_common_axis(sim_view, "distancelap", value_column, common_axis)
            source_x = common_axis
            offset_x = common_axis
            compare_offset_x = common_axis if compare_offset_real is not None else None
            sim_x = common_axis
        else:
            source_real = pd.to_numeric(source_real_view[value_column], errors="coerce").to_numpy(dtype=float)
            offset_real = pd.to_numeric(offset_real_view[value_column], errors="coerce").to_numpy(dtype=float)
            compare_offset_real = (
                pd.to_numeric(compare_offset_real_view[value_column], errors="coerce").to_numpy(dtype=float)
                if compare_offset_real_view is not None
                else None
            )
            sim = pd.to_numeric(sim_view[value_column], errors="coerce").to_numpy(dtype=float)
            source_x = normalized_axis(len(source_real))
            offset_x = normalized_axis(len(offset_real))
            compare_offset_x = normalized_axis(len(compare_offset_real)) if compare_offset_real is not None else None
            sim_x = normalized_axis(len(sim))
    else:
        source_real = pd.to_numeric(source_real_view[value_column], errors="coerce").to_numpy(dtype=float)
        offset_real = pd.to_numeric(offset_real_view[value_column], errors="coerce").to_numpy(dtype=float)
        compare_offset_real = (
            pd.to_numeric(compare_offset_real_view[value_column], errors="coerce").to_numpy(dtype=float)
            if compare_offset_real_view is not None
            else None
        )
        sim = pd.to_numeric(sim_view[value_column], errors="coerce").to_numpy(dtype=float)
        source_x = normalized_axis(len(source_real))
        offset_x = normalized_axis(len(offset_real))
        compare_offset_x = normalized_axis(len(compare_offset_real)) if compare_offset_real is not None else None
        sim_x = normalized_axis(len(sim))

    if show_delta:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(sim_x, sim, label="sim", color="#ff7f0e", linewidth=1.8, alpha=0.9, zorder=1)
    ax.plot(
        offset_x,
        offset_real,
        label=f"real {offset_label}",
        color="#2ca02c",
        linewidth=1.8,
        alpha=0.95,
        zorder=2,
    )
    if compare_offset_real is not None and compare_offset_x is not None:
        ax.plot(
            compare_offset_x,
            compare_offset_real,
            label=f"real {compare_label}",
            color="#d62728",
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )
    ax.plot(
        source_x,
        source_real,
        label="real original",
        color="#1f77b4",
        linewidth=2.2,
        linestyle="-",
        alpha=1.0,
        zorder=5,
    )
    ax.set_ylabel(value_column)
    ax.set_title(f"Run {run} {channel_set}: {title_scope} {signal_kind} overlay")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if show_delta:
        diff_original = np.interp(sim_x, source_x, source_real) - sim
        diff_offset = np.interp(sim_x, offset_x, offset_real) - sim
        diff_compare = (
            np.interp(sim_x, compare_offset_x, compare_offset_real) - sim
            if compare_offset_real is not None and compare_offset_x is not None
            else None
        )
        axes[1].plot(
            sim_x,
            diff_offset,
            label=f"{offset_label} - sim",
            color="#2ca02c",
            linewidth=1.4,
            zorder=2,
        )
        if diff_compare is not None:
            axes[1].plot(
                sim_x,
                diff_compare,
                label=f"{compare_label} - sim",
                color="#d62728",
                linewidth=1.4,
                zorder=3,
            )
        axes[1].plot(
            sim_x,
            diff_original,
            label="original real - sim",
            color="#1f77b4",
            linewidth=1.8,
            linestyle="-",
            alpha=1.0,
            zorder=5,
        )
        axes[1].axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("Delta")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")
    else:
        ax.set_xlabel(x_label)

    fig.tight_layout()
    if save:
        if output_path is None:
            raise ValueError("output_path is required when save=True")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    summary_rows = load_summary(args.summary_json)
    compare_summary_path = resolve_compare_summary_path(args.summary_json, args.compare_summary_json)
    compare_summary_rows = load_summary(compare_summary_path) if compare_summary_path else None
    compare_by_run = index_summary_by_run(compare_summary_rows) if compare_summary_rows is not None else {}
    if args.runs:
        wanted = set(args.runs)
        summary_rows = [row for row in summary_rows if int(row["run"]) in wanted]

    for row in summary_rows:
        run = int(row["run"])
        if args.scope == "pit_band":
            source_real_path = Path(str(row["source_segmented_dataset"]))
            offset_real_path = Path(str(row["output_segmented_dataset"]))
            sim_path = processed_simudata_dir(run) / f"barcelona_2026_simudata_segmented_run_{run}.csv"
        else:
            source_real_path = Path(str(row["source_full_run_dataset"]))
            offset_real_path = Path(str(row["output_full_run_dataset"]))
            sim_path = processed_simudata_dir(run) / f"barcelona_2026_simudata_cleaned_full_run_{run}.csv"
        plot_dir = offset_real_path.parent.parent / "plots"
        source_real_df = load_dataframe(source_real_path)
        offset_real_df = load_dataframe(offset_real_path)
        primary_amount_source = str(row.get("amount_source", "direct"))
        offset_label = "ml-offset" if primary_amount_source == "ml" else "direct-offset"
        compare_row = compare_by_run.get(run)
        compare_offset_real_df = None
        compare_label = None
        if compare_row is not None:
            compare_output_path = (
                Path(str(compare_row["output_segmented_dataset"]))
                if args.scope == "pit_band"
                else Path(str(compare_row["output_full_run_dataset"]))
            )
            compare_offset_real_df = load_dataframe(compare_output_path)
            compare_label = "ml-offset" if str(compare_row.get("amount_source", "")) == "ml" else "direct-offset"
        sim_df = load_dataframe(sim_path)
        column_map = signal_column_map(args.signal_kind)
        if args.channel_set is not None:
            column_map = {args.channel_set: column_map[args.channel_set]}
        for channel_set, value_column in column_map.items():
            output_path = (
                plot_dir
                / f"run_{run}_{channel_set}_{value_column}_{args.signal_kind}_{args.scope}_{args.pit_speed_band}_real_offset_overlay.png"
            )
            plot_channel_overlay(
                run=run,
                channel_set=channel_set,
                value_column=value_column,
                source_real_df=source_real_df,
                offset_real_df=offset_real_df,
                offset_label=offset_label,
                compare_offset_real_df=compare_offset_real_df,
                sim_df=sim_df,
                output_path=output_path if args.save else None,
                save=args.save,
                pit_speed_band=args.pit_speed_band,
                min_pit_band_samples=args.min_pit_band_samples,
                show_delta=args.show_delta,
                compare_label=compare_label,
                scope=args.scope,
                full_run_axis=args.full_run_axis,
                signal_kind=args.signal_kind,
            )
            if args.save:
                print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
