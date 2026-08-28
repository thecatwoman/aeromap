import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.apply_butterworth_filter import BUTTERWORTH_TARGET_COLUMNS
from src.run_paths import CURRENT_RUN, processed_run_dir


DEFAULT_CHANNELS = [
    "rh_f",
    "rh_r",
    "pushavg_c",
    "pushavd_c",
    "pusharg_c",
    "pushard_c",
    "pitot_c",
    "pair",
    "damper_fl_art",
    "damper_fr_art",
    "damper_rl_art",
    "damper_rr_art",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visually inspect raw vs despike vs despike+Butterworth outputs from the real filtering experiment.",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=CURRENT_RUN,
        help="Run number used for the filtering experiment outputs.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Optional explicit filtering experiment directory. Defaults to data/processed/Run_<run>/Filtering.",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=["dataset", "segmented"],
        default="dataset",
        help="Whether to inspect pre-segmentation branch datasets or segmented branch datasets.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Channels to inspect.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["sample", "time"],
        default="sample",
        help="X axis to use.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row for the plotted window when no segment selection is used.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1500,
        help="Number of rows to inspect when no segment selection is used.",
    )
    parser.add_argument(
        "--segment-id",
        type=int,
        default=None,
        help="Optional segment_id to inspect from the raw segmented branch. Overrides --start/--count.",
    )
    parser.add_argument(
        "--segment-label",
        choices=["pit", "straight", "corner", "transition"],
        default=None,
        help="Optional first segment label to inspect from the raw segmented branch. Overrides --start/--count unless --segment-id is also set.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Inspect the full run. Overrides --start/--count and segment selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plot images will be saved.",
    )
    return parser.parse_args()


def branch_base_dir(run_number: int) -> Path:
    return processed_run_dir(run_number) / "Filtering"


def branch_file(branch_dir: Path, branch_name: str, dataset_kind: str) -> Path:
    suffix = "dataset.csv" if dataset_kind == "dataset" else "segmented.csv"
    return branch_dir / branch_name / f"{branch_name}_{suffix}"


def load_branch_frames(base_dir: Path, dataset_kind: str) -> dict[str, pd.DataFrame]:
    return {
        "raw": pd.read_csv(branch_file(base_dir, "raw", dataset_kind), low_memory=False),
        "despike": pd.read_csv(branch_file(base_dir, "despike", dataset_kind), low_memory=False),
        "despike_butterworth": pd.read_csv(
            branch_file(base_dir, "despike_butterworth", dataset_kind),
            low_memory=False,
        ),
    }


def pick_window_from_segment(raw_segmented: pd.DataFrame, args: argparse.Namespace) -> tuple[int, int, str]:
    if args.full_run:
        return 0, len(raw_segmented), "full run"

    if args.segment_id is not None:
        seg = raw_segmented[raw_segmented["segment_id"] == args.segment_id]
        if seg.empty:
            raise ValueError(f"No rows found for segment_id={args.segment_id}.")
        start = int(seg.index.min())
        end = int(seg.index.max()) + 1
        label = str(seg["segment_final"].iloc[0]) if "segment_final" in seg.columns else "unknown"
        return start, end, f"segment_id={args.segment_id} ({label})"

    if args.segment_label is not None:
        seg = raw_segmented[raw_segmented["segment_final"].astype("string") == args.segment_label].copy()
        if seg.empty:
            raise ValueError(f"No rows found for segment_label={args.segment_label}.")
        seg["group"] = (seg.index.to_series().diff() != 1).cumsum()
        first_group = seg["group"].iloc[0]
        first = seg[seg["group"] == first_group]
        start = int(first.index.min())
        end = int(first.index.max()) + 1
        segment_id = int(first["segment_id"].iloc[0]) if "segment_id" in first.columns else -1
        return start, end, f"first {args.segment_label} segment_id={segment_id}"

    if args.start < 0:
        raise ValueError("--start must be zero or positive.")
    if args.count <= 0:
        raise ValueError("--count must be greater than zero.")
    return args.start, args.start + args.count, f"rows {args.start}:{args.start + args.count}"


def build_x_axis(df: pd.DataFrame, start: int, end: int, mode: str) -> pd.Series:
    window = df.iloc[start:end]
    if mode == "time" and "time" in window.columns:
        return pd.to_numeric(window["time"], errors="coerce")
    return pd.Series(window.index.to_numpy(), index=window.index, dtype="int64")


def summarize_delta(raw: pd.Series, other: pd.Series) -> str:
    aligned = pd.DataFrame({"raw": raw, "other": other}).dropna()
    if aligned.empty:
        return "no valid overlap"
    delta = aligned["raw"] - aligned["other"]
    changed = int((delta != 0).sum())
    return (
        f"changed={changed} | "
        f"mean={delta.mean():.4f} | "
        f"median={delta.median():.4f} | "
        f"max_abs={delta.abs().max():.4f}"
    )


def uses_butterworth(channel: str) -> bool:
    return channel in BUTTERWORTH_TARGET_COLUMNS


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir or branch_base_dir(args.run_number)
    frames = load_branch_frames(experiment_dir, args.dataset_kind)
    raw_segmented = load_branch_frames(experiment_dir, "segmented")["raw"]
    start, end, window_label = pick_window_from_segment(raw_segmented, args)

    output_dir = args.output_dir or (experiment_dir / "inspection_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting {window_label}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Output directory: {output_dir}")

    for channel in args.channels:
        missing = [name for name, df in frames.items() if channel not in df.columns]
        if missing:
            print(f"Skipping {channel}: missing in {missing}")
            continue

        butterworth_applied = uses_butterworth(channel)
        dbw_label = (
            "despike + butterworth"
            if butterworth_applied
            else "despike branch reused"
        )
        dbw_color = "tab:blue" if butterworth_applied else "tab:orange"
        delta_dbw_label = (
            "raw - despike + butterworth"
            if butterworth_applied
            else "raw - despike branch reused"
        )
        changed_dbw_label = (
            "changed vs despike + butterworth"
            if butterworth_applied
            else "changed vs despike branch reused"
        )
        text_dbw_label = (
            "raw vs despike+butterworth"
            if butterworth_applied
            else "raw vs despike branch reused"
        )

        raw = pd.to_numeric(frames["raw"].iloc[start:end][channel], errors="coerce")
        despike = pd.to_numeric(frames["despike"].iloc[start:end][channel], errors="coerce")
        dbw = pd.to_numeric(frames["despike_butterworth"].iloc[start:end][channel], errors="coerce")
        x = build_x_axis(frames["raw"], start, end, args.x_mode)

        raw_mask = x.notna() & raw.notna()
        dsp_mask = x.notna() & despike.notna()
        dbw_mask = x.notna() & dbw.notna()

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        axes[0].plot(x[raw_mask], raw[raw_mask], color="0.45", linewidth=1.0, alpha=0.75, label="raw")
        axes[0].plot(x[dsp_mask], despike[dsp_mask], color="tab:orange", linewidth=1.5, label="despike")
        axes[0].plot(x[dbw_mask], dbw[dbw_mask], color=dbw_color, linewidth=1.6, label=dbw_label)
        axes[0].set_title(f"{channel} | {window_label}")
        axes[0].set_ylabel(channel)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        delta_dsp = raw - despike
        delta_dbw = raw - dbw
        delta_dsp_mask = x.notna() & delta_dsp.notna()
        delta_dbw_mask = x.notna() & delta_dbw.notna()
        axes[1].plot(x[delta_dsp_mask], delta_dsp[delta_dsp_mask], color="tab:orange", linewidth=1.2, label="raw - despike")
        axes[1].plot(x[delta_dbw_mask], delta_dbw[delta_dbw_mask], color=dbw_color, linewidth=1.2, label=delta_dbw_label)
        axes[1].axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        axes[1].set_ylabel("delta")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        changed_dsp = x[delta_dsp_mask & (delta_dsp != 0)]
        changed_dbw = x[delta_dbw_mask & (delta_dbw != 0)]
        axes[2].scatter(changed_dsp, delta_dsp[delta_dsp_mask & (delta_dsp != 0)], s=14, alpha=0.75, color="tab:orange", label="changed vs despike")
        axes[2].scatter(changed_dbw, delta_dbw[delta_dbw_mask & (delta_dbw != 0)], s=14, alpha=0.65, color=dbw_color, label=changed_dbw_label)
        axes[2].axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        axes[2].set_ylabel("changed delta")
        axes[2].set_xlabel("time" if args.x_mode == "time" else "global sample index")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        text = (
            "raw vs despike: " + summarize_delta(raw, despike) + "\n"
            f"{text_dbw_label}: " + summarize_delta(raw, dbw)
        )
        axes[0].text(
            0.99,
            0.98,
            text,
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        plt.tight_layout()
        output_path = output_dir / f"run_{args.run_number}_{channel}_{args.dataset_kind}_{start}_{end}.png"
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
