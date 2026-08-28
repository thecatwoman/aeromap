import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_paths import (
    cleaned_merged_full_run_file,
    cleaned_simudata_full_run_file,
    processed_run_dir,
    processed_simudata_dir,
)


@dataclass
class SegmenterConfig:
    speed_col: str = "carspeed_art"
    accx_col: str = "avg_accx"
    accy_col: str = "avg_accy"
    steer_col: str = "steer0_w"
    yaw_col: str = "yaw0_c"
    brake_col: str = "pbrake_tot"
    lap_col: str = "lap"
    dist_col: str = "distancelap"
    time_col: str = "time.1"
    max_segment_speed_range_kmh: float = 5.0
    smooth_seconds: float = 0.25
    stable_window_seconds: float = 1.0
    stable_speed_std_kmh: float = 1.6
    stable_accx_abs: float = 0.40
    stable_speed_grad_abs_kmh_s: float = 7.5
    min_segment_seconds: float = 0.70
    min_segment_points: int = 50
    merge_gap_seconds: float = 0.25
    pit_speed_max: float = 65.0
    pit_accy_max: float = 0.18
    pit_min_seconds: float = 1.0
    straight_accy_max: float = 0.40
    corner_accy_min: float = 0.60
    straight_steer_max: float = 1.0
    corner_steer_min: float = 1.5
    straight_yaw_max: float = 6.0
    corner_yaw_min: float = 12.0
    apply_internal_smoothing: bool = True


class TestSegmenter:
    def __init__(self, config: SegmenterConfig | None = None):
        self.cfg = config or SegmenterConfig()
        self.sample_dt_: float | None = None
        self.thresholds_: dict[str, float] = {}
        self.debug_stats_: dict[str, object] = {}

    def _infer_dt(self, df: pd.DataFrame) -> float:
        c = self.cfg
        if c.dist_col in df.columns and c.speed_col in df.columns:
            dist = pd.to_numeric(df[c.dist_col], errors="coerce")
            sp = pd.to_numeric(df[c.speed_col], errors="coerce") / 3.6
            same = pd.Series(True, index=df.index)
            if c.lap_col in df.columns:
                same = (
                    pd.to_numeric(df[c.lap_col], errors="coerce")
                    .diff()
                    .fillna(0)
                    .eq(0)
                )
            est = (dist.diff() / sp.replace(0, np.nan)).where(same)
            est = est[(est > 0.002) & (est < 0.05)]
            if len(est) > 100:
                return float(est.median())
        if c.time_col in df.columns:
            t = pd.to_numeric(df[c.time_col], errors="coerce")
            d = t.diff().dropna()
            d = d[(d > 0) & (d < d.quantile(0.9))]
            if len(d):
                return float(d.median())
        return 0.01

    @staticmethod
    def _range(s: pd.Series, w: int) -> pd.Series:
        mp = max(3, w // 3)
        return (
            s.rolling(w, center=True, min_periods=mp).max()
            - s.rolling(w, center=True, min_periods=mp).min()
        )

    @staticmethod
    def _close(mask: np.ndarray, max_gap: int) -> np.ndarray:
        mask = mask.astype(bool).copy()
        n = len(mask)
        i = 0
        while i < n:
            if mask[i]:
                i += 1
                continue
            j = i
            while j < n and not mask[j]:
                j += 1
            if i > 0 and j < n and j - i <= max_gap:
                mask[i:j] = True
            i = j
        return mask

    @staticmethod
    def _remove_short(mask: np.ndarray, min_len: int) -> np.ndarray:
        mask = mask.astype(bool).copy()
        n = len(mask)
        i = 0
        while i < n:
            j = i
            while j < n and mask[j] == mask[i]:
                j += 1
            if mask[i] and j - i < min_len:
                mask[i:j] = False
            i = j
        return mask

    @staticmethod
    def _enforce_range(
        mask: np.ndarray,
        speed: pd.Series,
        max_range: float,
        min_len: int,
    ) -> np.ndarray:
        mask = mask.astype(bool)
        speed_arr = np.asarray(speed, float)
        out = np.zeros_like(mask, dtype=bool)
        n = len(mask)
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            start = i
            mn = mx = speed_arr[i]
            k = i + 1
            while k < j:
                mn2 = min(mn, speed_arr[k])
                mx2 = max(mx, speed_arr[k])
                if mx2 - mn2 <= max_range:
                    mn, mx = mn2, mx2
                    k += 1
                else:
                    if k - start >= min_len:
                        out[start:k] = True
                    start = k
                    mn = mx = speed_arr[k]
                    k += 1
            if j - start >= min_len:
                out[start:j] = True
            i = j
        return out

    @staticmethod
    def _segment_ids(labels: np.ndarray) -> np.ndarray:
        ids = np.zeros(len(labels), int)
        sid = 0
        prev = None
        for i, lab in enumerate(labels):
            if lab != prev:
                sid += 1
                prev = lab
            ids[i] = sid
        return ids

    def _pit_mask(self, out: pd.DataFrame, dt: float) -> np.ndarray:
        c = self.cfg
        sw = max(3, int(round(0.35 / dt)))
        stable_w = max(5, int(round(2.0 / dt)))
        min_len = max(1, int(round(1.2 / dt)))
        gap_len = max(1, int(round(0.7 / dt)))
        pit_min_len = max(1, int(round(c.pit_min_seconds / dt)))

        if c.apply_internal_smoothing:
            speed_smooth = (
                out[c.speed_col]
                .astype(float)
                .rolling(sw, center=True, min_periods=1)
                .median()
                .rolling(sw, center=True, min_periods=1)
                .mean()
            )
            accx_smooth = (
                out[c.accx_col].astype(float).rolling(sw, center=True, min_periods=1).median()
            )
            accy_smooth = (
                out[c.accy_col].astype(float).rolling(sw, center=True, min_periods=1).median()
            )
        else:
            speed_smooth = out[c.speed_col].astype(float)
            accx_smooth = out[c.accx_col].astype(float)
            accy_smooth = out[c.accy_col].astype(float)

        speed_range = self._range(speed_smooth, stable_w)
        speed_std = speed_smooth.rolling(
            stable_w, center=True, min_periods=max(3, stable_w // 3)
        ).std()
        speed_grad = (
            speed_smooth.diff().abs().rolling(
                max(3, sw), center=True, min_periods=1
            ).median() / dt
        )

        stable = (
            (speed_range <= 5.0)
            & (speed_std <= 1.35)
            & (accx_smooth.abs() <= 0.16)
            & (speed_grad <= 4.0)
        ).fillna(False).to_numpy()
        stable = self._remove_short(self._close(stable, gap_len), min_len)
        pit = (
            stable
            & (speed_smooth <= c.pit_speed_max).to_numpy()
            & (accy_smooth.abs() <= c.pit_accy_max).to_numpy()
        )
        pit = self._close(pit, gap_len)
        pit = self._remove_short(pit, pit_min_len)
        return pit

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        input_row_count = int(len(out))
        for col in [
            c.speed_col,
            c.accx_col,
            c.accy_col,
            c.steer_col,
            c.yaw_col,
            c.brake_col,
        ]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
                if c.apply_internal_smoothing:
                    out[col] = out[col].interpolate(limit_direction="both")

        dt = self._infer_dt(out)
        self.sample_dt_ = dt
        sw = max(3, round(c.smooth_seconds / dt))
        ww = max(5, round(c.stable_window_seconds / dt))
        min_len = max(c.min_segment_points, round(c.min_segment_seconds / dt))
        gap = max(1, round(c.merge_gap_seconds / dt))

        sp = out[c.speed_col].astype(float)
        if c.apply_internal_smoothing:
            out["speed_smooth"] = (
                sp.rolling(sw, center=True, min_periods=1).median()
                .rolling(sw, center=True, min_periods=1)
                .mean()
            )
            for raw, new in [
                (c.accx_col, "accx_smooth"),
                (c.accy_col, "accy_smooth"),
                (c.steer_col, "steer_smooth"),
                (c.yaw_col, "yaw_smooth"),
            ]:
                out[new] = out[raw].astype(float).rolling(sw, center=True, min_periods=1).median()
        else:
            out["speed_smooth"] = sp
            for raw, new in [
                (c.accx_col, "accx_smooth"),
                (c.accy_col, "accy_smooth"),
                (c.steer_col, "steer_smooth"),
                (c.yaw_col, "yaw_smooth"),
            ]:
                out[new] = out[raw].astype(float)

        out["speed_std"] = out["speed_smooth"].rolling(
            ww, center=True, min_periods=max(3, ww // 3)
        ).std()
        out["speed_range"] = self._range(out["speed_smooth"], ww)
        out["speed_grad"] = (
            out["speed_smooth"].diff().abs().rolling(
                max(3, sw), center=True, min_periods=1
            ).median() / dt
        )

        stable = (
            (out["speed_range"] <= c.max_segment_speed_range_kmh)
            & (out["speed_std"] <= c.stable_speed_std_kmh)
            & (out["accx_smooth"].abs() <= c.stable_accx_abs)
            & (out["speed_grad"] <= c.stable_speed_grad_abs_kmh_s)
        ).fillna(False).to_numpy()
        stable = self._remove_short(self._close(stable, gap), min_len)
        stable = self._enforce_range(
            stable,
            out["speed_smooth"],
            c.max_segment_speed_range_kmh,
            min_len,
        )
        out["stable_speed"] = stable
        stable_speed_count = int(np.count_nonzero(stable))

        pit = self._pit_mask(out, dt)

        th = {
            "straight_accy_max": c.straight_accy_max,
            "straight_yaw_max": c.straight_yaw_max,
            "straight_steer_max": c.straight_steer_max,
            "corner_accy_min": c.corner_accy_min,
            "corner_yaw_min": c.corner_yaw_min,
            "corner_steer_min": c.corner_steer_min,
        }
        self.thresholds_ = th

        aa = out["accy_smooth"].abs()
        ay = out["yaw_smooth"].abs()
        st = out["steer_smooth"].abs()

        straight = (
            out["stable_speed"].to_numpy()
            & ~pit
            & ((aa <= th["straight_accy_max"]) | (ay <= th["straight_yaw_max"])).to_numpy()
            & (st <= th["straight_steer_max"] * 1.5).to_numpy()
        )
        straight = self._enforce_range(
            self._remove_short(self._close(straight, gap), min_len),
            out["speed_smooth"],
            c.max_segment_speed_range_kmh,
            min_len,
        )
        corner = (
            out["stable_speed"].to_numpy()
            & ~pit
            & ~straight
            & (
                (aa >= th["corner_accy_min"])
                | (ay >= th["corner_yaw_min"])
                | (st >= th["corner_steer_min"])
            ).to_numpy()
        )
        corner = self._enforce_range(
            self._remove_short(self._close(corner, gap), min_len),
            out["speed_smooth"],
            c.max_segment_speed_range_kmh,
            min_len,
        )

        labels = np.full(len(out), "transition", object)
        labels[pit] = "pit"
        labels[straight] = "straight"
        labels[corner] = "corner"
        out["segment_test"] = labels
        out["segment_id_test"] = self._segment_ids(labels)

        self.debug_stats_ = {
            "input_row_count": input_row_count,
            "stable_speed_count": stable_speed_count,
            "counts_per_label": {
                str(label): int(count)
                for label, count in out["segment_test"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            },
        }
        return out

    def segments_table(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        rows: list[dict[str, object]] = []
        for sid, g in df.groupby("segment_id_test", sort=True):
            lab = g["segment_test"].iloc[0]
            if lab == "transition":
                continue
            rows.append(
                {
                    "segment_id_test": sid,
                    "segment": lab,
                    "lap_start": int(g[c.lap_col].iloc[0]) if c.lap_col in g else None,
                    "lap_end": int(g[c.lap_col].iloc[-1]) if c.lap_col in g else None,
                    "index_start": int(g.index[0]),
                    "index_end": int(g.index[-1]),
                    "duration_s": round(len(g) * (self.sample_dt_ or 0.01), 3),
                    "distance_start": float(g[c.dist_col].iloc[0]) if c.dist_col in g else np.nan,
                    "distance_end": float(g[c.dist_col].iloc[-1]) if c.dist_col in g else np.nan,
                    "speed_mean": round(float(g[c.speed_col].mean()), 2),
                    "speed_min": round(float(g[c.speed_col].min()), 2),
                    "speed_max": round(float(g[c.speed_col].max()), 2),
                    "speed_range": round(float(g[c.speed_col].max() - g[c.speed_col].min()), 2),
                    "accy_abs_mean": round(float(g[c.accy_col].abs().mean()), 4),
                    "steer_abs_mean": round(float(g[c.steer_col].abs().mean()), 4),
                    "yaw_abs_mean": round(float(g[c.yaw_col].abs().mean()), 4),
                }
            )
        return pd.DataFrame(rows)


def get_source_paths(source: str) -> tuple[Path, Path, Path, Path]:
    if source == "simu":
        base_dir = processed_simudata_dir()
        return (
            cleaned_simudata_full_run_file(),
            base_dir / "barcelona_2026_simudata_segmented_test_run_46.csv",
            base_dir / "barcelona_2026_simudata_segment_summary_test_run_46.csv",
            base_dir / "barcelona_2026_simudata_segment_debug_test_run_46.json",
        )

    base_dir = processed_run_dir() / "Segmented"
    return (
        cleaned_merged_full_run_file(),
        base_dir / "barcelona_2026_merged_cleaned_test_run_46_segmented.csv",
        base_dir / "barcelona_2026_merged_cleaned_segment_summary_test_run_46.csv",
        base_dir / "barcelona_2026_merged_cleaned_segment_debug_test_run_46.json",
    )


def run_segmentation(
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, TestSegmenter, Path, Path, Path]:
    input_path, output_path, summary_path, debug_path = get_source_paths(source)
    df = pd.read_csv(input_path, low_memory=False)
    seg = TestSegmenter(
        SegmenterConfig(
            apply_internal_smoothing=(source != "simu"),
        )
    )
    out = seg.fit_transform(df)
    table = seg.segments_table(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    table.to_csv(summary_path, index=False)
    debug_payload = {
        "source": source,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "input_row_count": int(len(df)),
        "inferred_dt_s": float(seg.sample_dt_ or 0.0),
        "stable_speed_count": int(np.count_nonzero(out["stable_speed"].to_numpy())),
        "thresholds": {key: float(value) for key, value in seg.thresholds_.items()},
        "counts_per_label": {
            str(label): int(count)
            for label, count in out["segment_test"].value_counts(dropna=False).sort_index().items()
        },
        "internal_smoothing_enabled": bool(seg.cfg.apply_internal_smoothing),
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2) + "\n", encoding="utf-8")
    return out, table, seg, output_path, summary_path, debug_path


def plot_segmentation(df: pd.DataFrame, source: str) -> None:
    color_map = {
        "straight": "lime",
        "corner": "red",
        "pit": "magenta",
        "transition": "white",
    }
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(df))
    speed = pd.to_numeric(df["carspeed_art"], errors="coerce")
    ax.plot(x, speed, color="gray", linewidth=0.8, alpha=0.6)
    for label, color in color_map.items():
        mask = df["segment_test"] == label
        if mask.any():
            ax.scatter(x[mask], speed[mask], color=color, s=4, label=label)
    ax.set_title(f"Test segmenter - {source}")
    ax.set_xlabel("global_sample")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the downloaded test segmenter to our telemetry.",
    )
    parser.add_argument("--source", choices=["real", "simu"], default="real")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out, table, seg, output_path, summary_path, debug_path = run_segmentation(args.source)
    print(f"Sample dt: {seg.sample_dt_:.4f}s")
    print("Thresholds:", seg.thresholds_)
    print(
        f"Internal smoothing: {'enabled' if seg.cfg.apply_internal_smoothing else 'disabled'}"
    )
    print(out["segment_test"].value_counts(dropna=False))
    print(f"Saved segmented data to: {output_path}")
    print(f"Saved segment summary to: {summary_path}")
    print(f"Saved debug report to: {debug_path}")
    if args.plot:
        plot_segmentation(out, args.source)


if __name__ == "__main__":
    main()
