import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.apply_butterworth_filter import (
    BUTTERWORTH_TARGET_COLUMNS,
    CHANNEL_CUTOFFS_HZ,
    RIDE_HEIGHT_BASE_CUTOFF_HZ,
    RIDE_HEIGHT_CORNER_CUTOFF_HZ,
)
from src.data.apply_despike_filter import (
    RIDE_HEIGHT_MAD_K,
    RIDE_HEIGHT_COLUMNS,
    RIDE_HEIGHT_WINDOW,
)
from src.run_paths import (
    CURRENT_RUN,
    cleaned_merged_full_run_file,
    processed_run_dir,
    raw_run_dir,
    scz_dir,
    segmented_rh_run_file,
)
from src.segment_with_macroway import SEGMENTATION_KWARGS


BASELINE_SEGMENTER = "macroway"
BASELINE_SEGMENT_LABEL = "corner"
FROZEN_BASELINE_DIRNAME = "Frozen_Baseline"


@dataclass
class BaselineManifest:
    run_number: int
    segmenter: str
    segment_label: str
    raw_run_dir: str
    raw_scz_dir: str
    cleaned_merged_dataset: str
    segmented_dataset: str
    corner_table_front: str
    corner_table_rear: str
    front_plot_png: str
    front_plot_html: str
    rear_plot_png: str
    rear_plot_html: str
    frozen_baseline_dir: str
    frozen_outputs_manifest: str
    frozen_settings_json: str


@dataclass
class FrozenBaselineSettings:
    run_number: int
    segmenter: str
    segment_label: str
    segmentation_kwargs: dict
    despike_columns: list[str]
    despike_settings: dict
    butterworth_target_columns: list[str]
    butterworth_cutoff_hz_by_column: dict[str, float]
    no_baseline_filter_columns: list[str]


def run_step(description: str, command: list[str]) -> None:
    print(f"\n[baseline] {description}")
    print("[baseline] Command:", " ".join(command))
    subprocess.run(command, check=True)


def working_artifact_paths() -> dict[str, Path]:
    corner_dir = Path("data/processed/tables/corner_maps")
    plot_dir = Path("data/processed/plots/reference_table")
    return {
        "cleaned_merged_dataset": cleaned_merged_full_run_file(),
        "segmented_dataset": segmented_rh_run_file(),
        "corner_table_front": corner_dir
        / f"run_{CURRENT_RUN}_corner_scz_push_f_pitot_{BASELINE_SEGMENTER}_two_way_table_exact.csv",
        "corner_table_rear": corner_dir
        / f"run_{CURRENT_RUN}_corner_scz_push_r_pitot_{BASELINE_SEGMENTER}_two_way_table_exact.csv",
        "front_plot_png": plot_dir
        / f"run_{CURRENT_RUN}_scz_push_f_pitot_{BASELINE_SEGMENTER}_overlay_shifted_axes.png",
        "front_plot_html": plot_dir
        / f"run_{CURRENT_RUN}_scz_push_f_pitot_{BASELINE_SEGMENTER}_overlay_shifted_axes.html",
        "rear_plot_png": plot_dir
        / f"run_{CURRENT_RUN}_scz_push_r_pitot_{BASELINE_SEGMENTER}_overlay_shifted_axes.png",
        "rear_plot_html": plot_dir
        / f"run_{CURRENT_RUN}_scz_push_r_pitot_{BASELINE_SEGMENTER}_overlay_shifted_axes.html",
    }


def frozen_baseline_dir() -> Path:
    return processed_run_dir() / FROZEN_BASELINE_DIRNAME


def write_settings_snapshot(settings_path: Path) -> None:
    settings = FrozenBaselineSettings(
        run_number=CURRENT_RUN,
        segmenter=BASELINE_SEGMENTER,
        segment_label=BASELINE_SEGMENT_LABEL,
        segmentation_kwargs=SEGMENTATION_KWARGS,
        despike_columns=RIDE_HEIGHT_COLUMNS,
        despike_settings={
            "ride_height_window": RIDE_HEIGHT_WINDOW,
            "ride_height_mad_k": RIDE_HEIGHT_MAD_K,
        },
        butterworth_target_columns=BUTTERWORTH_TARGET_COLUMNS,
        butterworth_cutoff_hz_by_column={
            "rh_f": RIDE_HEIGHT_BASE_CUTOFF_HZ,
            "rh_r": RIDE_HEIGHT_CORNER_CUTOFF_HZ,
            **CHANNEL_CUTOFFS_HZ,
        },
        no_baseline_filter_columns=["pair"],
    )
    settings_path.write_text(
        json.dumps(asdict(settings), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[baseline] Wrote settings snapshot: {settings_path}")


def freeze_outputs(freeze_dir: Path) -> tuple[dict[str, str], Path, Path]:
    artifacts = working_artifact_paths()
    datasets_dir = freeze_dir / "datasets"
    tables_dir = freeze_dir / "tables"
    plots_dir = freeze_dir / "plots"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    frozen_paths: dict[str, str] = {}
    for key, source_path in artifacts.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Baseline artifact missing: {source_path}")

        if "dataset" in key:
            target_path = datasets_dir / source_path.name
        elif "table" in key:
            target_path = tables_dir / source_path.name
        else:
            target_path = plots_dir / source_path.name

        shutil.copy2(source_path, target_path)
        frozen_paths[key] = str(target_path)
        print(f"[baseline] Frozen {key}: {target_path}")

    settings_path = freeze_dir / "baseline_settings.json"
    write_settings_snapshot(settings_path)

    outputs_manifest_path = freeze_dir / "frozen_outputs_manifest.json"
    outputs_manifest_path.write_text(
        json.dumps(frozen_paths, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[baseline] Wrote frozen outputs manifest: {outputs_manifest_path}")
    return frozen_paths, outputs_manifest_path, settings_path


def write_manifest(manifest_path: Path) -> None:
    freeze_dir = frozen_baseline_dir()
    frozen_paths, outputs_manifest_path, settings_path = freeze_outputs(freeze_dir)

    manifest = BaselineManifest(
        run_number=CURRENT_RUN,
        segmenter=BASELINE_SEGMENTER,
        segment_label=BASELINE_SEGMENT_LABEL,
        raw_run_dir=str(raw_run_dir()),
        raw_scz_dir=str(scz_dir()),
        cleaned_merged_dataset=str(cleaned_merged_full_run_file()),
        segmented_dataset=str(segmented_rh_run_file()),
        corner_table_front=frozen_paths["corner_table_front"],
        corner_table_rear=frozen_paths["corner_table_rear"],
        front_plot_png=frozen_paths["front_plot_png"],
        front_plot_html=frozen_paths["front_plot_html"],
        rear_plot_png=frozen_paths["rear_plot_png"],
        rear_plot_html=frozen_paths["rear_plot_html"],
        frozen_baseline_dir=str(freeze_dir),
        frozen_outputs_manifest=str(outputs_manifest_path),
        frozen_settings_json=str(settings_path),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    print(f"\n[baseline] Wrote manifest: {manifest_path}")


def main() -> None:
    python = sys.executable
    manifest_path = processed_run_dir() / "baseline_real_pipeline_manifest.json"

    run_step(
        "Build cleaned merged real dataset with Scz merge, recalculated front/rear push Scz, and automatic Butterworth filtering",
        [python, "-m", "src.data.build_dataset", "--source", "real"],
    )
    run_step(
        "Run official baseline segmentation on real data",
        [python, "-m", "src.segment_with_macroway", "--source", "real"],
    )
    run_step(
        "Extract official corner tables for recalculated front/rear push Scz",
        [python, "-m", "src.data.build_push_corner_tables", "--segmenter", BASELINE_SEGMENTER],
    )
    run_step(
        "Create official reference-map comparison plots and interactive HTML",
        [python, "-m", "src.data.plot_push_corner_3d", "--side", "both", "--segmenter", BASELINE_SEGMENTER],
    )

    write_manifest(manifest_path)
    print("\n[baseline] Trusted real-data baseline pipeline completed.")


if __name__ == "__main__":
    main()
