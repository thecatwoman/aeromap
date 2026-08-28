import argparse
import csv
import json
from pathlib import Path
import re


DEFAULT_AUDIT_CSV = Path(
    "data/processed/pitlane_validation_batch_logs/frozen_audits/baseline_v1/pitlane_validation_audit.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/ml_datasets/baseline_v1")
DEFAULT_LOG_ROOT = Path("data/processed/pitlane_validation_batch_logs")
DEFAULT_TRAINING_DIR = DEFAULT_OUTPUT_DIR / "training_exports"
DEFAULT_REPORT_DIR = DEFAULT_OUTPUT_DIR / "reports"
DEFAULT_BAND_LEVEL_DIR = DEFAULT_OUTPUT_DIR / "band_level"

DIAGNOSTIC_FLAG_IDS = {
    "band_consistent": 0,
    "basis_shift_between_bands": 1,
    "cross_band_conflict": 2,
}

BASIS_IDS = {
    "within_threshold": 0,
    "higher_than_sim": 1,
    "lower_than_sim": -1,
    "higher_than_pair": 2,
    "lower_than_pair": -2,
    "unknown": 99,
}

STATE_IDS = {
    "within_threshold": 0,
    "higher_than_sim": 1,
    "lower_than_sim": -1,
    "higher_than_pair": 2,
    "lower_than_pair": -2,
    "comparable": 3,
    "unavailable": 99,
    "unknown": 100,
}

CONFIDENCE_IDS = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "unavailable": 99,
}

FIT_STATUS_IDS = {
    "match": 1,
    "mismatch": 0,
    "unknown": 99,
}

BOOL_WORD_IDS = {
    "true": 1,
    "false": 0,
    "unavailable": 99,
    "unknown": 100,
}

PUSH_TARGET = "push_offset_required"
RH_TARGET = "rh_offset_required"
IDENTIFIER_COLUMNS = {"run", "channel_set", "segment_label"}
DECISION_LEAKAGE_COLUMNS = {
    "official_decision_band",
    "official_offset_decision",
    "official_offset_action",
    "official_offset_reason",
    "official_offset_required_binary",
    "official_offset_channel_count",
    "official_offset_channels",
    "rh_offset_required",
    "diagnostic_flag",
    "diagnostic_flag_id",
    "rh_band_status",
    "push_band_status",
    "slow_fast_rh_action_match",
    "slow_fast_push_action_match",
    "slow_fast_rh_basis_match",
    "slow_fast_push_basis_match",
    "slow_rh_action",
    "slow_push_action",
    "fast_rh_action",
    "fast_push_action",
    "slow_rh_basis",
    "slow_push_basis",
    "fast_rh_basis",
    "fast_push_basis",
    "slow_rh_basis_id",
    "slow_push_basis_id",
    "fast_rh_basis_id",
    "fast_push_basis_id",
    "slow_rh_outcome",
    "slow_push_outcome",
    "fast_rh_outcome",
    "fast_push_outcome",
    "slow_final_outcome",
    "fast_final_outcome",
}

CHANNEL_PUSH_SIGNAL = {
    "fl": "pushavg_c",
    "fr": "pushavd_c",
    "rl": "pusharg_c",
    "rr": "pushard_c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a first ML-ready dataset from a frozen pitlane audit CSV. "
            "This export keeps the frozen audit as source-of-truth and writes derived labels/features separately."
        )
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
        help="Frozen audit CSV to use as source-of-truth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the derived ML dataset and metadata will be written.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Batch log root containing run_<N>/<channel>_<band>.log files.",
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help="Directory where leakage-safe training exports will be written.",
    )
    parser.add_argument(
        "--report-output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where dataset readiness and offset-label reports will be written.",
    )
    parser.add_argument(
        "--band-level-output-dir",
        type=Path,
        default=DEFAULT_BAND_LEVEL_DIR,
        help="Directory where band-level datasets and reports will be written.",
    )
    return parser.parse_args()


def normalize_basis(value: str) -> str:
    text = (value or "").strip()
    return text if text else "unknown"


def normalize_token(value: str) -> str:
    text = (value or "").strip()
    return text if text else "unknown"


def sanitize_name(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("%", "pct")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def parse_scalar(value: str) -> float | str:
    text = value.strip().rstrip(",")
    if text == "unavailable":
        return "unavailable"
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return value.strip()


def scalar_feature(prefix: str, name: str, value: float | str) -> dict[str, str | int | float]:
    key = f"{prefix}_{sanitize_name(name)}"
    if isinstance(value, float):
        return {key: value}
    token = normalize_token(str(value))
    feature: dict[str, str | int | float] = {key: token}
    if key.endswith("_state") or key.endswith("_basis"):
        feature[f"{key}_id"] = STATE_IDS.get(token, STATE_IDS["unknown"])
    elif key.endswith("_confidence"):
        feature[f"{key}_id"] = CONFIDENCE_IDS.get(token, CONFIDENCE_IDS["unavailable"])
    elif key.endswith("_status"):
        feature[f"{key}_id"] = FIT_STATUS_IDS.get(token, FIT_STATUS_IDS["unknown"])
    elif token in BOOL_WORD_IDS:
        feature[f"{key}_id"] = BOOL_WORD_IDS[token]
    return feature


def parse_key_values(text: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for part in text.split(","):
        chunk = part.strip()
        if "=" not in chunk:
            continue
        raw_key, raw_value = chunk.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        # Skip arithmetic display fragments like:
        # "median 37.2000 - 37.4418 = -0.2418"
        # These are parsed separately into a clean `median_delta` feature.
        if key.lower().startswith("median "):
            continue
        name = sanitize_name(key)
        result[name] = parse_scalar(value)
    return result


def extract_median_delta(text: str) -> float | None:
    match = re.search(r"median\s+[^=]+=\s*([^,]+)", text)
    if not match:
        return None
    value = parse_scalar(match.group(1))
    return value if isinstance(value, float) else None


def parse_selected_group_line(line: str) -> dict[str, float | str]:
    data = parse_key_values(line)
    speed_match = re.search(
        r"speed\[min/median/max\]=([^/]+)/([^/]+)/([^\s]+)",
        line,
    )
    if speed_match:
        data["speed_min"] = parse_scalar(speed_match.group(1))
        data["speed_median"] = parse_scalar(speed_match.group(2))
        data["speed_max"] = parse_scalar(speed_match.group(3))
    if "samples" in data:
        data["samples"] = parse_scalar(str(data["samples"]))
    if "group" in data:
        data["group"] = parse_scalar(str(data["group"]))
    indices_match = re.search(r"indices=([0-9]+)->([0-9]+)", line)
    if indices_match:
        data["index_start"] = float(indices_match.group(1))
        data["index_end"] = float(indices_match.group(2))
    return data


def parse_metric_payload(text: str) -> tuple[str, str, dict[str, float | str]]:
    prefix, rest = text.split(":", 1)
    prefix = prefix.strip()
    rest = rest.strip()
    subject = rest.split(":", 1)[0].strip()
    if ":" in rest:
        _, payload = rest.split(":", 1)
    else:
        payload = ""
    return sanitize_name(prefix), sanitize_name(subject), parse_key_values(payload)


def parse_plateau_metric_line(text: str) -> tuple[str, str, dict[str, float | str]]:
    category, state, subject, payload = text.split(":", 3)
    category = sanitize_name(category.replace(" plateau", ""))
    state = sanitize_name(state)
    subject = sanitize_name(subject.replace(" plateau", ""))
    data = parse_key_values(payload)
    return category, state, subject, data


def parse_simple_metric_line(line: str) -> tuple[str, dict[str, float | str]]:
    label, payload = line.split(":", 1)
    data = parse_key_values(payload)
    median_delta = extract_median_delta(payload)
    if median_delta is not None:
        data["median_delta"] = median_delta
    return sanitize_name(label), data


def parse_evidence_fit_line(line: str) -> tuple[str, dict[str, float | str]]:
    status, rest = line.split(":", 1)
    relation, payload = rest.split(":", 1)
    data = parse_key_values(payload)
    data["status"] = normalize_token(status)
    return sanitize_name(relation), data


def parse_helper_agreement_line(line: str) -> tuple[str, dict[str, float | str]]:
    label, payload = line.split(":", 1)
    data: dict[str, float | str] = {}
    for part in payload.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[sanitize_name(key)] = normalize_token(value)
    return sanitize_name(label), data


def parse_counts_line(line: str) -> dict[str, float | str]:
    return parse_key_values(line)


def parse_log_features(log_path: Path) -> dict[str, str | int | float]:
    if not log_path.exists():
        return {"log_present": 0}

    features: dict[str, str | int | float] = {"log_present": 1}
    lines = log_path.read_text(encoding="utf-8").splitlines()
    section = ""
    submode = ""
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "Selected pit groups:":
            section = "selected_pit_groups"
            i += 1
            continue
        if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            section = sanitize_name(stripped)
            submode = ""
            i += 2
            continue

        if section == "selected_pit_groups":
            if stripped.startswith("real: ") or stripped.startswith("simu: "):
                role, payload = stripped.split(": ", 1)
                for key, value in parse_selected_group_line(payload).items():
                    features.update(scalar_feature(f"selected_{sanitize_name(role)}", key, value))
                if i + 1 < len(lines) and "quality column=" in lines[i + 1]:
                    for key, value in parse_key_values(lines[i + 1]).items():
                        features.update(scalar_feature(f"selected_{sanitize_name(role)}", key, value))
                    i += 1

        elif section == "synchronization":
            if stripped.startswith("Full run coarse sync: "):
                payload = stripped.split(": ", 1)[1]
                for key, value in parse_key_values(payload).items():
                    features.update(scalar_feature("sync_coarse", key, value))
            elif stripped.startswith("Pit local fine sync: "):
                payload = stripped.split(": ", 1)[1]
                for key, value in parse_key_values(payload).items():
                    features.update(scalar_feature("sync_fine", key, value))

        elif section == "median_comparisons":
            if ":" in stripped:
                state, subject, data = parse_metric_payload(stripped)
                features.update(scalar_feature(f"median_{subject}", "state", state))
                for key, value in data.items():
                    features.update(scalar_feature(f"median_{subject}", key, value))

        elif section == "plateau_detection":
            if stripped.startswith("stable plateau "):
                for key, value in parse_key_values(stripped).items():
                    features.update(scalar_feature("plateau_stable", key, value))
            elif ":" in stripped:
                category, state, subject, data = parse_plateau_metric_line(stripped)
                features.update(scalar_feature(f"plateau_{category}_{subject}", "state", state))
                for key, value in data.items():
                    features.update(scalar_feature(f"plateau_{category}_{subject}", key, value))

        elif section == "condition_comparability":
            if "[comparable]" in stripped or "[FAIL]" in stripped:
                label_part, payload = stripped.split("]: ", 1)
                signal, status = label_part.split("[", 1)
                features.update(scalar_feature(f"comparability_{sanitize_name(signal)}", "state", status.rstrip("]")))
                for key, value in parse_key_values(payload).items():
                    features.update(scalar_feature(f"comparability_{sanitize_name(signal)}", key, value))

        elif section == "raw_progress_aligned_primary_metrics":
            if ":" in stripped and stripped.split(":", 1)[0] in {
                "ride_height",
                "push_load",
                "damper",
                "pitot",
                "pair",
                "tpms",
            }:
                label, data = parse_simple_metric_line(stripped)
                for key, value in data.items():
                    features.update(scalar_feature(f"raw_primary_{label}", key, value))

        elif section == "synchronized_secondary_metrics":
            if ":" in stripped and stripped.split(":", 1)[0] in {"ride_height", "push_load", "damper"}:
                label, data = parse_simple_metric_line(stripped)
                for key, value in data.items():
                    features.update(scalar_feature(f"sync_secondary_{label}", key, value))

        elif section == "tyre_pressure_detail":
            if stripped.startswith("track TPMS - sim TPMS = "):
                match = re.search(r"= ([^ ]+) psi \(threshold ([^ ]+) psi\)", stripped)
                if match:
                    features.update(scalar_feature("tpms_detail", "delta_psi", parse_scalar(match.group(1))))
                    features.update(scalar_feature("tpms_detail", "threshold_psi", parse_scalar(match.group(2))))
            elif stripped.startswith("TPMS delta exceeds threshold"):
                features.update(scalar_feature("tpms_detail", "exceeds_threshold", "true"))

        elif section == "evidence_fits":
            if ":" in stripped and (stripped.startswith("match:") or stripped.startswith("mismatch:")):
                relation, data = parse_evidence_fit_line(stripped)
                for key, value in data.items():
                    features.update(scalar_feature(f"evidence_fit_{relation}", key, value))

        elif section == "fit_quality":
            if ":" in stripped and "fit_quality:" in stripped:
                relation, payload = stripped.split(":", 1)
                _, payload = payload.split("fit_quality:", 1)
                for key, value in parse_key_values(payload).items():
                    features.update(scalar_feature(f"fit_quality_{sanitize_name(relation)}", key, value))

        elif section == "time_series_agreement":
            if ":" in stripped and "time-series:" in stripped:
                label, payload = stripped.split(":", 1)
                _, payload = payload.split("time-series:", 1)
                for key, value in parse_key_values(payload).items():
                    features.update(scalar_feature(f"time_series_{sanitize_name(label)}", key, value))

        elif section == "helper_agreement":
            if stripped.startswith("ride_height:") or stripped.startswith("push_load:"):
                label, data = parse_helper_agreement_line(stripped)
                for key, value in data.items():
                    features.update(scalar_feature(f"helper_{label}", key, value))
            elif stripped.startswith("supports="):
                counts = parse_counts_line(stripped)
                if "supports" in counts:
                    target = "helper_ride_height" if "helper_ride_height_supports" not in features else "helper_push_load"
                    for key, value in counts.items():
                        features.update(scalar_feature(target, key, value))

        elif section == "decision_confidence":
            if stripped.startswith("ride_height:") or stripped.startswith("push_load:"):
                label, value = stripped.split(":", 1)
                features.update(
                    scalar_feature(f"decision_confidence_{sanitize_name(label)}", "confidence", normalize_token(value))
                )

        i += 1

    return features


def parse_official_offset_channels(decision_text: str) -> list[str]:
    prefix = "Approved offsets:"
    text = (decision_text or "").strip()
    if not text.startswith(prefix):
        return []

    payload = text[len(prefix) :].strip()
    if not payload or payload == "none":
        return []

    return [item.strip() for item in payload.split(",") if item.strip()]


def build_ml_row(row: dict[str, str]) -> dict[str, str | int]:
    official_channels = parse_official_offset_channels(row.get("official_offset_decision", ""))
    slow_rh_action = row.get("slow_rh_action", "")
    slow_push_action = row.get("slow_push_action", "")
    fast_rh_action = row.get("fast_rh_action", "")
    fast_push_action = row.get("fast_push_action", "")

    slow_rh_basis = normalize_basis(row.get("slow_rh_basis", ""))
    slow_push_basis = normalize_basis(row.get("slow_push_basis", ""))
    fast_rh_basis = normalize_basis(row.get("fast_rh_basis", ""))
    fast_push_basis = normalize_basis(row.get("fast_push_basis", ""))

    official_offset_required_binary = int(bool(official_channels))
    push_offset_required = int(slow_push_action == "apply_offset")
    rh_offset_required = int(slow_rh_action == "apply_offset")

    return {
        "run": int(row["run"]),
        "channel_set": row["channel_set"],
        "segment_label": row["segment_label"],
        "official_decision_band": row["official_decision_band"],
        "official_offset_decision": row["official_offset_decision"],
        "official_offset_action": row["official_offset_action"],
        "official_offset_reason": row["official_offset_reason"],
        "official_offset_required_binary": official_offset_required_binary,
        "official_offset_channel_count": len(official_channels),
        "official_offset_channels": ",".join(official_channels) if official_channels else "none",
        "push_offset_required": push_offset_required,
        "rh_offset_required": rh_offset_required,
        "diagnostic_flag": row["diagnostic_flag"],
        "diagnostic_flag_id": DIAGNOSTIC_FLAG_IDS[row["diagnostic_flag"]],
        "rh_band_status": row["rh_band_status"],
        "push_band_status": row["push_band_status"],
        "slow_fast_rh_action_match": int(slow_rh_action == fast_rh_action),
        "slow_fast_push_action_match": int(slow_push_action == fast_push_action),
        "slow_fast_rh_basis_match": int(slow_rh_basis == fast_rh_basis),
        "slow_fast_push_basis_match": int(slow_push_basis == fast_push_basis),
        "slow_rh_action": slow_rh_action,
        "slow_push_action": slow_push_action,
        "fast_rh_action": fast_rh_action,
        "fast_push_action": fast_push_action,
        "slow_rh_basis": slow_rh_basis,
        "slow_push_basis": slow_push_basis,
        "fast_rh_basis": fast_rh_basis,
        "fast_push_basis": fast_push_basis,
        "slow_rh_basis_id": BASIS_IDS.get(slow_rh_basis, BASIS_IDS["unknown"]),
        "slow_push_basis_id": BASIS_IDS.get(slow_push_basis, BASIS_IDS["unknown"]),
        "fast_rh_basis_id": BASIS_IDS.get(fast_rh_basis, BASIS_IDS["unknown"]),
        "fast_push_basis_id": BASIS_IDS.get(fast_push_basis, BASIS_IDS["unknown"]),
        "slow_rh_outcome": row["slow_rh_outcome"],
        "slow_push_outcome": row["slow_push_outcome"],
        "fast_rh_outcome": row["fast_rh_outcome"],
        "fast_push_outcome": row["fast_push_outcome"],
        "slow_final_outcome": row["slow_final_outcome"],
        "fast_final_outcome": row["fast_final_outcome"],
    }


def enrich_ml_row(
    base_row: dict[str, str | int],
    source_row: dict[str, str],
    log_root: Path,
) -> dict[str, str | int | float]:
    enriched: dict[str, str | int | float] = dict(base_row)
    run = source_row["run"]
    channel_set = source_row["channel_set"]
    for band in ("slow", "fast"):
        log_path = log_root / f"run_{run}" / f"{channel_set}_{band}.log"
        band_features = parse_log_features(log_path)
        for key, value in band_features.items():
            enriched[f"{band}_{key}"] = value
    return enriched


def write_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows available to export.")

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(output_dir: Path, source_csv: Path, log_root: Path, row_count: int) -> None:
    metadata = {
        "dataset_name": "baseline_v1_ml_dataset",
        "source_audit_csv": str(source_csv),
        "source_log_root": str(log_root),
        "row_count": row_count,
        "primary_targets": [
            "official_offset_required_binary",
            "push_offset_required",
            "rh_offset_required",
        ],
        "secondary_target": "diagnostic_flag",
        "diagnostic_flag_ids": DIAGNOSTIC_FLAG_IDS,
        "basis_ids": BASIS_IDS,
        "notes": [
            "This dataset is derived from frozen baseline_v1 audit labels.",
            "The frozen audit CSV remains the source-of-truth; this file is a derived ML export.",
            "The base export keeps label/context fields only.",
            "The enriched export joins numeric evidence features parsed from the batch slow/fast log files.",
            "Slow-band labels remain the official offset target; fast-band fields remain diagnostic.",
        ],
    }
    metadata_path = output_dir / "baseline_v1_ml_dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def is_decision_leakage_column(name: str) -> bool:
    if name in DECISION_LEAKAGE_COLUMNS:
        return True
    if name.endswith("_reason"):
        return True
    if name.endswith("_action"):
        return True
    if name.endswith("_outcome"):
        return True
    if name.endswith("_basis") or name.endswith("_basis_id"):
        return True
    if name.startswith("decision_confidence_"):
        return True
    return False


def build_push_training_rows(
    enriched_rows: list[dict[str, str | int | float]],
) -> tuple[list[dict[str, str | int | float]], list[str], dict[str, int]]:
    if not enriched_rows:
        raise ValueError("No enriched rows available for training export.")

    feature_columns: list[str] = []
    seen_features: set[str] = set()
    for row in enriched_rows:
        for name in row.keys():
            if (
                name not in IDENTIFIER_COLUMNS
                and name != PUSH_TARGET
                and not is_decision_leakage_column(name)
                and name not in seen_features
            ):
                seen_features.add(name)
                feature_columns.append(name)

    training_rows: list[dict[str, str | int | float]] = []
    class_balance = {"target_0": 0, "target_1": 0}
    for row in enriched_rows:
        target_value = int(row[PUSH_TARGET])
        class_balance[f"target_{target_value}"] += 1
        export_row: dict[str, str | int | float] = {
            "run": row["run"],
            "channel_set": row["channel_set"],
            "segment_label": row["segment_label"],
            PUSH_TARGET: target_value,
        }
        for name in feature_columns:
            export_row[name] = row.get(name, "")
        training_rows.append(export_row)

    return training_rows, feature_columns, class_balance


def write_training_export(
    output_dir: Path,
    training_rows: list[dict[str, str | int | float]],
    feature_columns: list[str],
    class_balance: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_v1_push_training_dataset.csv"
    write_csv(training_rows, csv_path)

    metadata = {
        "dataset_name": "baseline_v1_push_training_dataset",
        "target": PUSH_TARGET,
        "row_count": len(training_rows),
        "identifier_columns": ["run", "channel_set", "segment_label"],
        "target_column": PUSH_TARGET,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "excluded_leakage_columns": sorted(
            name
            for name in training_rows[0].keys()
            if False
        ),
        "class_balance": class_balance,
        "notes": [
            "This export is leakage-safe for the push offset target.",
            "It keeps identifiers for grouped split by run.",
            "It excludes official decisions, per-band decisions, reasons, outcomes, bases, and confidence fields.",
        ],
    }
    # Record exclusion policy explicitly rather than trying to reconstruct it from rows.
    metadata["excluded_leakage_columns"] = sorted(
        {
            name
            for name in DECISION_LEAKAGE_COLUMNS
        }
    ) + [
        "<all *_reason columns>",
        "<all *_action columns>",
        "<all *_outcome columns>",
        "<all *_basis columns>",
        "<all *_basis_id columns>",
        "<all decision_confidence_* columns>",
    ]
    metadata_path = output_dir / "baseline_v1_push_training_dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_float_cell(value: str | int | float | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text == "unavailable":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def derive_push_offset_amount_label(
    row: dict[str, str | int | float],
) -> dict[str, str | int | float]:
    channel_set = str(row["channel_set"])
    signal = CHANNEL_PUSH_SIGNAL[channel_set]
    basis = str(row.get("slow_push_basis", "unknown"))

    plateau_state = str(row.get(f"slow_plateau_push_load_{signal}_state", "unknown"))
    plateau_delta = parse_float_cell(row.get(f"slow_plateau_push_load_{signal}_rolling_median_delta"))
    median_state = str(row.get(f"slow_median_{signal}_state", "unknown"))
    median_delta = parse_float_cell(row.get(f"slow_median_{signal}_rolling_median_delta"))
    raw_delta = parse_float_cell(row.get("slow_raw_primary_push_load_median_delta"))

    selected_delta = None
    selected_source = "unavailable"
    if plateau_delta is not None and plateau_state == basis:
        selected_delta = plateau_delta
        selected_source = "slow_plateau_push_signal_rolling_median_delta"
    elif median_delta is not None and median_state == basis:
        selected_delta = median_delta
        selected_source = "slow_median_push_signal_rolling_median_delta"
    elif raw_delta is not None:
        selected_delta = raw_delta
        selected_source = "slow_raw_primary_push_load_median_delta"

    recommended_real_offset = -selected_delta if selected_delta is not None else ""

    return {
        "run": row["run"],
        "channel_set": channel_set,
        "segment_label": row["segment_label"],
        "push_offset_required": row[PUSH_TARGET],
        "slow_push_basis": basis,
        "push_signal": signal,
        "recommended_real_push_offset": recommended_real_offset,
        "recommended_real_push_offset_abs": abs(recommended_real_offset) if recommended_real_offset != "" else "",
        "recommended_real_push_offset_source": selected_source,
        "slow_plateau_push_signal_state": plateau_state,
        "slow_plateau_push_signal_delta": plateau_delta if plateau_delta is not None else "",
        "slow_median_push_signal_state": median_state,
        "slow_median_push_signal_delta": median_delta if median_delta is not None else "",
        "slow_raw_primary_push_load_median_delta": raw_delta if raw_delta is not None else "",
        "official_offset_decision": row["official_offset_decision"],
        "slow_push_outcome": row["slow_push_outcome"],
    }


def write_push_offset_amount_export(
    output_dir: Path,
    enriched_rows: list[dict[str, str | int | float]],
) -> dict[str, int]:
    positive_rows = [
        derive_push_offset_amount_label(row)
        for row in enriched_rows
        if int(row[PUSH_TARGET]) == 1
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_v1_push_offset_amount_labels.csv"
    write_csv(positive_rows, csv_path)

    with_amount = sum(1 for row in positive_rows if row["recommended_real_push_offset"] != "")
    counts = {
        "positive_rows": len(positive_rows),
        "rows_with_amount_label": with_amount,
        "rows_missing_amount_label": len(positive_rows) - with_amount,
    }
    metadata = {
        "dataset_name": "baseline_v1_push_offset_amount_labels",
        "row_count": len(positive_rows),
        "counts": counts,
        "label_definition": {
            "recommended_real_push_offset": (
                "Signed offset to apply to real push load so real moves toward simulation. "
                "Positive means real should increase; negative means real should decrease."
            ),
            "selection_policy": [
                "Prefer slow plateau push-signal rolling median delta when its state matches slow_push_basis.",
                "Else use slow whole-band push-signal rolling median delta when its state matches slow_push_basis.",
                "Else fall back to slow raw primary push-load median delta.",
            ],
        },
    }
    metadata_path = output_dir / "baseline_v1_push_offset_amount_labels_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return counts


def build_readiness_report(
    ml_rows: list[dict[str, str | int]],
    enriched_rows: list[dict[str, str | int | float]],
    push_class_balance: dict[str, int],
    amount_counts: dict[str, int],
) -> tuple[dict[str, object], str]:
    push_counts = {"0": 0, "1": 0}
    rh_counts = {"0": 0, "1": 0}
    official_counts = {"0": 0, "1": 0}
    diagnostic_counts: dict[str, int] = {}
    channel_push_counts: dict[str, dict[str, int]] = {}
    channel_diag_counts: dict[str, dict[str, int]] = {}

    for row in ml_rows:
        push_value = str(row[PUSH_TARGET])
        rh_value = str(row[RH_TARGET])
        official_value = str(row["official_offset_required_binary"])
        diagnostic = str(row["diagnostic_flag"])
        channel = str(row["channel_set"])

        push_counts[push_value] += 1
        rh_counts[rh_value] += 1
        official_counts[official_value] += 1
        diagnostic_counts[diagnostic] = diagnostic_counts.get(diagnostic, 0) + 1

        channel_push_counts.setdefault(channel, {"0": 0, "1": 0})[push_value] += 1
        channel_diag_counts.setdefault(channel, {})
        channel_diag_counts[channel][diagnostic] = channel_diag_counts[channel].get(diagnostic, 0) + 1

    def is_binary_trainable(counts: dict[str, int]) -> bool:
        return counts.get("0", 0) > 0 and counts.get("1", 0) > 0

    report = {
        "row_count": len(ml_rows),
        "run_count": len({row["run"] for row in ml_rows}),
        "channel_set_count": len({row["channel_set"] for row in ml_rows}),
        "targets": {
            "official_offset_required_binary": {
                "counts": official_counts,
                "trainable": is_binary_trainable(official_counts),
            },
            PUSH_TARGET: {
                "counts": push_counts,
                "trainable": is_binary_trainable(push_counts),
            },
            RH_TARGET: {
                "counts": rh_counts,
                "trainable": is_binary_trainable(rh_counts),
            },
            "diagnostic_flag": {
                "counts": diagnostic_counts,
                "trainable_multiclass": len([v for v in diagnostic_counts.values() if v > 0]) >= 2,
            },
        },
        "per_channel_push_counts": channel_push_counts,
        "per_channel_diagnostic_counts": channel_diag_counts,
        "push_offset_amount_labels": amount_counts,
        "notes": [
            "Readiness is measured only from real frozen validated cases.",
            "No synthetic balancing or invented cases are used.",
            "A binary target is marked trainable only if both classes are present.",
        ],
    }

    lines = [
        "# Dataset Readiness Report: baseline_v1",
        "",
        f"- Rows: `{report['row_count']}`",
        f"- Runs: `{report['run_count']}`",
        f"- Channel sets: `{report['channel_set_count']}`",
        "",
        "## Target Balance",
        "",
        f"- `official_offset_required_binary`: 0={official_counts['0']} 1={official_counts['1']} | trainable={report['targets']['official_offset_required_binary']['trainable']}",
        f"- `push_offset_required`: 0={push_counts['0']} 1={push_counts['1']} | trainable={report['targets'][PUSH_TARGET]['trainable']}",
        f"- `rh_offset_required`: 0={rh_counts['0']} 1={rh_counts['1']} | trainable={report['targets'][RH_TARGET]['trainable']}",
        f"- `diagnostic_flag`: {diagnostic_counts} | trainable_multiclass={report['targets']['diagnostic_flag']['trainable_multiclass']}",
        "",
        "## Push Offset Amount Labels",
        "",
        f"- positive rows: `{amount_counts['positive_rows']}`",
        f"- rows with amount label: `{amount_counts['rows_with_amount_label']}`",
        f"- rows missing amount label: `{amount_counts['rows_missing_amount_label']}`",
        "",
        "## Interpretation",
        "",
    ]
    if not report["targets"][PUSH_TARGET]["trainable"]:
        lines.append(
            "- `push_offset_required` is not trainable yet because the frozen dataset currently contains only one observed class."
        )
    else:
        lines.append("- `push_offset_required` has both classes and is trainable as a binary target.")
    if not report["targets"][RH_TARGET]["trainable"]:
        lines.append(
            "- `rh_offset_required` is not trainable yet because the frozen dataset currently contains only one observed class."
        )
    else:
        lines.append("- `rh_offset_required` has both classes and is trainable as a binary target.")
    if report["targets"]["diagnostic_flag"]["trainable_multiclass"]:
        lines.append("- `diagnostic_flag` has at least two observed classes and can be used as a diagnostic target.")
    else:
        lines.append("- `diagnostic_flag` does not yet have enough observed class diversity for a meaningful multiclass model.")

    return report, "\n".join(lines) + "\n"


def write_readiness_report(
    output_dir: Path,
    report: dict[str, object],
    markdown_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_v1_dataset_readiness_report.json"
    md_path = output_dir / "baseline_v1_dataset_readiness_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_text, encoding="utf-8")


def build_band_level_rows(
    enriched_rows: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    band_rows: list[dict[str, str | int | float]] = []
    for row in enriched_rows:
        for band in ("slow", "fast"):
            band_row: dict[str, str | int | float] = {
                "run": row["run"],
                "channel_set": row["channel_set"],
                "segment_label": row["segment_label"],
                "pit_band": band,
                "push_band_offset_required": int(row[f"{band}_push_action"] == "apply_offset"),
                "rh_band_offset_required": int(row[f"{band}_rh_action"] == "apply_offset"),
                "push_band_action": row[f"{band}_push_action"],
                "rh_band_action": row[f"{band}_rh_action"],
                "push_band_outcome": row[f"{band}_push_outcome"],
                "rh_band_outcome": row[f"{band}_rh_outcome"],
                "push_band_basis": row[f"{band}_push_basis"],
                "rh_band_basis": row[f"{band}_rh_basis"],
            }
            prefix = f"{band}_"
            for key, value in row.items():
                if not key.startswith(prefix):
                    continue
                band_key = key[len(prefix) :]
                band_row[f"band_{band_key}"] = value
            band_rows.append(band_row)
    return band_rows


def build_band_level_push_training_rows(
    band_rows: list[dict[str, str | int | float]],
) -> tuple[list[dict[str, str | int | float]], list[str], dict[str, int]]:
    if not band_rows:
        raise ValueError("No band-level rows available.")

    feature_columns: list[str] = []
    seen: set[str] = set()
    excluded = {
        "run",
        "channel_set",
        "segment_label",
        "pit_band",
        "push_band_offset_required",
        "rh_band_offset_required",
        "push_band_action",
        "rh_band_action",
        "push_band_outcome",
        "rh_band_outcome",
        "push_band_basis",
        "rh_band_basis",
    }
    for row in band_rows:
        for key in row.keys():
            if key in excluded:
                continue
            if key.startswith("band_decision_confidence_"):
                continue
            if key not in seen:
                seen.add(key)
                feature_columns.append(key)

    training_rows: list[dict[str, str | int | float]] = []
    class_balance = {"target_0": 0, "target_1": 0}
    for row in band_rows:
        target = int(row["push_band_offset_required"])
        class_balance[f"target_{target}"] += 1
        export_row: dict[str, str | int | float] = {
            "run": row["run"],
            "channel_set": row["channel_set"],
            "segment_label": row["segment_label"],
            "pit_band": row["pit_band"],
            "push_band_offset_required": target,
        }
        for key in feature_columns:
            export_row[key] = row.get(key, "")
        training_rows.append(export_row)
    return training_rows, feature_columns, class_balance


def derive_band_level_push_amount_label(
    row: dict[str, str | int | float],
) -> dict[str, str | int | float]:
    signal = CHANNEL_PUSH_SIGNAL[str(row["channel_set"])]
    basis = str(row["push_band_basis"])
    plateau_state = str(row.get(f"band_plateau_push_load_{signal}_state", "unknown"))
    plateau_delta = parse_float_cell(row.get(f"band_plateau_push_load_{signal}_rolling_median_delta"))
    median_state = str(row.get(f"band_median_{signal}_state", "unknown"))
    median_delta = parse_float_cell(row.get(f"band_median_{signal}_rolling_median_delta"))
    raw_delta = parse_float_cell(row.get("band_raw_primary_push_load_median_delta"))

    selected_delta = None
    selected_source = "unavailable"
    if plateau_delta is not None and plateau_state == basis:
        selected_delta = plateau_delta
        selected_source = "band_plateau_push_signal_rolling_median_delta"
    elif median_delta is not None and median_state == basis:
        selected_delta = median_delta
        selected_source = "band_median_push_signal_rolling_median_delta"
    elif raw_delta is not None:
        selected_delta = raw_delta
        selected_source = "band_raw_primary_push_load_median_delta"

    recommended_real_offset = -selected_delta if selected_delta is not None else ""

    return {
        "run": row["run"],
        "channel_set": row["channel_set"],
        "segment_label": row["segment_label"],
        "pit_band": row["pit_band"],
        "push_band_offset_required": row["push_band_offset_required"],
        "push_band_basis": basis,
        "push_signal": signal,
        "recommended_real_push_offset": recommended_real_offset,
        "recommended_real_push_offset_abs": abs(recommended_real_offset) if recommended_real_offset != "" else "",
        "recommended_real_push_offset_source": selected_source,
        "band_plateau_push_signal_state": plateau_state,
        "band_plateau_push_signal_delta": plateau_delta if plateau_delta is not None else "",
        "band_median_push_signal_state": median_state,
        "band_median_push_signal_delta": median_delta if median_delta is not None else "",
        "band_raw_primary_push_load_median_delta": raw_delta if raw_delta is not None else "",
        "push_band_outcome": row["push_band_outcome"],
    }


def write_band_level_exports(
    output_dir: Path,
    band_rows: list[dict[str, str | int | float]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "baseline_v1_band_level_dataset.csv"
    write_csv(band_rows, dataset_path)

    training_rows, feature_columns, class_balance = build_band_level_push_training_rows(band_rows)
    training_path = output_dir / "baseline_v1_band_level_push_training_dataset.csv"
    write_csv(training_rows, training_path)

    amount_rows = [
        derive_band_level_push_amount_label(row)
        for row in band_rows
        if int(row["push_band_offset_required"]) == 1
    ]
    amount_path = output_dir / "baseline_v1_band_level_push_amount_labels.csv"
    write_csv(amount_rows, amount_path)

    push_counts = {"0": 0, "1": 0}
    rh_counts = {"0": 0, "1": 0}
    band_counts = {"slow": 0, "fast": 0}
    for row in band_rows:
        push_counts[str(row["push_band_offset_required"])] += 1
        rh_counts[str(row["rh_band_offset_required"])] += 1
        band_counts[str(row["pit_band"])] += 1

    metadata = {
        "row_count": len(band_rows),
        "band_counts": band_counts,
        "push_band_offset_required_counts": push_counts,
        "rh_band_offset_required_counts": rh_counts,
        "push_training_feature_count": len(feature_columns),
        "notes": [
            "One original (run, channel_set, segment) row becomes two rows: slow and fast.",
            "Group split must still be done by run to avoid leakage between related bands.",
        ],
    }
    write_json_path = output_dir / "baseline_v1_band_level_metadata.json"
    write_json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Band-Level Dataset Readiness: baseline_v1",
        "",
        f"- Rows: `{len(band_rows)}`",
        f"- Slow rows: `{band_counts['slow']}`",
        f"- Fast rows: `{band_counts['fast']}`",
        "",
        "## Target Balance",
        "",
        f"- `push_band_offset_required`: 0={push_counts['0']} 1={push_counts['1']}",
        f"- `rh_band_offset_required`: 0={rh_counts['0']} 1={rh_counts['1']}",
        "",
        "## Interpretation",
        "",
        "- `push_band_offset_required` is now trainable if both classes are present.",
        "- `rh_band_offset_required` remains blocked unless both classes are present.",
        "- Rows are correlated within each run, so all validation must group by run.",
    ]
    (output_dir / "baseline_v1_band_level_readiness_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    return {
        "dataset_path": str(dataset_path),
        "training_path": str(training_path),
        "amount_path": str(amount_path),
        "row_count": len(band_rows),
        "push_counts": push_counts,
        "rh_counts": rh_counts,
        "feature_count": len(feature_columns),
    }


def main() -> None:
    args = parse_args()
    if not args.audit_csv.exists():
        raise FileNotFoundError(f"Frozen audit CSV not found: {args.audit_csv}")
    if not args.log_root.exists():
        raise FileNotFoundError(f"Log root not found: {args.log_root}")

    with args.audit_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)

    ml_rows = [build_ml_row(row) for row in source_rows]
    enriched_rows = [
        enrich_ml_row(base_row, source_row, args.log_root)
        for base_row, source_row in zip(ml_rows, source_rows, strict=True)
    ]
    output_csv = args.output_dir / "baseline_v1_ml_dataset.csv"
    enriched_output_csv = args.output_dir / "baseline_v1_ml_dataset_with_features.csv"
    write_csv(ml_rows, output_csv)
    write_csv(enriched_rows, enriched_output_csv)
    write_metadata(args.output_dir, args.audit_csv, args.log_root, len(ml_rows))
    training_rows, feature_columns, class_balance = build_push_training_rows(enriched_rows)
    write_training_export(args.training_output_dir, training_rows, feature_columns, class_balance)
    amount_counts = write_push_offset_amount_export(args.training_output_dir, enriched_rows)
    readiness_report, readiness_markdown = build_readiness_report(
        ml_rows,
        enriched_rows,
        class_balance,
        amount_counts,
    )
    write_readiness_report(args.report_output_dir, readiness_report, readiness_markdown)
    band_level_info = write_band_level_exports(args.band_level_output_dir, build_band_level_rows(enriched_rows))

    print(f"Source audit CSV: {args.audit_csv}")
    print(f"Source log root: {args.log_root}")
    print(f"Rows exported: {len(ml_rows)}")
    print(f"Saved ML dataset: {output_csv}")
    print(f"Saved enriched ML dataset: {enriched_output_csv}")
    print(f"Saved metadata: {args.output_dir / 'baseline_v1_ml_dataset_metadata.json'}")
    print(
        "Saved push training export: "
        f"{args.training_output_dir / 'baseline_v1_push_training_dataset.csv'}"
    )
    print(
        "Saved push training metadata: "
        f"{args.training_output_dir / 'baseline_v1_push_training_dataset_metadata.json'}"
    )
    print(
        "Push target class balance: "
        f"0={class_balance['target_0']} 1={class_balance['target_1']}"
    )
    print(
        "Saved push offset amount labels: "
        f"{args.training_output_dir / 'baseline_v1_push_offset_amount_labels.csv'}"
    )
    print(
        "Saved readiness report: "
        f"{args.report_output_dir / 'baseline_v1_dataset_readiness_report.md'}"
    )
    print(
        "Saved band-level dataset: "
        f"{args.band_level_output_dir / 'baseline_v1_band_level_dataset.csv'}"
    )
    print(
        "Saved band-level push training dataset: "
        f"{args.band_level_output_dir / 'baseline_v1_band_level_push_training_dataset.csv'}"
    )
    print(
        "Band-level push class balance: "
        f"0={band_level_info['push_counts']['0']} 1={band_level_info['push_counts']['1']}"
    )


if __name__ == "__main__":
    main()
