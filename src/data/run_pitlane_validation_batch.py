import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CHANNEL_SETS = ("fl", "fr", "rl", "rr")
DEFAULT_PIT_SPEED_BANDS = ("slow", "fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trusted baseline pipeline and/or pitlane decision tree across "
            "multiple runs, channel sets, and pit speed bands."
        )
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        required=True,
        help="Run numbers to process, e.g. --runs 46 47 48",
    )
    parser.add_argument(
        "--channel-sets",
        nargs="+",
        default=list(DEFAULT_CHANNEL_SETS),
        choices=list(DEFAULT_CHANNEL_SETS),
        help="Corner channel sets to validate.",
    )
    parser.add_argument(
        "--pit-speed-bands",
        nargs="+",
        default=list(DEFAULT_PIT_SPEED_BANDS),
        choices=["slow", "fast"],
        help="Pit speed bands to validate independently.",
    )
    parser.add_argument(
        "--segment-label",
        default="pit",
        choices=["pit", "straight", "corner"],
        help="Segment label to pass into the decision tree.",
    )
    parser.add_argument(
        "--min-pit-band-samples",
        type=int,
        default=50,
        help="Minimum samples required inside each pit speed band.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline_real_pipeline step.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately if any baseline or pitlane validation command fails.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/processed/pitlane_validation_batch_logs"),
        help="Directory where per-command logs will be written.",
    )
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    run_number: int,
    log_path: Path,
) -> tuple[int, str]:
    env = os.environ.copy()
    env["AEROMAP_RUN"] = str(run_number)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout
    if result.stderr:
        output = f"{output}\n[stderr]\n{result.stderr}" if output else f"[stderr]\n{result.stderr}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    return result.returncode, output


def parse_named_block(text: str, block_name: str) -> dict[str, str]:
    lines = text.splitlines()
    target = block_name.strip().lower()

    for index, line in enumerate(lines):
        if line.strip().lower() != target:
            continue

        parsed: dict[str, str] = {}
        cursor = index + 1
        while cursor < len(lines):
            current = lines[cursor]
            stripped = current.strip()
            if not stripped or not current.startswith("  "):
                break
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                parsed[key.strip().lower()] = value.strip()
            cursor += 1
        return parsed

    return {}


def parse_critical_channel_summary(text: str, channel_name: str) -> dict[str, str]:
    lines = text.splitlines()
    target = channel_name.strip().lower()

    for index, line in enumerate(lines):
        if not line.lower().startswith(f"{target}: "):
            continue

        parsed: dict[str, str] = {"header": line.strip()}
        cursor = index + 1
        while cursor < len(lines):
            current = lines[cursor]
            stripped = current.strip()
            if not stripped or not current.startswith("  "):
                break
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                parsed[key.strip().lower()] = value.strip()
            cursor += 1
        return parsed

    return {}


def derive_band_status(
    slow_block: dict[str, str],
    fast_block: dict[str, str],
) -> str:
    slow_action = slow_block.get("action", "unknown")
    fast_action = fast_block.get("action", "unknown")
    slow_basis = slow_block.get("actual decision basis", "unknown")
    fast_basis = fast_block.get("actual decision basis", "unknown")

    if slow_action == fast_action and slow_basis == fast_basis:
        return "band_consistent"
    if slow_action == fast_action:
        return "basis_shift_between_bands"
    return "cross_band_conflict"


def build_audit_row(
    *,
    run_number: int,
    channel_set: str,
    segment_label: str,
    slow_text: str,
    fast_text: str,
) -> dict[str, str | int]:
    slow_rh_decision = parse_named_block(slow_text, "ride height")
    slow_push_decision = parse_named_block(slow_text, "push load")
    slow_final_decision = parse_named_block(slow_text, "final")
    fast_rh_decision = parse_named_block(fast_text, "ride height")
    fast_push_decision = parse_named_block(fast_text, "push load")
    fast_final_decision = parse_named_block(fast_text, "final")

    slow_rh_summary = parse_critical_channel_summary(slow_text, "ride height")
    slow_push_summary = parse_critical_channel_summary(slow_text, "push load")
    fast_rh_summary = parse_critical_channel_summary(fast_text, "ride height")
    fast_push_summary = parse_critical_channel_summary(fast_text, "push load")

    rh_status = derive_band_status(
        slow_rh_summary | slow_rh_decision,
        fast_rh_summary | fast_rh_decision,
    )
    push_status = derive_band_status(
        slow_push_summary | slow_push_decision,
        fast_push_summary | fast_push_decision,
    )

    if "cross_band_conflict" in {rh_status, push_status}:
        diagnostic_flag = "cross_band_conflict"
    elif "basis_shift_between_bands" in {rh_status, push_status}:
        diagnostic_flag = "basis_shift_between_bands"
    else:
        diagnostic_flag = "band_consistent"

    row: dict[str, str | int] = {
        "run": run_number,
        "channel_set": channel_set,
        "segment_label": segment_label,
        "official_decision_band": "slow",
        "official_offset_decision": slow_final_decision.get("outcome", ""),
        "official_offset_action": slow_final_decision.get("action", ""),
        "official_offset_reason": slow_final_decision.get("reason", ""),
        "diagnostic_flag": diagnostic_flag,
        "rh_band_status": rh_status,
        "push_band_status": push_status,
        "slow_rh_outcome": slow_rh_decision.get("outcome", ""),
        "slow_rh_action": slow_rh_decision.get("action", ""),
        "slow_rh_reason": slow_rh_decision.get("reason", ""),
        "slow_rh_basis": slow_rh_summary.get("actual decision basis", ""),
        "slow_push_outcome": slow_push_decision.get("outcome", ""),
        "slow_push_action": slow_push_decision.get("action", ""),
        "slow_push_reason": slow_push_decision.get("reason", ""),
        "slow_push_basis": slow_push_summary.get("actual decision basis", ""),
        "slow_final_outcome": slow_final_decision.get("outcome", ""),
        "slow_final_action": slow_final_decision.get("action", ""),
        "slow_final_reason": slow_final_decision.get("reason", ""),
        "fast_rh_outcome": fast_rh_decision.get("outcome", ""),
        "fast_rh_action": fast_rh_decision.get("action", ""),
        "fast_rh_reason": fast_rh_decision.get("reason", ""),
        "fast_rh_basis": fast_rh_summary.get("actual decision basis", ""),
        "fast_push_outcome": fast_push_decision.get("outcome", ""),
        "fast_push_action": fast_push_decision.get("action", ""),
        "fast_push_reason": fast_push_decision.get("reason", ""),
        "fast_push_basis": fast_push_summary.get("actual decision basis", ""),
        "fast_final_outcome": fast_final_decision.get("outcome", ""),
        "fast_final_action": fast_final_decision.get("action", ""),
        "fast_final_reason": fast_final_decision.get("reason", ""),
    }
    return row


def write_audit_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    python = sys.executable
    failures: list[tuple[int, str, Path]] = []
    audit_rows: list[dict[str, str | int]] = []

    for run_number in args.runs:
        print(f"\n=== Run {run_number} ===")
        run_band_outputs: dict[tuple[str, str], str] = {}

        if not args.skip_baseline:
            baseline_log = args.log_dir / f"run_{run_number}" / "baseline_real_pipeline.log"
            print("[batch] Running baseline_real_pipeline")
            code, _ = run_command(
                [python, "-m", "src.data.baseline_real_pipeline"],
                run_number=run_number,
                log_path=baseline_log,
            )
            print(f"[batch] baseline_real_pipeline -> exit {code}")
            if code != 0:
                failures.append((run_number, "baseline_real_pipeline", baseline_log))
                if args.stop_on_failure:
                    break
                continue

        for pit_speed_band in args.pit_speed_bands:
            for channel_set in args.channel_sets:
                label = f"{channel_set}_{pit_speed_band}"
                log_path = args.log_dir / f"run_{run_number}" / f"{label}.log"
                command = [
                    python,
                    "-m",
                    "src.data.pitlane_decision_tree",
                    "--channel-set",
                    channel_set,
                    "--segment-label",
                    args.segment_label,
                    "--pit-speed-band",
                    pit_speed_band,
                    "--min-pit-band-samples",
                    str(args.min_pit_band_samples),
                ]
                print(f"[batch] Running {label}")
                code, _ = run_command(
                    command,
                    run_number=run_number,
                    log_path=log_path,
                )
                print(f"[batch] {label} -> exit {code}")
                if code != 0:
                    failures.append((run_number, label, log_path))
                    if args.stop_on_failure:
                        break
                else:
                    run_band_outputs[(channel_set, pit_speed_band)] = log_path.read_text(encoding="utf-8")
            if args.stop_on_failure and failures:
                break
        if args.stop_on_failure and failures:
            break

        if not args.stop_on_failure or not failures:
            for channel_set in args.channel_sets:
                slow_text = run_band_outputs.get((channel_set, "slow"))
                fast_text = run_band_outputs.get((channel_set, "fast"))
                if slow_text is None or fast_text is None:
                    continue
                audit_rows.append(
                    build_audit_row(
                        run_number=run_number,
                        channel_set=channel_set,
                        segment_label=args.segment_label,
                        slow_text=slow_text,
                        fast_text=fast_text,
                    )
                )

    print("\n=== Batch Summary ===")
    if not failures:
        audit_csv_path = args.log_dir / "pitlane_validation_audit.csv"
        write_audit_csv(audit_rows, audit_csv_path)
        print("All commands completed successfully.")
        print(f"Logs written under: {args.log_dir}")
        print(f"Audit CSV written to: {audit_csv_path}")
        return

    if audit_rows:
        audit_csv_path = args.log_dir / "pitlane_validation_audit_partial.csv"
        write_audit_csv(audit_rows, audit_csv_path)
        print(f"Partial audit CSV written to: {audit_csv_path}")
    print("Failures detected:")
    for run_number, label, log_path in failures:
        print(f"- Run {run_number} | {label} | log: {log_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
