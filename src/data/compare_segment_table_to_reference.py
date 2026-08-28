import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.data.plot_reference_table_3d import X_VALUES as FRONT_X_VALUES, Y_VALUES as FRONT_Y_VALUES, Z_VALUES as FRONT_Z_VALUES
from src.data.plot_reference_table_3d_scz_r import X_VALUES as REAR_X_VALUES, Y_VALUES as REAR_Y_VALUES, Z_VALUES as REAR_Z_VALUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sparse segment-map points against the front or rear reference surface.",
    )
    parser.add_argument("--primary-table", type=Path, required=True, help="Primary exact sparse table CSV.")
    parser.add_argument("--compare-table", type=Path, default=None, help="Optional second exact sparse table CSV.")
    parser.add_argument("--side", choices=["front", "rear"], required=True, help="Reference surface side.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-point reference-distance results.",
    )
    return parser.parse_args()


def reference_grid(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side == "front":
        return FRONT_X_VALUES, FRONT_Y_VALUES, FRONT_Z_VALUES
    return REAR_X_VALUES, REAR_Y_VALUES, REAR_Z_VALUES


def bilinear_interpolate(x: float, y: float, x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray) -> float:
    if x < float(x_values.min()) or x > float(x_values.max()) or y < float(y_values.min()) or y > float(y_values.max()):
        raise ValueError(f"Point ({x}, {y}) is outside the reference surface domain.")

    x_hi = int(np.searchsorted(x_values, x, side="right"))
    y_hi = int(np.searchsorted(y_values, y, side="right"))
    x_hi = min(max(x_hi, 1), len(x_values) - 1)
    y_hi = min(max(y_hi, 1), len(y_values) - 1)
    x_lo = x_hi - 1
    y_lo = y_hi - 1

    x1 = float(x_values[x_lo])
    x2 = float(x_values[x_hi])
    y1 = float(y_values[y_lo])
    y2 = float(y_values[y_hi])

    z11 = float(z_values[y_lo, x_lo])
    z21 = float(z_values[y_lo, x_hi])
    z12 = float(z_values[y_hi, x_lo])
    z22 = float(z_values[y_hi, x_hi])

    if x2 == x1 and y2 == y1:
        return z11
    if x2 == x1:
        ty = 0.0 if y2 == y1 else (y - y1) / (y2 - y1)
        return z11 * (1.0 - ty) + z12 * ty
    if y2 == y1:
        tx = (x - x1) / (x2 - x1)
        return z11 * (1.0 - tx) + z21 * tx

    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)
    return (
        z11 * (1.0 - tx) * (1.0 - ty)
        + z21 * tx * (1.0 - ty)
        + z12 * (1.0 - tx) * ty
        + z22 * tx * ty
    )


def load_sparse_points(table_path: Path) -> pd.DataFrame:
    table = pd.read_csv(table_path, index_col=0)
    rows: list[dict[str, float]] = []
    for y_label, row in table.iterrows():
        y_value = float(y_label)
        for x_label, cell in row.items():
            if pd.isna(cell):
                continue
            rows.append({"rh_f": float(x_label), "rh_r": y_value, "value": float(cell)})
    if not rows:
        raise ValueError(f"No numeric points in {table_path}")
    return pd.DataFrame(rows).sort_values(["rh_r", "rh_f"]).reset_index(drop=True)


def summarize(name: str, frame: pd.DataFrame) -> None:
    abs_distance = frame["abs_distance_to_reference"]
    signed_distance = frame["signed_distance_to_reference"]
    print(f"{name}:")
    print(f"  points={len(frame)}")
    print(f"  mean signed distance={signed_distance.mean():.6f}")
    print(f"  median signed distance={signed_distance.median():.6f}")
    print(f"  mean abs distance={abs_distance.mean():.6f}")
    print(f"  median abs distance={abs_distance.median():.6f}")
    print(f"  max abs distance={abs_distance.max():.6f}")


def main() -> None:
    args = parse_args()
    x_values, y_values, z_values = reference_grid(args.side)

    primary = load_sparse_points(args.primary_table)
    primary["reference_value"] = [
        bilinear_interpolate(float(x), float(y), x_values, y_values, z_values)
        for x, y in zip(primary["rh_f"], primary["rh_r"], strict=False)
    ]
    primary["signed_distance_to_reference"] = primary["value"] - primary["reference_value"]
    primary["abs_distance_to_reference"] = primary["signed_distance_to_reference"].abs()
    primary["table_role"] = "primary"
    summarize("primary", primary)

    output_frames = [primary]

    if args.compare_table is not None:
        compare = load_sparse_points(args.compare_table)
        compare["reference_value"] = [
            bilinear_interpolate(float(x), float(y), x_values, y_values, z_values)
            for x, y in zip(compare["rh_f"], compare["rh_r"], strict=False)
        ]
        compare["signed_distance_to_reference"] = compare["value"] - compare["reference_value"]
        compare["abs_distance_to_reference"] = compare["signed_distance_to_reference"].abs()
        compare["table_role"] = "compare"
        summarize("compare", compare)

        merged = primary.merge(
            compare,
            on=["rh_f", "rh_r"],
            how="outer",
            suffixes=("_primary", "_compare"),
        )
        comparable = merged.dropna(
            subset=["abs_distance_to_reference_primary", "abs_distance_to_reference_compare"]
        ).copy()
        if not comparable.empty:
            comparable["abs_distance_improvement"] = (
                comparable["abs_distance_to_reference_compare"] - comparable["abs_distance_to_reference_primary"]
            )
            print("comparison:")
            print(f"  shared points={len(comparable)}")
            print(
                "  mean abs-distance improvement (positive means primary is closer)="
                f"{comparable['abs_distance_improvement'].mean():.6f}"
            )
            print(
                "  median abs-distance improvement (positive means primary is closer)="
                f"{comparable['abs_distance_improvement'].median():.6f}"
            )
        output_frames.append(compare)

    if args.output_csv is not None:
        out = pd.concat(output_frames, ignore_index=True)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv, index=False)
        print(f"Saved comparison CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
