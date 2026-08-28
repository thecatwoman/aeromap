import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sum two sparse corner two-way tables cell by cell.",
    )
    parser.add_argument(
        "--front-table",
        type=Path,
        default=Path("data/processed/tables/corner_maps/corner_scz_f_map_macroway_two_way_table_exact.csv"),
        help="Sparse front-table CSV.",
    )
    parser.add_argument(
        "--rear-table",
        type=Path,
        default=Path("data/processed/tables/corner_maps/corner_scz_r_map_macroway_two_way_table_exact.csv"),
        help="Sparse rear-table CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tables/corner_maps/corner_scz_total_map_macroway_two_way_table_exact.csv"),
        help="Output CSV for the summed sparse table.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimals for saved values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    front = pd.read_csv(args.front_table, index_col=0)
    rear = pd.read_csv(args.rear_table, index_col=0)

    front.index = front.index.map(float)
    rear.index = rear.index.map(float)
    front.columns = [float(col) for col in front.columns]
    rear.columns = [float(col) for col in rear.columns]

    combined_index = sorted(set(front.index).union(rear.index))
    combined_columns = sorted(set(front.columns).union(rear.columns))

    front = front.reindex(index=combined_index, columns=combined_columns)
    rear = rear.reindex(index=combined_index, columns=combined_columns)

    mask = front.notna() & rear.notna()
    total = front.add(rear, fill_value=0.0)
    total = total.where(mask)
    total = total.round(args.decimals)
    total.index.name = "rh_r"
    total.columns.name = "rh_f"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(args.output, float_format=f"%.{args.decimals}f")
    print(f"Saved summed sparse table: {args.output}")


if __name__ == "__main__":
    main()
