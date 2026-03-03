import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(prog="aeromap")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check environment and basic dependencies")
    args = parser.parse_args()

    if args.cmd == "doctor":
        import pandas  # noqa
        import sklearn  # noqa
        import pyarrow  # noqa
        print("OK: pandas, sklearn, pyarrow imported")
        try:
            import xgboost  # noqa
            print("OK: xgboost imported")
        except Exception as e:
            print("WARN: xgboost not available:", e)

        print("Project root:", Path(__file__).resolve().parents[1])

if __name__ == "__main__":
    main()
