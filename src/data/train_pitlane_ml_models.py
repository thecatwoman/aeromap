import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.build_pitlane_ml_dataset import (
    DECISION_LEAKAGE_COLUMNS,
    DIAGNOSTIC_FLAG_IDS,
    IDENTIFIER_COLUMNS,
    PUSH_TARGET,
    RH_TARGET,
    is_decision_leakage_column,
)


DEFAULT_ML_DIR = Path("data/processed/ml_datasets/baseline_v1")
DEFAULT_ENRICHED_DATASET = DEFAULT_ML_DIR / "baseline_v1_ml_dataset_with_features.csv"
DEFAULT_PUSH_AMOUNT_LABELS = DEFAULT_ML_DIR / "training_exports" / "baseline_v1_push_offset_amount_labels.csv"
DEFAULT_BAND_LEVEL_DATASET = DEFAULT_ML_DIR / "band_level" / "baseline_v1_band_level_push_training_dataset.csv"
DEFAULT_BAND_LEVEL_AMOUNT_LABELS = DEFAULT_ML_DIR / "band_level" / "baseline_v1_band_level_push_amount_labels.csv"
DEFAULT_OUTPUT_DIR = Path("data/processed/ml_models/baseline_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train v1 pitlane ML models from the frozen baseline_v1 datasets. "
            "This script only trains models for targets that are genuinely trainable in the frozen data."
        )
    )
    parser.add_argument(
        "--enriched-dataset",
        type=Path,
        default=DEFAULT_ENRICHED_DATASET,
        help="Path to the enriched baseline_v1 ML dataset.",
    )
    parser.add_argument(
        "--push-amount-labels",
        type=Path,
        default=DEFAULT_PUSH_AMOUNT_LABELS,
        help="Path to the push offset amount label table.",
    )
    parser.add_argument(
        "--band-level-dataset",
        type=Path,
        default=DEFAULT_BAND_LEVEL_DATASET,
        help="Path to the band-level push training dataset.",
    )
    parser.add_argument(
        "--band-level-amount-labels",
        type=Path,
        default=DEFAULT_BAND_LEVEL_AMOUNT_LABELS,
        help="Path to the band-level push amount label table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where trained models and reports will be written.",
    )
    return parser.parse_args()


def make_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [
        name for name in feature_columns if pd.api.types.is_numeric_dtype(df[name])
    ]
    categorical_columns = [name for name in feature_columns if name not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_diagnostic_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS)
    excluded.update(
        {
            "official_offset_required_binary",
            PUSH_TARGET,
            RH_TARGET,
            "diagnostic_flag",
            "diagnostic_flag_id",
        }
    )
    feature_columns: list[str] = []
    for name in df.columns:
        if name in excluded:
            continue
        if is_decision_leakage_column(name):
            continue
        feature_columns.append(name)
    return feature_columns


def build_push_amount_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS)
    excluded.update(
        {
            "official_offset_required_binary",
            PUSH_TARGET,
            RH_TARGET,
            "diagnostic_flag",
            "diagnostic_flag_id",
            "recommended_real_push_offset",
            "recommended_real_push_offset_abs",
            "recommended_real_push_offset_source",
            "push_signal",
            "slow_push_basis",
            "slow_push_outcome",
        }
    )
    feature_columns: list[str] = []
    for name in df.columns:
        if name in excluded:
            continue
        if is_decision_leakage_column(name):
            continue
        feature_columns.append(name)
    return feature_columns


def evaluate_diagnostic_model(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    feature_columns = build_diagnostic_feature_columns(df)
    groups = df["run"].astype(int).to_numpy()
    y = df["diagnostic_flag_id"].astype(int).to_numpy()
    observed_classes = sorted(np.unique(y).tolist())

    if len(observed_classes) < 2:
        report = {
            "target": "diagnostic_flag",
            "status": "skipped",
            "reason": "Only one observed class in frozen dataset.",
            "observed_classes": observed_classes,
        }
        return report

    X = df[feature_columns].copy()
    preprocessor = make_preprocessor(df, feature_columns)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, float | int]] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_run": int(groups[test_idx][0]),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            }
        )

    accuracy = float(accuracy_score(y_true_all, y_pred_all))
    balanced_accuracy = float(balanced_accuracy_score(y_true_all, y_pred_all))
    f1_macro = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))

    model.fit(X, y)
    model_path = output_dir / "diagnostic_flag_model.joblib"
    joblib.dump(model, model_path)

    report = {
        "target": "diagnostic_flag",
        "status": "trained",
        "row_count": int(len(df)),
        "group_count": int(df["run"].nunique()),
        "feature_count": int(len(feature_columns)),
        "observed_classes": observed_classes,
        "class_mapping": DIAGNOSTIC_FLAG_IDS,
        "cv_strategy": "LeaveOneGroupOut(run)",
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1_macro": f1_macro,
        },
        "folds": fold_rows,
        "model_path": str(model_path),
        "feature_columns": feature_columns,
    }
    return report


def build_push_amount_training_frame(
    enriched_df: pd.DataFrame,
    amount_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = enriched_df.merge(
        amount_df,
        on=["run", "channel_set", "segment_label"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged[merged["recommended_real_push_offset"].notna()].copy()
    merged = merged[merged["recommended_real_push_offset"] != ""].copy()
    merged["recommended_real_push_offset"] = merged["recommended_real_push_offset"].astype(float)
    return merged


def evaluate_push_amount_model(
    enriched_df: pd.DataFrame,
    amount_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    train_df = build_push_amount_training_frame(enriched_df, amount_df)
    feature_columns = build_push_amount_feature_columns(train_df)

    if train_df.empty:
        return {
            "target": "recommended_real_push_offset",
            "status": "skipped",
            "reason": "No rows with amount labels available.",
        }

    groups = train_df["run"].astype(int).to_numpy()
    y = train_df["recommended_real_push_offset"].astype(float).to_numpy()
    X = train_df[feature_columns].copy()

    preprocessor = make_preprocessor(train_df, feature_columns)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                ),
            ),
        ]
    )

    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, float | int]] = []
    y_true_all: list[float] = []
    y_pred_all: list[float] = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_run": int(groups[test_idx][0]),
                "n_test": int(len(test_idx)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": rmse,
                "r2": float(r2_score(y_test, y_pred)),
            }
        )

    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    r2 = float(r2_score(y_true_all, y_pred_all))

    model.fit(X, y)
    model_path = output_dir / "push_offset_amount_model.joblib"
    joblib.dump(model, model_path)

    report = {
        "target": "recommended_real_push_offset",
        "status": "trained",
        "row_count": int(len(train_df)),
        "group_count": int(train_df["run"].nunique()),
        "feature_count": int(len(feature_columns)),
        "cv_strategy": "LeaveOneGroupOut(run)",
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "folds": fold_rows,
        "model_path": str(model_path),
        "feature_columns": feature_columns,
    }
    return report


def build_band_level_push_feature_columns(df: pd.DataFrame) -> list[str]:
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
        "recommended_real_push_offset",
        "recommended_real_push_offset_abs",
        "recommended_real_push_offset_source",
        "push_signal",
    }
    feature_columns: list[str] = []
    for name in df.columns:
        if name in excluded:
            continue
        if name.startswith("band_decision_confidence_"):
            continue
        feature_columns.append(name)
    return feature_columns


def evaluate_band_level_push_classifier(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    feature_columns = build_band_level_push_feature_columns(df)
    groups = df["run"].astype(int).to_numpy()
    y = df["push_band_offset_required"].astype(int).to_numpy()
    observed_classes = sorted(np.unique(y).tolist())

    if len(observed_classes) < 2:
        return {
            "target": "push_band_offset_required",
            "status": "skipped",
            "reason": "Only one observed class in band-level dataset.",
            "observed_classes": observed_classes,
        }

    X = df[feature_columns].copy()
    preprocessor = make_preprocessor(df, feature_columns)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, float | int]] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_run": int(groups[test_idx][0]),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            }
        )

    accuracy = float(accuracy_score(y_true_all, y_pred_all))
    balanced_accuracy = float(balanced_accuracy_score(y_true_all, y_pred_all))
    f1_macro = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))

    model.fit(X, y)
    model_path = output_dir / "push_band_offset_required_model.joblib"
    joblib.dump(model, model_path)

    return {
        "target": "push_band_offset_required",
        "status": "trained",
        "row_count": int(len(df)),
        "group_count": int(df["run"].nunique()),
        "feature_count": int(len(feature_columns)),
        "observed_classes": observed_classes,
        "cv_strategy": "LeaveOneGroupOut(run)",
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1_macro": f1_macro,
        },
        "folds": fold_rows,
        "model_path": str(model_path),
        "feature_columns": feature_columns,
    }


def build_band_level_push_amount_training_frame(
    band_df: pd.DataFrame,
    amount_df: pd.DataFrame,
) -> pd.DataFrame:
    label_frame = amount_df[
        ["run", "channel_set", "segment_label", "pit_band", "recommended_real_push_offset"]
    ].copy()
    label_frame = label_frame[label_frame["recommended_real_push_offset"].notna()].copy()
    label_frame = label_frame[label_frame["recommended_real_push_offset"] != ""].copy()
    label_frame["recommended_real_push_offset"] = label_frame["recommended_real_push_offset"].astype(float)

    keys = ["run", "channel_set", "segment_label", "pit_band"]
    key_tuples = {
        tuple(row[key] for key in keys)
        for row in label_frame.to_dict(orient="records")
    }
    filtered = band_df[
        band_df.apply(lambda row: tuple(row[key] for key in keys) in key_tuples, axis=1)
    ].copy()
    merged = filtered.merge(label_frame, on=keys, how="inner", validate="one_to_one")
    return merged


def evaluate_band_level_push_amount_model(
    band_df: pd.DataFrame,
    amount_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    train_df = build_band_level_push_amount_training_frame(band_df, amount_df)
    feature_columns = build_band_level_push_feature_columns(train_df)

    if train_df.empty:
        return {
            "target": "band_recommended_real_push_offset",
            "status": "skipped",
            "reason": "No band-level rows with amount labels available.",
        }

    groups = train_df["run"].astype(int).to_numpy()
    y = train_df["recommended_real_push_offset"].astype(float).to_numpy()
    X = train_df[feature_columns].copy()
    preprocessor = make_preprocessor(train_df, feature_columns)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )

    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, float | int]] = []
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_run": int(groups[test_idx][0]),
                "n_test": int(len(test_idx)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": float(r2_score(y_test, y_pred)),
            }
        )

    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    r2 = float(r2_score(y_true_all, y_pred_all))

    model.fit(X, y)
    model_path = output_dir / "push_band_offset_amount_model.joblib"
    joblib.dump(model, model_path)

    return {
        "target": "band_recommended_real_push_offset",
        "status": "trained",
        "row_count": int(len(train_df)),
        "group_count": int(train_df["run"].nunique()),
        "feature_count": int(len(feature_columns)),
        "cv_strategy": "LeaveOneGroupOut(run)",
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "folds": fold_rows,
        "model_path": str(model_path),
        "feature_columns": feature_columns,
    }


def build_summary_markdown(
    diagnostic_report: dict,
    amount_report: dict,
    band_push_report: dict,
    band_amount_report: dict,
) -> str:
    lines = [
        "# ML Training Report: baseline_v1",
        "",
        "## Diagnostic Model",
        "",
        f"- status: `{diagnostic_report['status']}`",
    ]
    if diagnostic_report["status"] == "trained":
        metrics = diagnostic_report["metrics"]
        lines.extend(
            [
                f"- rows: `{diagnostic_report['row_count']}`",
                f"- groups: `{diagnostic_report['group_count']}`",
                f"- features: `{diagnostic_report['feature_count']}`",
                f"- accuracy: `{metrics['accuracy']:.4f}`",
                f"- balanced_accuracy: `{metrics['balanced_accuracy']:.4f}`",
                f"- f1_macro: `{metrics['f1_macro']:.4f}`",
                f"- model: `{diagnostic_report['model_path']}`",
            ]
        )
    else:
        lines.append(f"- reason: {diagnostic_report['reason']}")

    lines.extend(
        [
            "",
            "## Push Offset Amount Model",
            "",
            f"- status: `{amount_report['status']}`",
        ]
    )
    if amount_report["status"] == "trained":
        metrics = amount_report["metrics"]
        lines.extend(
            [
                f"- rows: `{amount_report['row_count']}`",
                f"- groups: `{amount_report['group_count']}`",
                f"- features: `{amount_report['feature_count']}`",
                f"- mae: `{metrics['mae']:.4f}`",
                f"- rmse: `{metrics['rmse']:.4f}`",
                f"- r2: `{metrics['r2']:.4f}`",
                f"- model: `{amount_report['model_path']}`",
            ]
        )
    else:
        lines.append(f"- reason: {amount_report['reason']}")

    lines.extend(
        [
            "",
            "## Band-Level Push Classifier",
            "",
            f"- status: `{band_push_report['status']}`",
        ]
    )
    if band_push_report["status"] == "trained":
        metrics = band_push_report["metrics"]
        lines.extend(
            [
                f"- rows: `{band_push_report['row_count']}`",
                f"- groups: `{band_push_report['group_count']}`",
                f"- features: `{band_push_report['feature_count']}`",
                f"- accuracy: `{metrics['accuracy']:.4f}`",
                f"- balanced_accuracy: `{metrics['balanced_accuracy']:.4f}`",
                f"- f1_macro: `{metrics['f1_macro']:.4f}`",
                f"- model: `{band_push_report['model_path']}`",
            ]
        )
    else:
        lines.append(f"- reason: {band_push_report['reason']}")

    lines.extend(
        [
            "",
            "## Band-Level Push Amount Model",
            "",
            f"- status: `{band_amount_report['status']}`",
        ]
    )
    if band_amount_report["status"] == "trained":
        metrics = band_amount_report["metrics"]
        lines.extend(
            [
                f"- rows: `{band_amount_report['row_count']}`",
                f"- groups: `{band_amount_report['group_count']}`",
                f"- features: `{band_amount_report['feature_count']}`",
                f"- mae: `{metrics['mae']:.4f}`",
                f"- rmse: `{metrics['rmse']:.4f}`",
                f"- r2: `{metrics['r2']:.4f}`",
                f"- model: `{band_amount_report['model_path']}`",
            ]
        )
    else:
        lines.append(f"- reason: {band_amount_report['reason']}")

    lines.extend(
        [
            "",
            "## Operational Note",
            "",
            "- `push_offset_required` and `rh_offset_required` classifiers are not trained here unless both classes exist in frozen validated data.",
            "- The diagnostic model can be trained now because the frozen dataset contains both `band_consistent` and `cross_band_conflict`.",
            "- The push offset amount model is trained only on cases already labeled `push_offset_required = 1`.",
            "- The band-level push classifier uses `slow` and `fast` as separate correlated samples, but validation still groups by run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    enriched_df = load_csv(args.enriched_dataset)
    amount_df = load_csv(args.push_amount_labels)
    band_df = load_csv(args.band_level_dataset)
    band_amount_df = load_csv(args.band_level_amount_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_report = evaluate_diagnostic_model(
        enriched_df,
        args.output_dir,
    )
    amount_report = evaluate_push_amount_model(
        enriched_df,
        amount_df,
        args.output_dir,
    )
    band_push_report = evaluate_band_level_push_classifier(
        band_df,
        args.output_dir,
    )
    band_amount_report = evaluate_band_level_push_amount_model(
        band_df,
        band_amount_df,
        args.output_dir,
    )

    summary = {
        "dataset": str(args.enriched_dataset),
        "amount_labels": str(args.push_amount_labels),
        "diagnostic_model": diagnostic_report,
        "push_offset_amount_model": amount_report,
        "band_level_dataset": str(args.band_level_dataset),
        "band_level_amount_labels": str(args.band_level_amount_labels),
        "band_level_push_classifier": band_push_report,
        "band_level_push_amount_model": band_amount_report,
    }
    write_json(args.output_dir / "baseline_v1_ml_training_report.json", summary)
    write_markdown(
        args.output_dir / "baseline_v1_ml_training_report.md",
        build_summary_markdown(
            diagnostic_report,
            amount_report,
            band_push_report,
            band_amount_report,
        ),
    )

    print(f"Saved training report: {args.output_dir / 'baseline_v1_ml_training_report.md'}")
    print(f"Saved training report JSON: {args.output_dir / 'baseline_v1_ml_training_report.json'}")
    if diagnostic_report["status"] == "trained":
        print(f"Saved diagnostic model: {diagnostic_report['model_path']}")
    else:
        print(f"Diagnostic model skipped: {diagnostic_report['reason']}")
    if amount_report["status"] == "trained":
        print(f"Saved push offset amount model: {amount_report['model_path']}")
    else:
        print(f"Push offset amount model skipped: {amount_report['reason']}")
    if band_push_report["status"] == "trained":
        print(f"Saved band-level push classifier: {band_push_report['model_path']}")
    else:
        print(f"Band-level push classifier skipped: {band_push_report['reason']}")
    if band_amount_report["status"] == "trained":
        print(f"Saved band-level push amount model: {band_amount_report['model_path']}")
    else:
        print(f"Band-level push amount model skipped: {band_amount_report['reason']}")


if __name__ == "__main__":
    main()
