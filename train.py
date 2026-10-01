"""Train a leakage-safe passenger survival classifier from a Titanic CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "survived"  # This is the observed outcome to predict.
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]  # Retain fare as a float; do not truncate currency values.
CATEGORICAL_FEATURES = ["sex", "embarked", "alone"]  # Use only pre-voyage attributes, not target proxies such as alive.
REQUIRED_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]  # Enforce a small, documented feature contract.


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load a Titanic-style CSV and remove target-proxy columns by construction."""
    frame = pd.read_csv(path)  # Read the supplied dataset without modifying it.
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))  # Report incompatible inputs clearly.
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")  # Prevent silent feature mistakes.
    frame = frame[REQUIRED_COLUMNS].copy()  # Do not carry `alive`, `class`, or other duplicate/proxy columns forward.
    frame = frame.dropna(subset=[TARGET])  # A label is required for supervised learning.
    frame[TARGET] = frame[TARGET].astype(int)  # Store the binary target as integers.
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The survived column must contain only 0 and 1.")  # Make the classification contract explicit.
    return frame


def make_preprocessor() -> ColumnTransformer:
    """Build transformations that learn only from a pipeline's training folds."""
    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Impute age and fare using training-fold medians.
            ("scaler", StandardScaler()),  # Scale numeric features for logistic regression.
        ]
    )
    category_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing embarkation values safely.
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),  # Avoid artificial order among ports or sex labels.
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_steps, NUMERIC_FEATURES), ("categorical", category_steps, CATEGORICAL_FEATURES)]  # Apply each strategy only to its own columns.
    )


def scoring(y_true: pd.Series, probabilities, predictions) -> dict[str, float]:
    """Return classification metrics beyond raw accuracy."""
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),  # Overall correct predictions.
        "f1_survived": round(float(f1_score(y_true, predictions)), 4),  # Balance precision and recall for survivors.
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),  # Evaluate ranking quality across thresholds.
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)  # Provide a reproducible command-line entry point.
    parser.add_argument("--data", required=True, help="Path to a Titanic CSV")  # Keep external data outside source control.
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model artifacts")  # Keep generated outputs predictable.
    args = parser.parse_args()

    frame = load_and_clean(args.data)  # Remove target leakage risks before splitting.
    X = frame.drop(columns=TARGET)  # Separate passenger features from the label.
    y = frame[TARGET]  # Retain labels in their own series.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Preserve class balance in the test partition.
    )
    candidates = {
        "dummy_prior": DummyClassifier(strategy="prior"),  # Establish the class-frequency baseline.
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),  # Provide an interpretable model.
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1  # Capture non-linear interactions.
        ),
    }
    validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Use stable folds for training-set model comparison.
    results = {}  # Keep cross-validation and held-out metrics for every candidate.
    trained = {}  # Retain fitted pipelines for export.
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])  # Keep preprocessing inside every fold.
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=validation, scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}  # Compare multiple relevant scores.
        )
        pipeline.fit(X_train, y_train)  # Refit the full training set after cross-validation.
        predicted = pipeline.predict(X_test)  # Generate class predictions for held-out rows.
        probability = pipeline.predict_proba(X_test)[:, 1]  # Generate probabilities for AUC.
        results[name] = {
            "cv_mean": {key.replace("test_", ""): round(float(value.mean()), 4) for key, value in cv_scores.items() if key.startswith("test_")},  # Summarise validation performance.
            "test": scoring(y_test, probability, predicted),  # Preserve one untouched final result.
        }
        trained[name] = pipeline  # Keep the deployed pipeline candidate.

    best_name = max(results, key=lambda name: results[name]["cv_mean"]["roc_auc"])  # Choose without optimising against the test set.
    winning_pipeline = trained[best_name]  # Retrieve the model selected by training-fold evidence.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output storage if absent.
    joblib.dump(winning_pipeline, output_dir / "model.joblib")  # Export the full preprocessing-and-model pipeline.
    report = {
        "dataset_rows": len(frame),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_model_by_cv_roc_auc": best_name,
        "results": results,
        "best_test_classification_report": classification_report(y_test, winning_pipeline.predict(X_test), output_dict=True),  # Include per-class performance.
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")  # Save auditable metrics.
    print(json.dumps(report, indent=2))  # Display the evaluation report.


if __name__ == "__main__":
    main()  # Run training only for direct execution.
