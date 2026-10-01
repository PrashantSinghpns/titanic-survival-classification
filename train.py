"""Train a leakage-safe passenger survival classifier from a Titanic CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluation import evaluate_and_export
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "survived"
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked", "alone"]
REQUIRED_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate numeric inputs and make categorical missing values sklearn-compatible."""
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column]).any():
            raise ValueError(f"Feature {column} contains an infinite value.")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        frame[column] = frame[column].replace("", np.nan).astype(object)
        frame[column] = frame[column].where(frame[column].notna(), np.nan)
    return frame


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load a Titanic-style CSV and remove target-proxy columns by construction."""
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame = frame.dropna(subset=[TARGET])
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The survived column must contain only 0 and 1.")
    frame[TARGET] = frame[TARGET].astype(int)
    return normalize_features(frame)


def make_preprocessor() -> ColumnTransformer:
    """Build transformations that learn only from a pipeline's training folds."""
    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    category_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_steps, NUMERIC_FEATURES), ("categorical", category_steps, CATEGORICAL_FEATURES)]
    )



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the documented CSV schema")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    frame = load_and_clean(args.data)
    candidates = {"dummy_prior": DummyClassifier(strategy="prior"),
                  "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                  "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                          class_weight="balanced", random_state=42, n_jobs=-1)}
    report = evaluate_and_export(frame, TARGET, make_preprocessor, candidates, args.data,
                                 args.output_dir, classification=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
