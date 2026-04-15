# Reusing script from our DVC homework. Separating functions/Py scripts for clarity.

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

FEATURES = [
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "RatecodeID",
    "pickup_hour",
]

TARGET = "fare_amount"


def load_parquet_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("No parquet files were provided.")

    dfs = [pd.read_parquet(path) for path in paths]
    return pd.concat(dfs, ignore_index=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"lpep_pickup_datetime", "lpep_dropoff_datetime"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Missing required datetime columns: {sorted(required_cols - set(df.columns))}"
        )

    df = df.copy()

    df["lpep_pickup_datetime"] = pd.to_datetime(
        df["lpep_pickup_datetime"], errors="coerce"
    )
    df["lpep_dropoff_datetime"] = pd.to_datetime(
        df["lpep_dropoff_datetime"], errors="coerce"
    )

    df["trip_duration_min"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed = set(FEATURES + [TARGET])
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    df = df.dropna(subset=[TARGET, "trip_distance", "trip_duration_min"])

    df = df[df[TARGET].between(0.5, 300)]
    df = df[df["trip_distance"].between(0.01, 100)]
    df = df[df["trip_duration_min"].between(0.1, 300)]

    if "passenger_count" in df.columns:
        df = df[df["passenger_count"].between(1, 8)]

    if "RatecodeID" in df.columns:
        df = df[df["RatecodeID"].between(1, 6)]

    if "pickup_hour" in df.columns:
        df = df[df["pickup_hour"].between(0, 23)]

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_derived_features(df)
    df = basic_clean(df)

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    return X, y


def build_model(
    random_state: int = 69,
    n_estimators: int = 50,  # default, going to apply grid search on it later; reduced from 300 -> due to OOM error
    max_depth: int | None = None,  # default, gonna mlflow this
    min_samples_leaf: int = 2,  # default, ditto
) -> Pipeline:
    preprocessor = SimpleImputer(strategy="median")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,  # reduced to 1 (from -1) to avoid OOM
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    return Pipeline(
        steps=[
            ("imputer", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }
