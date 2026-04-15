from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

# Using stuff from the "model_utils.py" file (a module), where "src" acts as a package
from src.model_utils import (
    build_model,
    evaluate_model,
    load_parquet_files,
    prepare_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-files",
        nargs="+",
        default=[
            "data/green_tripdata_2021-01.parquet",
            # "data/green_tripdata_2021-02.parquet", # removed this due to OOM...
            # "data/green_tripdata_2021-03.parquet", # commented out for less memory usage too. Moreover, in practice, the months start to make a difference in the price too
        ],
        help="List of parquet files to combine",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=69)

    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"),
    )

    parser.add_argument("--experiment-name", default="nyc-green-taxi-regression")

    parser.add_argument("--run-name", default="rf-single-run")
    parser.add_argument("--grid-search", action="store_true")

    parser.add_argument("--n-estimators", type=int, default=50)  # Reduced due to OOM
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=2)

    parser.add_argument("--outdir", default="artifacts")

    return parser.parse_args()


def load_and_split_data(
    data_files: list[str],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    df = load_parquet_files(data_files)
    rows_before_cleaning = len(df)

    X, y = prepare_features(df)
    rows_after_cleaning = len(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    info = {
        "rows_before_cleaning": rows_before_cleaning,
        "rows_after_cleaning": rows_after_cleaning,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    return X_train, X_test, y_train, y_test, info


def run_single_experiment(args, X_train, X_test, y_train, y_test, data_info) -> None:
    max_depth = None if args.max_depth == 0 else args.max_depth

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("data_files", json.dumps(args.data_files))
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", max_depth if max_depth is not None else "None")
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)

        for key, value in data_info.items():
            mlflow.log_metric(key, value)

        model = build_model(  # RandomForestRegressor used here as compromise between LR and XGB
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
        )
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        active_run = mlflow.active_run()
        run_id = active_run.info.run_id

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.set_tag("run_model_uri", f"runs:/{run_id}/model")
        print(f"Run model URI: runs:/{run_id}/model", flush=True)

        summary = {
            "mode": "single_run",
            "data_files": args.data_files,
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": args.min_samples_leaf,
            },
            "metrics": metrics,
            **data_info,
        }

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "single_run_summary.json").write_text(json.dumps(summary, indent=2))

        print("\nSingle run completed")
        print(json.dumps(summary, indent=2))


def run_grid_search(args, X_train, X_test, y_train, y_test, data_info) -> None:
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    n_estimators_grid = [100, 300]
    max_depth_grid = [None, 10, 20]
    min_samples_leaf_grid = [1, 2, 4]

    combinations = list(
        itertools.product(
            n_estimators_grid,
            max_depth_grid,
            min_samples_leaf_grid,
        )
    )

    best_rmse = float("inf")
    best_result = None
    all_results = []

    print("\nGrid search combinations:")
    for idx, combo in enumerate(combinations, start=1):
        print(
            f"{idx:02d}: n_estimators={combo[0]}, max_depth={combo[1]}, min_samples_leaf={combo[2]}"
        )

    for n_estimators, max_depth, min_samples_leaf in combinations:
        run_name = f"rf-grid-ne{n_estimators}-md{max_depth}-msl{min_samples_leaf}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("search_mode", "grid")
            mlflow.log_param("data_files", json.dumps(args.data_files))
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param(
                "max_depth", max_depth if max_depth is not None else "None"
            )
            mlflow.log_param("min_samples_leaf", min_samples_leaf)

            for key, value in data_info.items():
                mlflow.log_metric(key, value)

            model = build_model(
                random_state=args.random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            active_run = mlflow.active_run()
            run_id = active_run.info.run_id

            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.set_tag("run_model_uri", f"runs:/{run_id}/model")
            print(f"Run model URI: runs:/{run_id}/model", flush=True)

            improved = metrics["rmse"] < best_rmse
            if improved:
                best_rmse = metrics["rmse"]
                best_result = {
                    "run_name": run_name,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "metrics": metrics,
                }

            result = {
                "run_name": run_name,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "metrics": metrics,
                "improved_best_rmse": improved,
            }
            all_results.append(result)

            print(
                f"Run={run_name} | RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f} | "
                f"Improved best={improved}"
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": "grid_search",
        "data_files": args.data_files,
        "best_result": best_result,
        "all_results": all_results,
        **data_info,
    }
    (outdir / "grid_search_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nBest grid-search result:")
    print(json.dumps(best_result, indent=2))


def main() -> None:
    args = parse_args()

    X_train, X_test, y_train, y_test, data_info = load_and_split_data(
        args.data_files,
        args.test_size,
        args.random_state,
    )

    print("Loaded and split data")
    print(json.dumps(data_info, indent=2))

    if args.grid_search:
        run_grid_search(args, X_train, X_test, y_train, y_test, data_info)
    else:
        run_single_experiment(args, X_train, X_test, y_train, y_test, data_info)


if __name__ == "__main__":
    main()
