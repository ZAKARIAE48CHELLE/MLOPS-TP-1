import os

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import argparse
import sys
import warnings
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    """Calculate regression metrics: RMSE, MAE, R2."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data(data_path=None):
    """Load dataset from local CSV file or fallback to Scikit-Learn Wine dataset."""
    if data_path and os.path.exists(data_path):
        print(f"Loading data from local path: {data_path}")
        return pd.read_csv(data_path, sep=None, engine="python")

    # Scikit-learn wine dataset as default reliable source
    wine = load_wine(as_frame=True)
    df = wine.frame
    # Use 'target' as target variable
    df.rename(columns={"target": "quality"}, inplace=True)
    return df


def train(alpha=0.01, l1_ratio=0.1, data_path=None, experiment_name="WineQuality-ElasticNet", run_name=None):
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Set default run name if none provided
    if not run_name:
        run_name = f"ElasticNet_alpha_{alpha}_l1_{l1_ratio}"

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # 1. Load data
    data = load_data(data_path)

    # 2. Split dataset into train and test sets
    train_df, test_df = train_test_split(data, test_size=0.25, random_state=42)

    target_col = "quality"
    train_x = train_df.drop([target_col], axis=1)
    test_x = test_df.drop([target_col], axis=1)
    train_y = train_df[[target_col]]
    test_y = test_df[[target_col]]

    # 3. Start MLflow run with custom run name
    with mlflow.start_run(run_name=run_name):
        print(f"Starting MLflow Run: '{run_name}' in experiment: '{experiment_name}'")

        # Train ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Predict & Evaluate
        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")

        # Log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model with signature and input example
        signature = infer_signature(train_x, lr.predict(train_x))
        input_example = train_x.iloc[:5]

        # Log model artifact with signature and input example
        mlflow.sklearn.log_model(
            sk_model=lr,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        print(f"Run '{run_name}' logged successfully to MLflow!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ElasticNet model with MLflow tracking")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha hyperparameter (default: 0.01)")
    parser.add_argument("--l1-ratio", type=float, default=0.1, help="L1 ratio hyperparameter (default: 0.1)")
    parser.add_argument("--data-path", type=str, default=None, help="Optional path to local dataset CSV")
    parser.add_argument("--experiment", type=str, default="WineQuality-ElasticNet", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")

    args = parser.parse_args()

    # Prompt user for run name if not provided via command line
    run_name = args.run_name
    if not run_name:
        try:
            user_input = input("Enter a name for this MLflow run (press Enter for default): ").strip()
            if user_input:
                run_name = user_input
        except (EOFError, KeyboardInterrupt):
            pass

    train(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        data_path=args.data_path,
        experiment_name=args.experiment,
        run_name=run_name,
    )
