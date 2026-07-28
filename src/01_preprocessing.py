import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Paths — anchored to this script's location (not the current working
# directory), so the script runs the same whether invoked as
# `python 01_preprocessing.py` from src/ or `python ml_pipline/src/01_preprocessing.py`
# from the project root.
BASE_DIR = Path(__file__).resolve().parent.parent

# Config
CONFIG = {
    # Can be overridden by env variables
    "clean_data_path": os.getenv(
        "CLEAN_DATA_PATH",
        str(BASE_DIR / "data" / "processed" / "df_clean.csv"),
    ),
    "processed_dir": os.getenv(
        "PROCESSED_DIR",
        str(BASE_DIR / "data" / "processed"),
    ),
    "models_dir": os.getenv(
        "MODELS_DIR",
        str(BASE_DIR / "models"),
    ),
    # Split
    "test_size": 0.2,
    "random_state": 42,
    # Target column
    "target": "Churn",
    # Feature groups 
    "numerical_features": [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ],
    "binary_features": [
        # Already 0/1 integers in df_clean,
        # but they need scaling for linear numbers.
        "SeniorCitizen",
    ],
    "categorical_features": [
        # Will be one-hot encoded
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ],
}



# Step 1 — Load data
def load_data(path: str) -> pd.DataFrame:
    #Load the cleaned CSV and perform a basic integrity check
    log.info("Loading data from: %s", path)
    df = pd.read_csv(path)
    log.info("Loaded shape: %s", df.shape)

    # Make sure the target column is present
    assert CONFIG["target"] in df.columns, (
        f"Target column '{CONFIG['target']}' not found. "
        "EDA.ipynb should run first?"
    )

    # Guard: no nulls should remain after EDA cleaning
    null_count = df.isnull().sum().sum()
    assert null_count == 0, (
        f"Found {null_count} null values. Re-run Stage 6 of EDA.ipynb."
    )

    log.info("Data integrity checks passed.")
    return df


# Step 2 — Feature engineering
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["new", "mid", "loyal"],
        include_lowest=True,
    ).astype(str)  # convert to str for OneHotEncoder


    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["total_services"] = (df[service_cols] == "Yes").sum(axis=1)

    log.info(
        "Engineered features added: charges_per_tenure, tenure_group, total_services"
    )
    return df



# Step 3 — Split
def split_data(df: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[CONFIG["target"]])
    y = df[CONFIG["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,  # preserves class ratio in both splits
    )

    log.info(
        "Split — train: %s  test: %s  churn_train: %.1f%%  churn_test: %.1f%%",
        X_train.shape,
        X_test.shape,
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test



# Step 4 — Build preprocessing pipeline
def build_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:

    # Update feature lists to include engineered features
    numerical = CONFIG["numerical_features"] + ["charges_per_tenure", "total_services"]
    binary = CONFIG["binary_features"]
    categorical = CONFIG["categorical_features"] + ["tenure_group"]

    # Verify all expected columns exist in the data
    missing = [
        c for c in numerical + binary + categorical
        if c not in X_train.columns
    ]
    assert not missing, f"Missing columns in X_train: {missing}"

    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        (
            "onehot",
            OneHotEncoder(
                drop="first",           # avoid multicollinearity
                handle_unknown="ignore", # robust to new categories at inference
                sparse_output=False,     # return dense array (easier to inspect)
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical + binary),
            ("cat", categorical_transformer, categorical),
        ],
        remainder="drop",  # explicitly drop any unlisted columns
    )

    log.info("Pipeline built.")
    log.info("Numerical + binary features (%d): %s", len(numerical + binary), numerical + binary)
    log.info("Categorical features (%d): %s", len(categorical), categorical)
    return preprocessor


 
# Step 5 — Fit pipeline and transform data
def fit_transform(preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame,) -> tuple[np.ndarray, np.ndarray, list[str]]:
    log.info("Fitting pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)

    log.info("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)

    # Reconstruct feature names after one-hot encoding
    num_binary_features = (
        CONFIG["numerical_features"]
        + ["charges_per_tenure", "total_services"]
        + CONFIG["binary_features"]
    )
    cat_feature_names = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CONFIG["categorical_features"] + ["tenure_group"])
        .tolist()
    )
    feature_names = num_binary_features + cat_feature_names

    log.info(
        "Processed shapes — train: %s  test: %s  features: %d",
        X_train_processed.shape,
        X_test_processed.shape,
        len(feature_names),
    )
    return X_train_processed, X_test_processed, feature_names


 
# Step 6 — Save artefacts
def save_artefacts(X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series, preprocessor: ColumnTransformer, feature_names: list[str],) -> None:
    processed_dir = Path(CONFIG["processed_dir"])
    models_dir = Path(CONFIG["models_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    artefacts = {
        processed_dir / "X_train.pkl": X_train,
        processed_dir / "X_test.pkl":  X_test,
        processed_dir / "y_train.pkl": y_train,
        processed_dir / "y_test.pkl":  y_test,
    }

    for path, obj in artefacts.items():
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        log.info("Saved: %s", path)

    pipeline_path = models_dir / "preprocessing_pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(preprocessor, f)
    log.info("Saved pipeline: %s", pipeline_path)

    feature_names_path = models_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info("Saved feature names: %s", feature_names_path)

    log.info("All artefacts saved successfully.")


 
# Orchestrator
 
def preprocess() -> None:
    log.info("=" * 55)
    log.info("  Starting preprocessing pipeline")
    log.info("=" * 55)

    df = load_data(CONFIG["clean_data_path"])
    df = add_engineered_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor  = build_pipeline(X_train)
    X_train_p, X_test_p, feature_names = fit_transform(
        preprocessor, X_train, X_test
    )
    save_artefacts(
        X_train_p, X_test_p,
        y_train, y_test,
        preprocessor, feature_names,
    )

    log.info("=" * 55)
    log.info("  Preprocessing complete.")
    log.info("=" * 55)


 
# Entry point
 
if __name__ == "__main__":
    preprocess()