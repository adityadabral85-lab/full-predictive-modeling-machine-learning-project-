from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
TARGET_COLUMN = "churn"
DROP_COLUMNS = ["customer_id"]


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def split_features_target(df):
    features = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])
    target = df[TARGET_COLUMN]
    return features, target


def build_preprocessor(features):
    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def prepare_train_test_data(test_size=0.25, random_state=42):
    df = load_dataset()
    features, target = split_features_target(df)

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
