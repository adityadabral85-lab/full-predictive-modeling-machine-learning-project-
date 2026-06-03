from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data_loader import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def train_models(x_train, y_train):
    preprocessor = build_preprocessor(x_train)
    trained_models = {}

    for model_name, estimator in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        trained_models[model_name] = pipeline

    return trained_models


def save_best_model(model, path=BEST_MODEL_PATH):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
