"""
Train five classifiers on the UCI Bank Marketing dataset.
Run: python train_models.py
Downloads bank-full.csv, creates test_data.csv, saves models and metrics.csv.
"""
from pathlib import Path
import io
import zipfile
import urllib.request

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_FILE = ROOT / "bank-full.csv"


def get_dataset():
    """Download and extract the full dataset if it is not already present."""

    if not DATA_FILE.exists():
        print("Downloading UCI Bank Marketing dataset...")

        request = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()

        # Open the outer ZIP
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:

            print("Outer ZIP files:", archive.namelist())

            # Extract the nested bank.zip
            nested_zip = archive.read("bank.zip")

            with zipfile.ZipFile(io.BytesIO(nested_zip)) as bank_archive:

                print("Bank ZIP files:", bank_archive.namelist())

                matches = [
                    name for name in bank_archive.namelist()
                    if name.replace("\\", "/").split("/")[-1].lower()
                    == "bank-full.csv"
                ]

                if not matches:
                    raise FileNotFoundError(
                        "bank-full.csv not found inside bank.zip"
                    )

                DATA_FILE.write_bytes(
                    bank_archive.read(matches[0])
                )

                print("Dataset extracted successfully!")

    # Read the dataset
    return pd.read_csv(DATA_FILE, sep=";")


def main():
    df = get_dataset()
    target = "y"
    X = df.drop(columns=[target])
    y = df[target].str.strip().str.lower()

    # This dataset has 7 numeric and 9 categorical predictors (16 total).
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # Dense output works for all five models, including Gaussian Naive Bayes.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_features),
        ],
        remainder="drop"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []
    for display_name, estimator in models.items():
        print(f"\nTraining {display_name}...")
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", estimator)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        yes_index = list(pipeline.named_steps["classifier"].classes_).index("yes")
        auc = roc_auc_score((y_test == "yes").astype(int), probabilities[:, yes_index])

        row = {
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": auc,
            "Precision": precision_score(y_test, predictions, pos_label="yes", zero_division=0),
            "Recall": recall_score(y_test, predictions, pos_label="yes", zero_division=0),
            "F1": f1_score(y_test, predictions, pos_label="yes", zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions),
        }
        results.append(row)
        filename = display_name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(pipeline, MODEL_DIR / filename)
        print(row)

    # The assignment asks for test data with both predictors and target.
    test_output = X_test.copy()
    test_output[target] = y_test
    test_output.to_csv(ROOT / "test_data.csv", index=False)

    metrics = pd.DataFrame(results)
    metrics.to_csv(ROOT / "metrics.csv", index=False)
    print("\nSaved five models in model/, test_data.csv and metrics.csv.")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
