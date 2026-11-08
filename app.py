from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_DATA_FILE = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

st.set_page_config(page_title="Bank Marketing Classifier Lab", page_icon="📊", layout="wide")
st.title("📊 Bank Marketing Subscription Predictor")
st.write(
    "Select a trained classifier to evaluate the Bank Marketing test dataset. "
    "The app loads saved models; it does not retrain them. You can also upload "
    "a different labelled test CSV."
)

with st.sidebar:
    st.header("Model settings")
    selected_model = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    st.caption("All five models were trained on the same UCI Bank Marketing dataset.")
    st.divider()
    st.markdown("**Developed By:** KARUNYA RAGHAVAN")
    st.markdown("**BITS ID:** 2026AK05519")

uploaded_file = st.file_uploader(
    "Optional: Upload a different test dataset (CSV)",
    type=["csv"]
)

# Use the uploaded CSV when provided; otherwise load test_data.csv from the app's project folder.
if uploaded_file is not None:
    data_source = uploaded_file
    source_label = "Uploaded test dataset"
elif DEFAULT_DATA_FILE.exists():
    data_source = DEFAULT_DATA_FILE
    source_label = "Default test_data.csv"
else:
    st.error(
        "The default test_data.csv file was not found in the project folder. "
        "Add test_data.csv beside app.py in the GitHub repository, or upload a CSV above."
    )
    st.stop()

try:
    data = pd.read_csv(data_source)
    st.caption(f"Data source: {source_label}")
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if "y" not in data.columns:
    st.error("The test CSV must include the target column named 'y'.")
    st.stop()

model_path = MODEL_DIR / MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}. Run train_models.py first.")
    st.stop()

model = joblib.load(model_path)
X_test = data.drop(columns=["y"])
y_true = data["y"].astype(str).str.strip().str.lower()

try:
    expected_features = list(model.feature_names_in_)
    missing = sorted(set(expected_features) - set(X_test.columns))
    extra = sorted(set(X_test.columns) - set(expected_features))
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()
    if extra:
        st.warning(f"Extra columns will be ignored: {extra}")
    X_test = X_test[expected_features]
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = list(model.named_steps["classifier"].classes_)
    if "yes" not in classes:
        st.error("The model does not contain the expected positive class 'yes'.")
        st.stop()
    yes_index = classes.index("yes")
     y_binary = (y_true == "yes").astype(int)
    auc = roc_auc_score(y_binary, probabilities[:, yes_index])
except Exception as exc:
