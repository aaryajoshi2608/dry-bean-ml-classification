
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification Dashboard")
st.write("Compare five machine learning models trained on the Dry Bean Dataset.")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_models():
    folder = Path("model")
    models = {
        name: joblib.load(folder / filename)
        for name, filename in model_files.items()
    }
    scaler = joblib.load(folder / "scaler.joblib")
    features = joblib.load(folder / "feature_names.joblib")
    return models, scaler, features

models, scaler, feature_names = load_models()

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload Dry Bean Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    required_columns = feature_names + ["Class"]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    X = data[feature_names]
    y = data["Class"]

    if X.isnull().any().any() or y.isnull().any():
        st.error("Uploaded dataset contains missing values.")
        st.stop()

    model = models[selected_model]

    if selected_model == "Decision Tree":
        X_processed = X
    else:
        X_processed = scaler.transform(X)

    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average="weighted", zero_division=0)
    recall = recall_score(y, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y, predictions, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y, predictions)

    try:
        auc = roc_auc_score(
            y, probabilities,
            multi_class="ovr",
            average="weighted",
            labels=model.classes_
        )
    except ValueError:
        auc = None

    st.subheader(f"Evaluation Results: {selected_model}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion Matrix")

    labels = model.classes_
    cm = confusion_matrix(y, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Prediction Results")
    results = data.copy()
    results["Predicted_Class"] = predictions
    st.dataframe(results.head(20))

    st.download_button(
        "Download Predictions",
        results.to_csv(index=False),
        file_name="bean_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload test_data.csv to view model performance.")
