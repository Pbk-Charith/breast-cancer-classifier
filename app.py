import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Diagnostic Classifier")
st.markdown("""
This interactive dashboard evaluates multiple classification models on the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
Upload a test CSV dataset or use the default test sample to observe performance metrics, confusion matrices, and classification reports.
""")

# Sidebar - Dataset Upload and Model Selection
st.sidebar.header("Configuration")

# 1. Dataset upload option
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (.csv)", type=["csv"])

# 2. Model selection dropdown
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl"
}

selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))

# Load data
@st.cache_data
def load_default_data():
    return pd.read_csv("test_data.csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom CSV uploaded successfully!")
else:
    data = load_default_data()
    st.sidebar.info("Using default `test_data.csv`.")

# Check for target column
if "target" not in data.columns:
    st.error("The uploaded CSV must contain a 'target' column for evaluation.")
else:
    X_test = data.drop(columns=["target"])
    y_test = data["target"]

    # Load Model
    model_path = model_options[selected_model_name]
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from `{model_path}`: {e}")
        st.stop()

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # 3. Display Evaluation Metrics in Cards
    st.subheader(f"📊 Evaluation Metrics: {selected_model_name}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    st.markdown("---")

    # 4. Visualizations: Confusion Matrix and Classification Report
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

    with col_right:
        st.subheader("Classification Report")
        report_dict = classification_report(
            y_test, y_pred, target_names=["Malignant (0)", "Benign (1)"], output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    # Data preview toggle
    with st.expander("🔍 View Test Dataset Sample"):
        st.dataframe(data.head(10))