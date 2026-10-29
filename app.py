import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Classification Model Demo")

# Automatically load test_data.csv
try:
    test_data = pd.read_csv("test_data.csv")
    st.success("✅ test_data.csv loaded successfully!")
    st.dataframe(test_data.head())
except FileNotFoundError:
    st.error("❌ test_data.csv not found in the project folder.")
    st.stop()

# Sidebar hyperparameter controls
st.sidebar.header("Hyperparameter Tuning")
knn_k = st.sidebar.slider("K in KNN", 1, 20, 5)
tree_depth = st.sidebar.slider("Max Depth in Decision Tree", 1, 20, 5)
svm_c = st.sidebar.slider("SVM Regularization (C)", 0.1, 10.0, 1.0)
log_reg_c = st.sidebar.slider("Logistic Regression (C)", 0.01, 10.0, 1.0)

# Sidebar scaling toggle
scale_features = st.sidebar.checkbox("Apply Feature Scaling (StandardScaler)", value=True)

# Sidebar train/test split ratio
split_ratio = st.sidebar.slider("Train/Test Split Ratio (Test Size %)", 10, 50, 20)

# Sidebar metric selection
st.sidebar.header("Select Metrics to Display")
selected_metrics = st.sidebar.multiselect(
    "Choose metrics for summary table",
    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    default=["Accuracy", "F1", "MCC"]
)

# Sidebar model selection
st.sidebar.header("Select Models to Compare")
selected_models = st.sidebar.multiselect(
    "Choose models for comparison",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"],
    default=["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"]
)

# --- Core evaluation logic ---
def run_models(df):
    if "target" not in df.columns:
        st.error("❌ The dataset must contain a 'target' column for evaluation.")
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    test_size = split_ratio / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    models_dict = {
        "Logistic Regression": LogisticRegression(C=log_reg_c, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=tree_depth),
        "KNN": KNeighborsClassifier(n_neighbors=knn_k),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(C=svm_c, probability=True)
    }

    results = []
    for name in selected_models:
        model = models_dict[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1": f1_score(y_test, y_pred, average="weighted"),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)

    summary_df = pd.DataFrame(results)[["Model"] + selected_metrics]
    return summary_df

# Run models and display results
st.subheader("Model Evaluation Results")
summary_df = run_models(test_data)
st.dataframe(summary_df)

# Visualization
st.subheader("Metric Comparison Chart")
for metric in selected_metrics:
    fig, ax = plt.subplots()
    sns.barplot(data=summary_df, x="Model", y=metric, ax=ax)
    ax.set_title(f"{metric} Comparison Across Models")
    st.pyplot(fig)
