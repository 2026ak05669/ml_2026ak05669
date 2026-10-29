import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
import base64
import numpy as np

# --- Page setup ---
st.set_page_config(page_title="ML Assignment 2 - 2026AK05669", layout="wide")

# --- Sidebar: Dataset and Upload ---
with st.sidebar:
    st.header("ML Assignment 2")
    st.caption("2026AK05669 • Classifier Comparison")

    st.markdown("### 01 / Evaluation Data")
    st.subheader("Test data")

    st.markdown("Download a matching test sample or upload labeled test data for evaluation.")
    st.download_button("⬇️ Download test_data.csv", data=open("test_data.csv", "rb"), file_name="test_data.csv")

    # ✅ Working “Open CSV in new tab” link using base64 encoding
    with open("test_data.csv", "rb") as f:
        csv_bytes = f.read()
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:file/csv;base64,{b64}" target="_blank">Open CSV in a new tab</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.info("If an embedded preview blocks saving, open the local app in your regular browser. The same sample is also in the project folder.")

    st.markdown("### Upload test CSV")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])
    st.caption("Limit 5MB per file • CSV")

# --- Main content ---
st.markdown("## 🧠 Machine Learning / Classification & Regression")
st.title("ML Assignment 2 • 2026AK05669")
st.subheader("Dataset: UCI Websites to Check Wine Quality")

st.markdown("""
Compare five models, explore prediction errors, and evaluate saved models on labeled test data.
""")

# --- Dataset summary cards ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Full source dataset", "6,497 rows")
with col2:
    st.metric("Used for training", "1,300 rows")
with col3:
    st.metric("Reserved for testing", "1,300 rows")

st.markdown("""
The models learned from the training rows. The downloadable sample contains the reserved test rows.
Upload a labeled test CSV to evaluate the saved models.
""")

# --- Expandable section ---
with st.expander("📘 How the dataset is used"):
    st.write("""
    - The dataset is split into training and testing subsets.
    - Training data is used to fit five models.
    - Testing data is reserved for evaluation and comparison.
    - Each model is assessed using appropriate metrics for classification or regression.
    """)

# --- Evaluation section ---
st.markdown("### 🧪 Ready to Evaluate")
st.write("Upload your test data to begin.")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("✅ Test data uploaded successfully!")
    st.write("### Preview of Uploaded Data")
    st.dataframe(test_df.head())

    # --- Preprocessing ---
    if "target" not in test_df.columns:
        st.error("❌ The uploaded CSV must contain a 'target' column for evaluation.")
        st.stop()

    for col in test_df.columns:
        if test_df[col].dtype == "object":
            test_df[col] = LabelEncoder().fit_transform(test_df[col].astype(str))

    test_df = test_df.replace([float("inf"), float("-inf")], pd.NA)
    test_df = test_df.fillna(test_df.mean(numeric_only=True))
    test_df["target"] = pd.to_numeric(test_df["target"], errors="coerce").fillna(0)

    X = test_df.drop("target", axis=1)
    y = test_df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Detect classification vs regression ---
    unique_targets = np.unique(y)
    if len(unique_targets) < 2:
        st.warning("⚠️ Target column has only one unique value. Showing descriptive statistics instead of model results.")
        st.write(test_df.describe())
        st.stop()
    elif len(unique_targets) <= 10 and all(y.astype(int) == y):
        mode = "classification"
        st.info("🔍 Detected Classification Task")
    else:
        mode = "regression"
        st.info("📈 Detected Regression Task")

    results = []
    reports = {}

    # --- Define models based on mode ---
    if mode == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True)
        }

        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

                metrics = {
                    "Model": name,
                    "Accuracy": accuracy_score(y, y_pred),
                    "Precision": precision_score(y, y_pred, average="weighted"),
                    "Recall": recall_score(y, y_pred, average="weighted"),
                    "F1 Score": f1_score(y, y_pred, average="weighted"),
                    "MCC": matthews_corrcoef(y, y_pred),
                    "AUC": roc_auc_score(y, y_prob) if y_prob is not None else "N/A"
                }
                results.append(metrics)

                reports[name] = {
                    "Confusion Matrix": confusion_matrix(y, y_pred),
                    "Classification Report": classification_report(y, y_pred, output_dict=True)
                }
            except ValueError as e:
                st.error(f"⚠️ {name} could not be trained: {e}")

    else:  # Regression mode
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
            "SVM Regressor": SVR(),
            "Naive Bayes (Gaussian)": GaussianNB()  # fallback for numeric data
        }

        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                metrics = {
                    "Model": name,
                    "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
                    "R² Score": r2_score(y, y_pred)
                }
                results.append(metrics)
            except ValueError as e:
                st.error(f"⚠️ {name} could not be trained: {e}")

    # --- Display results ---
    st.markdown("### 📊 Evaluation Metrics Across Models")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # --- Download results ---
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Evaluation Results (CSV)",
        data=csv_data,
        file_name="evaluation_results.csv",
        mime="text/csv"
    )

    # --- Visual comparison charts ---
    st.markdown("### 📈 Visual Comparison Charts")
    for metric in results_df.columns[1:]:
        fig, ax = plt.subplots()
        sns.barplot(data=results_df, x="Model", y=metric, palette="coolwarm", ax=ax)
        ax.set_title(f"{metric} Comparison Across Models")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        st.pyplot(fig)

    # --- Classification reports (only for classification mode) ---
    if mode == "classification":
        st.markdown("### 🧾 Classification Reports")
        for name, report in reports.items():
            st.write(f"**{name}**")
            st.dataframe(pd.DataFrame(report["Classification Report"]).transpose())

    st.info("✅ Evaluation complete! Metrics, charts, and reports displayed for all models.")
else:
    st.warning("Please upload your test_data.csv file to begin evaluation.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
