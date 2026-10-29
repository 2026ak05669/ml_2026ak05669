import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)
import base64

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
st.markdown("## 🧠 Machine Learning / Classification")
st.title("ML Assignment 2 • 2026AK05669")
st.subheader("Dataset: UCI Websites to Check Wine Quality")

st.markdown("""
Compare five classifiers, explore prediction errors, and evaluate saved models on labeled test data.
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
    - Training data is used to fit five classification models.
    - Testing data is reserved for evaluation and comparison.
    - Each model is assessed using six metrics: Accuracy, Precision, Recall, F1, MCC, and AUC.
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

    # ✅ Convert continuous target to binary if needed
    unique_targets = test_df["target"].unique()
    if len(unique_targets) < 2:
        st.error("❌ Target column has only one unique value. Classification requires at least two distinct classes.")
        st.stop()
    elif len(unique_targets) > 2:
        median_val = test_df["target"].median()
        test_df["target"] = (test_df["target"] >=