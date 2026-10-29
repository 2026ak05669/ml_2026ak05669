import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

    # Working “Open CSV in new tab” link using base64 encoding
    with open("test_data.csv", "rb") as f:
        csv_bytes = f.read()
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:file/csv;base64,{b64}" target="_blank">Open CSV in a new tab</a>'
        st.markdown("[Open CSV in a new tab](test_data.csv)", unsafe_allow_html=True)

    st.info("If an embedded preview blocks saving, open the local app in your regular browser. The same sample is also in the project folder.")

    st.markdown("### Upload test CSV")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])
    st.caption("Limit 5MB per file • CSV")

# --- Main content ---
st.markdown("## 🧠 Machine Learning / Classification")
st.title("ML Assignment 2 • 2026AK05669")
st.subheader("Dataset: UCI Wine Quality")

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

    unique_targets = test_df["target"].unique()
    if len(unique_targets) < 2:
        st.warning("⚠️ Target column has only one unique value. Showing descriptive statistics instead of classification results.")
        st.write(test_df.describe())
        st.stop()

    X = test_df.drop("target", axis=1)
    y = test_df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train-test split for realistic evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    # --- Define models ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True)
    }

    results = []
    reports = {}

    # --- Evaluate each model ---
    for name, model in models.items():
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

        reports[name] = {
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred, output_dict=True),
            "Metrics": metrics
        }

    # --- Comparison table ---
    st.markdown("### 📊 Model Comparison Across Metrics")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # --- Visual comparison charts ---
    st.markdown("### 📈 Visual Comparison Charts")
    for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        fig, ax = plt.subplots()
        sns.barplot(data=results_df, x="Model", y=metric, palette="coolwarm", ax=ax)
        ax.set_title(f"{metric} Comparison Across Models")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        st.pyplot(fig)

    # --- Closer Look Section ---
    st.markdown("## 🔍 02 / Look Closer")
    selected_model = st.selectbox("Select a model to inspect", list(models.keys()))

    model_metrics = reports[selected_model]["Metrics"]
    st.markdown(f"### {selected_model}: evaluation metrics")
    st.caption("Precision, recall and F1 use class 1 as the positive class. AUC uses continuous scores.")

    # Display metric cards
    metric_cols = st.columns(3)
    metric_cols[0].metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
    metric_cols[1].metric("AUC", f"{model_metrics['AUC']:.4f}")
    metric_cols[2].metric("Precision", f"{model_metrics['Precision']:.4f}")

    metric_cols2 = st.columns(3)
    metric_cols2[0].metric("Recall", f"{model_metrics['Recall']:.4f}")
    metric_cols2[1].metric("F1", f"{model_metrics['F1']:.4f}")
    metric_cols2[2].metric("MCC", f"{model_metrics['MCC']:.4f}")

    # --- Confusion Matrix ---
    st.markdown("### 🔢 Confusion Matrix")
    cm = reports[selected_model]["Confusion Matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --- Classification Report ---
    st.markdown("### 🧾 Classification Report")
    st.dataframe(pd.DataFrame(reports[selected_model]["Classification Report"]).transpose())

    st.info("✅ Evaluation complete! Comparison, detailed metrics, confusion matrix, and classification report displayed.")
else:
    st.warning("Please upload your test_data.csv file to begin evaluation.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
