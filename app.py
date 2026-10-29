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

st.set_page_config(page_title="ML Assignment 2 - 2026AK05669", layout="wide")

with st.sidebar:
    st.header("ML Assignment 2")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

st.title("ML Assignment 2 • 2026AK05669")
st.subheader("Wine Quality Classification")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Test data uploaded successfully!")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("❌ The uploaded CSV must contain a 'target' column.")
        st.stop()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.mean(numeric_only=True))
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0)

    if len(df["target"].unique()) < 2:
        st.warning("⚠️ Target column has only one unique value. Showing descriptive statistics instead.")
        st.write(df.describe())
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True)
    }

    results, reports = [], {}

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

    st.markdown("### 📊 Model Comparison Across Metrics")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    st.markdown("### 📈 Visual Comparison Charts")

    # 🔹 Display charts in a grid (2 per row)
    metrics_list = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for i in range(0, len(metrics_list), 2):
        cols = st.columns(2)
        for j, metric in enumerate(metrics_list[i:i+2]):
            with cols[j]:
                fig, ax = plt.subplots(figsize=(4, 3))  # smaller chart size
                sns.barplot(data=results_df, x="Model", y=metric, palette="coolwarm", ax=ax)
                ax.set_title(f"{metric} Comparison")
                st.pyplot(fig)

    st.markdown("## 🔍 Look Closer")
    selected_model = st.selectbox("Select a model", list(models.keys()))

    m = reports[selected_model]["Metrics"]
    cols = st.columns(3)
    cols[0].metric("Accuracy", f"{m['Accuracy']:.3f}")
    cols[1].metric("AUC", f"{m['AUC']:.3f}")
    cols[2].metric("Precision", f"{m['Precision']:.3f}")
    cols2 = st.columns(3)
    cols2[0].metric("Recall", f"{m['Recall']:.3f}")
    cols2[1].metric("F1", f"{m['F1']:.3f}")
    cols2[2].metric("MCC", f"{m['MCC']:.3f}")

    st.markdown("### 🔢 Confusion Matrix")
    cm = reports[selected_model]["Confusion Matrix"]
    fig, ax = plt.subplots(figsize=(3.5, 3))  # smaller matrix size
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("### 🧾 Classification Report")
    st.dataframe(pd.DataFrame(reports[selected_model]["Classification Report"]).transpose())
