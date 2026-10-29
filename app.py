import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

st.title("Classification Model Demo")

# --- Load dataset automatically ---
try:
    df = pd.read_csv("test_data.csv")
    st.success("✅ test_data.csv loaded successfully!")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("❌ test_data.csv not found in the project folder.")
    st.stop()

# --- Check for target column ---
if "target" not in df.columns:
    st.error("❌ The dataset must contain a 'target' column for evaluation.")
    st.stop()

# --- Clean and preprocess ---
# Encode categorical columns (features)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# --- Split features and labels ---
X = df.drop("target", axis=1)
y = df["target"]

# --- Sidebar controls ---
st.sidebar.header("Hyperparameter Tuning")
knn_k = st.sidebar.slider("K in KNN", 1, 20, 5)
tree_depth = st.sidebar.slider("Max Depth in Decision Tree", 1, 20, 5)
svm_c = st.sidebar.slider("SVM Regularization (C)", 0.1, 10.0, 1.0)
log_reg_c = st.sidebar.slider("Logistic Regression (C)", 0.01, 10.0, 1.0)
scale_features = st.sidebar.checkbox("Apply Feature Scaling (StandardScaler)", value=True)
split_ratio = st.sidebar.slider("Train/Test Split Ratio (Test Size %)", 10, 50, 20)

selected_metrics = st.sidebar.multiselect(
    "Choose metrics for summary table",
    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
    default=["Accuracy", "F1", "MCC"]
)

selected_models = st.sidebar.multiselect(
    "Choose models for comparison",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"],
    default=["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"]
)

# --- Train/test split ---
test_size = split_ratio / 100.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if scale_features:
    scaler = StandardScaler()
    X_train