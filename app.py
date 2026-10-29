import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page setup ---
st.set_page_config(page_title="Classification Model Demo", layout="centered")
st.title("🧠 Classification Model Demo")

st.markdown("""
This app demonstrates a simple classification workflow — it automatically loads your dataset,
trains models, and displays predictions and performance metrics.
""")

# --- Load dataset automatically ---
try:
    df = pd.read_csv("test_data.csv")
    st.success("✅ test_data.csv loaded successfully!")
except FileNotFoundError:
    st.error("❌ test_data.csv not found in the project folder.")
    st.stop()

# --- Preprocessing ---
if "target" not in df.columns:
    st.error("❌ The dataset must contain a 'target' column for evaluation.")
    st.stop()

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

X = df.drop("target", axis=1)
y = df["target"]

# --- Sidebar controls ---
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"])
scale_features = st.sidebar.checkbox("Apply Feature Scaling", value=True)

# --- Train/test split ---
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if scale_features:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# --- Model selection ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True)
}

model = models[model_choice]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

# --- Display results ---
st.subheader("📊 Model Evaluation Results")
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="weighted"),
    "Recall": recall_score(y_test, y_pred, average="weighted"),
    "F1 Score": f1_score(y_test, y_pred, average="weighted"),
    "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
}

st.write(pd.DataFrame([metrics]))

# --- Visualization ---
st.subheader("📈 Prediction Distribution")
fig, ax = plt.subplots()
sns.countplot(x=y_pred, palette="coolwarm", ax=ax)
ax.set_title(f"Predicted Class Distribution ({model_choice})")
st.pyplot(fig)

# --- Save model (optional) ---
joblib.dump(model, f"model/{model_choice.replace(' ', '_')}.joblib")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
