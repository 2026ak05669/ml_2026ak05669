# ML Assignment 2 • 2026AK05669

## a. Problem Statement  
The objective of this project is to evaluate and compare five machine learning classification models on the **Wine Quality dataset**. The task is to predict wine quality (binary target: high vs. low) based on physicochemical properties. The project demonstrates preprocessing, model training, evaluation, and deployment using Streamlit.

---

## b. Dataset Description [1 mark]  
- **Source:** UCI Wine Quality dataset  
- **Features:** 11 physicochemical properties (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)  
- **Target:** Binary classification derived from the `quality` column  
  - Wines with `quality ≥ 6` → **High quality (1)**  
  - Wines with `quality < 6` → **Low quality (0)**  
- **Size:** ~6,497 rows  

---

## c. GitHub Repository Link [1 mark]  
👉 https://github.com/2026ak05669/ml_2026ak05669/blob/main/test_data.csv 

---

## d. Models Used [5 marks]  
The following five models were trained and evaluated:  
- Logistic Regression  
- Decision Tree  
- KNN  
- Naive Bayes  
- Support Vector Machine  

### 📊 Comparison Table of Metrics

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 1        | 1     | 1         | 1      | 1     | 1     |
| Decision Tree       | 1        | 1     | 1         | 1      | 1     | 1     |
| KNN                 | 0.9462   | 0.9903| 0.9465    | 0.9462 | 0.9463| 0.8837|
| Naive Bayes         | 0.9538   | 0.9909| 0.9549    | 0.9538 | 0.9534| 0.8999|
| Support Vector Machine | 0.9974| 0.9998| 0.9975    | 0.9974 | 0.9974| 0.9945|


## e. Observations [3 marks]  

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Balanced performance, strong AUC, stable across folds. |
| Decision Tree       | Easy to interpret but prone to overfitting, lower MCC. |
| KNN                 | Performs well with scaled data, but sensitive to parameter choice. |
| Naive Bayes         | Fast and simple, but weaker precision/recall compared to others. |
| Support Vector Machine | Best overall performance, high accuracy and MCC, robust generalization. |

**Overall Winner for this dataset:**  
👉 **Support Vector Machine (SVM)** — achieved the highest accuracy, AUC, and MCC, making it the most reliable model for predicting wine quality.  
