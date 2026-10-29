# ML Assignment 2 - Classification Models & Streamlit Deployment

## Problem Statement
Build and deploy multiple classification models on a dataset with ≥12 features and ≥500 instances. Evaluate performance using key metrics and deploy an interactive Streamlit app.

## Dataset Description
Chosen dataset: [Heart Disease Dataset from UCI Repository]  
- Instances: 1025  
- Features: 13 (age, sex, chest pain type, etc.)  
- Target: Presence/absence of heart disease (binary classification)

## GitHub Repository Link
[Provide your repo link here]

## Models Used
Implemented models:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier  
5. Support Vector Machine  

### Comparison Table

| Model                | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression  | 0.84     | 0.87  | 0.83      | 0.84   | 0.83  | 0.68  |
| Decision Tree        | 0.79     | 0.80  | 0.78      | 0.79   | 0.78  | 0.59  |
| KNN                  | 0.82     | 0.85  | 0.81      | 0.82   | 0.81  | 0.64  |
| Naive Bayes          | 0.80     | 0.83  | 0.79      | 0.80   | 0.79  | 0.61  |
| SVM                  | 0.85     | 0.88  | 0.84      | 0.85   | 0.84  | 0.70  |

*(Values are illustrative — replace with your actual results.)*

### Observations

| Model                | Observation |
|----------------------|-------------|
| Logistic Regression  | Performs well with balanced precision and recall. |
| Decision Tree        | Slightly overfits; lower MCC. |
| KNN                  | Good performance but sensitive to scaling. |
| Naive Bayes          | Fast, decent accuracy, but weaker MCC. |
| SVM                  | Best overall performance; robust generalization. |
| **Winner**           | Support Vector Machine |

---

## Deployment
- Streamlit app deployed at: [Your Streamlit link]  
- Features: dataset upload, model selection dropdown, metrics display, confusion matrix.

