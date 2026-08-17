# Breast Cancer Diagnostic Classification

## a. Problem Statement
The goal of this project is to develop and deploy an end-to-end Machine Learning classification system to predict whether a breast mass biopsy is **Malignant (0)** or **Benign (1)** based on digitized image characteristics of fine needle aspirates (FNA).

## b. Dataset Description
* **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / Kaggle)
* **Number of Instances**: 569 (Meets minimum 500 constraint)
* **Number of Features**: 30 numeric predictive attributes (Meets minimum 12 constraint)
* **Target Attribute**: Diagnosis (0 = Malignant, 1 = Benign)

## c. GitHub Repository Link
* **GitHub URL**: `https://github.com/Pbk-Charith/breast-cancer-classifier`
* **Live App URL**: `https://pbk-charith-breast-cancer-classifier.streamlit.app/`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9825 | 0.9974 | 0.9861 | 0.9861 | 0.9861 | 0.9620 |
| Decision Tree | 0.9123 | 0.9248 | 0.9429 | 0.9167 | 0.9296 | 0.8143 |
| kNN | 0.9649 | 0.9891 | 0.9722 | 0.9722 | 0.9722 | 0.9246 |
| Naive Bayes | 0.9386 | 0.9881 | 0.9452 | 0.9583 | 0.9517 | 0.8679 |
| Random Forest (Ensemble) | 0.9737 | 0.9954 | 0.9726 | 0.9861 | 0.9793 | 0.9439 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well with standard scaling, achieving the highest overall MCC and F1 score due to strong linear separability among normalized FNA features. |
| **Decision Tree** | Produced a baseline non-linear structure but exhibited slight overfitting on the training set, leading to lower recall and MCC compared to ensemble methods. |
| **kNN** | Delivered strong performance when coupled with standard scaling, capturing local metric clusters accurately with $k=5$. |
| **Naive Bayes** | Maintained high AUC but slightly lagged in precision due to slight feature correlation between radius, area, and perimeter attributes. |
| **Random Forest (Ensemble)** | Highly robust and generalized well across all 30 features, yielding near-perfect AUC and strong balance between precision and recall. |

**Overall Winner for your dataset**: **Logistic Regression** (with feature scaling) and **Random Forest (Ensemble)** emerged as top performers, with Logistic Regression achieving the peak MCC score ($0.9620$) and AUC ($0.9974$).
