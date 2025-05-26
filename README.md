# MLforASDprediction
#  ASD Prediction – Early Detection Using Machine Learning

This project focuses on predicting early signs of Autism Spectrum Disorder (ASD) using a machine learning framework built with behavioral, physiological, and genetic data. The goal is to aid early diagnosis and intervention using data-driven techniques.

##  Project Overview

Autism Spectrum Disorder (ASD) is a developmental condition affecting communication, behavior, and social interaction. Early detection is crucial for improved long-term outcomes. This project applies various ML models to predict ASD in individuals based on key input features.

###  Objectives
- Develop a predictive model to detect early signs of ASD.
- Evaluate multiple ML algorithms and select the best-performing one.
- Support the early intervention process by providing accurate prediction results.
- Create a reproducible and clean pipeline for similar health-related applications.

---

##  Dataset

The dataset includes features such as:
- Behavioral patterns (communication score, social skill rating)
- Physiological metrics (eye contact frequency, response delay)
- Genetic factors (family history of ASD, related conditions)

> Note: Ensure ethical use of health data and compliance with any licensing agreements.

---

##  Technologies Used

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib / Seaborn** (for visualization)
- **AdaBoost** (as main classifier)

---

##  Model Pipeline

1. **Data Cleaning and Preprocessing**
   - Null value handling
   - Encoding categorical features
   - Normalization of inputs

2. **Exploratory Data Analysis (EDA)**
   - Visualizations of class distribution
   - Correlation analysis

3. **Feature Selection**
   - Feature importance using tree-based methods

4. **Model Training**
   - Base models: Decision Tree, Logistic Regression, Random Forest
   - Ensemble model: AdaBoost (best performer)

5. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

---

##  Results

| Model            | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| Logistic Regression | 85%   | 84%       | 83%    | 83.5%    |
| Decision Tree       | 87%   | 86%       | 85%    | 85.5%    |
| **AdaBoost**        | **92%** | **91%**   | **90%** | **90.5%** |

> AdaBoost provided the best performance and was selected as the final model.

---

##  Future Work

- Integrate deep learning methods for more nuanced pattern detection.
- Explore additional datasets to generalize the model further.
- Build a front-end interface for doctors and specialists to upload data and receive instant predictions.

---

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mulukuntlarohan/MLforASDprediction.git
   cd MLforASDprediction
