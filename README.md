# Assignments

### Assignment 8: Supervised Learning Classification (Bank Customer Churn)

This project, completed for the Data Analytics and Business Intelligence Analyst course, focuses on building a supervised classification model to predict customer churn for a bank.

## Project Goal
To implement, evaluate, and interpret two classification models (K-Nearest Neighbors and Logistic Regression) and propose a deployment strategy for the best-performing model to identify high-risk customers in real time.

## Key Technologies
- **Python** (Pandas, NumPy)
- **Scikit-learn** (KNeighborsClassifier, LogisticRegression, GridSearchCV)
- **joblib** (Model Persistence)
- **Matplotlib/Seaborn** (Visualization)

## Final Model Performance
The **K-Nearest Neighbors (KNN)** model was selected for deployment due to its superior performance on the minority class (Churn), achieving:
* **F1-Score (Churn):** 0.5205
* **ROC-AUC Score:** 0.7928

## Instructions to Set Up and Run the Notebook

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL Here]
    ```
2.  **Open in Google Colab:**
    * Upload the `Assignment_8.ipynb` file to your Google Drive.
    * Open it using Google Colaboratory.
3.  **Ensure Data is Available:**
    * The notebook assumes the `Bank Customer Churn Prediction.csv` dataset is accessible in the Colab environment or linked via Google Drive.
4.  **Run All Cells:**
    * Execute all code cells sequentially to perform data loading, preprocessing, model training, hyperparameter tuning, evaluation, and model saving.


### Assignment 10: Customer Review Sentiment Analysis

The primary goal of this project is to apply Natural Language Processing (NLP) techniques and traditional Machine Learning models to perform **Sentiment Analysis** on Amazon Alexa customer reviews. The core task is to build and optimize a Supervised Classification model to accurately classify reviews as **Positive (1)** or **Negative (0)** feedback, focusing on robust performance on the minority negative class.

## Dataset

* **Name:** Amazon Alexa Reviews
* **Source:** Kaggle / provided (`amazon_alexa.tsv`)
* **Description:** Contains 3150 customer reviews, product variations, and a binary target variable (`feedback`). The dataset is highly **imbalanced**, presenting a challenge for accurate prediction of negative sentiment.

##  Methodology & Key Techniques

### 1. Data Preprocessing
* **Missing Data:** Handled by replacing missing values in `verified_reviews` with empty strings.
* **Feature Engineering:** **One-Hot Encoding (OHE)** applied to the `variation` column.
* **Target Leakage Prevention:** Columns like `rating` were explicitly removed.
* **Data Split:** Used `train_test_split` with **stratification** (80% Train, 20% Test) to preserve class balance.

### 2. Feature Engineering (Vectorization)
* **Bag of Words (BoW):** Implemented using `CountVectorizer` to convert text into numerical feature vectors.
* **TF-IDF:** Implemented using `TfidfVectorizer` to weight word importance.
* **Final Feature Set:** Concatenation of BoW features and OHE product features.

### 3. Model Training & Optimization
* **Models:** Logistic Regression (Baseline) and Random Forest Classifier.
* **Optimization:** **Grid Search Cross-Validation (`GridSearchCV`)** was performed on the Random Forest model to tune hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).
* **Optimization Metric:** Weighted F1-Score.
* **Best Parameters Found:** `{'max_depth': 100, 'n_estimators': 200, 'min_samples_split': 2}`

### 4. Sentiment Analysis with LLM (Conceptual)
* The **DistilBERT** model from Hugging Face Transformers was conceptually implemented to demonstrate the superior contextual understanding of **Large Language Models** over traditional methods, especially in handling nuanced or implicit negative feedback.

## Model Evaluation Summary

Evaluation focused on the **F1-Score for the Negative Class (Class 0)**, as this metric is critical for identifying urgent customer issues in an imbalanced dataset.

| Model | Accuracy | F1-Score (Class 0) | Precision (Class 0) | Recall (Class 0) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | $0.93$ | $0.32$ | $0.65$ | $0.22$ |
| **Random Forest (Optimized)** | **$0.93$** | **$0.38$** | **$0.76$** | **$0.25$** |

* **Conclusion:** The **Random Forest Classifier** was the best-performing model for the minority class, achieving an F1-Score of $0.38$, demonstrating a better ability to distinguish negative sentiment patterns compared to Logistic Regression.

## Deployment Strategy (Hypothetical)

* **Deployment:** The optimized Random Forest model would be saved (e.g., using `pickle`) and deployed as a **REST API service** (e.g., using Flask) on a cloud platform to provide real-time sentiment prediction for new reviews.
* **Monitoring:** Continuous monitoring is essential, specifically tracking the **F1-Score for the Negative Class** on new, labeled data. An automated pipeline should be in place to **retrain the model** if performance drops below a predefined threshold (Model Drift).

---
