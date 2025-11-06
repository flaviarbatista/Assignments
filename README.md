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
