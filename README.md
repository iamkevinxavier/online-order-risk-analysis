# Online Order Risk Analysis

## 1. Overview

This project aims to predict risk based on a provided dataset. The dataset contains various features related to customer behavior and transactions. 

**Goal:** Build a machine learning model capable of accurately classifying instances as high or low risk.

## 2. Code Description

The project utilizes Python and several libraries for data analysis, visualization, and machine learning. The code is structured into the following sections:

### 2.1. Data Loading and Preprocessing

* **Import Libraries:**
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from scipy.stats import zscore, iqr
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
  ```

* **Load Dataset:**
  ```bash
  from google.colab import files
  uploaded = files.upload()
  dataset = pd.read_csv('risk-train.txt', sep='\t')
  ```
* **Exploratory Data Analysis (EDA):**
    - Examine dataset structure using `dataset.head()`, `dataset.shape`, `dataset.info()`.
    - Identify and visualize missing values using `PrettyTable` and `sns.heatmap`.
    - Check for duplicate rows with `dataset.duplicated()`.
    - Generate descriptive statistics for numerical features using `dataset.describe()`.
* **Outlier Detection and Handling:**
    - Calculate z-scores and IQR to detect outliers.
    - Visualize potential outliers using box plots (`sns.boxplot`) and histograms (`sns.histplot`).
    - Apply logarithmic transformation (`np.log`) to specific features to mitigate outlier impact.
* **Feature Scaling:**
    - Normalize numerical features to a range of 0 to 1 using `MinMaxScaler`.
* **Missing Value Imputation:**
    - Replace missing values ('?') with '0'.
* **Date and Time Feature Engineering:**
    - Convert date columns to datetime objects using `pd.to_datetime`.
    - Extract year, month, day, and day of the week as separate features.
    - Drop original date columns after feature extraction.
    - Handle time features similarly.
    - Fill any remaining NaN values with 0.
* **Categorical Variable Encoding:**
    - Encode categorical features using `LabelEncoder`.

### 2.2. Machine Learning

* **Model Selection:**
    - Utilize Logistic Regression (`sklearn.linear_model.LogisticRegression`) for classification.
* **Hyperparameter Tuning and Cross-Validation:**
    - Employ `GridSearchCV` and `KFold` for hyperparameter optimization and model evaluation.
    - Define a parameter grid for tuning the logistic regression model.
    - Perform 5-fold cross-validation.
* **Model Training and Evaluation:**
    - Split data into features (`X_train`) and target variable (`y_train`).
    - Train the model using the best hyperparameters found through grid search.
    - Print the best parameters and the corresponding model performance (accuracy).

### 2.3. Test Dataset

* **Load and Preprocess Test Data:**
    - Load the test dataset (`risk-test.txt`).
    - Apply the same preprocessing steps as performed on the training dataset to ensure consistency.

## 3. Environment Setup

* **Google Colab:** The project is designed to run in Google Colab.
* **Dependencies:** Install required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn prettytable
```

## 4. Running the Code

1. **Open Notebook:** Open the provided notebook in Google Colab.
2. **Upload Datasets:** Upload `risk-train.txt` and `risk-test.txt` using the file upload cell.
3. **Execute Cells:** Run each code cell in sequence to perform data preprocessing, model training, and evaluation.

## 5. Results

The best hyperparameters and the achieved accuracy on the validation set are printed after model training.

## 6. Additional Notes

* Ensure that dataset files are in the correct format and contain the expected features.
* Adjust file paths and variable names if using different datasets.
* Consider further analysis and model optimization based on project requirements.
