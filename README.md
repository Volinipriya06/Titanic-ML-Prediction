[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Volinipriya06/Titanic-ML-Prediction/blob/main/My-First-ML-Project.ipynb)

# Titanic Survival Prediction using Machine Learning

A beginner-friendly Machine Learning project built in Python using Pandas and Scikit-Learn to predict passenger survival on the Titanic. This project covers the full end-to-end data pipeline, achieving an **82.12% baseline accuracy**.

## 📊 Project Overview
This project was built to practice the foundational steps of the Machine Learning lifecycle using the classic Kaggle Titanic dataset. The objective is to build a classification model that predicts whether a passenger survived the shipwreck based on features like age, gender, ticket class, and fare.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Environment:** Google Colab
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `GridSearchCV`)

## 🔍 Technical Workflow & What I Did

### 1. Data Cleaning & Handling Missing Values
Real-world data is messy, so my first step was checking for missing values using `df.isnull().sum()`.
- **Age:** Filled missing values with the **median age** of passengers to preserve the data without skewing it.
- **Embarked:** Filled a few missing entries with the **mode** (the most frequent port: 'S').
- **Drop Noise:** Dropped columns that don't add predictive patterns like `PassengerId`, `Name`, `Ticket`, and `Cabin` (which was mostly empty).

### 2. Feature Engineering (Turning Text to Numbers)
Machine learning models only look at numbers. I used **One-Hot Encoding** (`pd.get_dummies`) to transform categorical text columns like `Sex` (male/female) and `Embarked` (C/Q/S) into binary `0` and `1` columns.

### 3. Model Training & Data Splitting
I separated the data into features ($X$) and the target variable ($y$ - `Survived`), then split them into an **80% training set** and a **20% testing set** to keep evaluation fair. 

I trained a **Random Forest Classifier**, which builds an ensemble of multiple decision trees to vote on the final outcome.

## 📈 Key Results & Process
1. **Data Cleaning:** Handled missing data in the `Age` and `Embarked` columns and dropped irrelevant features (`PassengerId`, `Name`, `Ticket`, `Cabin`).
2. **Feature Engineering:** Used One-Hot Encoding to convert categorical text features (`Sex`, `Embarked`) into numerical values.
3. **Modeling:** Trained a `RandomForestClassifier` on an 80/20 train-test split.
4. **Evaluation:** Achieved a final model accuracy of **82.12%** on the test dataset.

## 🚀 How to Run This Project

Follow these quick steps to execute the machine learning pipeline live:

1. **Open the Notebook:** Click the **Open In Colab** badge at the top of this page to launch the notebook directly in your browser.
2. **Download the Dataset:** Go to the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data) and download the `train.csv` file.
3. **Upload Data to Colab:** - On the left sidebar of your Google Colab window, click the **Folder icon**.
   - Drag and drop your downloaded `train.csv` file directly into the file storage panel.
4. **Execute the Code:** Go to the top menu bar in Colab, click on **Runtime**, and select **Run all** (or press `Ctrl + F9`). all**.
