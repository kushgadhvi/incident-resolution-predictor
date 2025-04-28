# incident-resolution-predictor  
Machine learning models to predict incident closure time and classify resolution severity based on ITSM event logs. Regression and classification pipelines with Random Forest, XGBoost, and Neural Networks.

# 🚀 Incident Management ML Project

This project explores and models data from the [UCI Incident Management Process Enriched Event Log dataset](https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log). The primary goal is to:

- **Regression**: Predict the number of hours it takes to close an incident (`time_to_close_hours`)
- **Classification**: Categorize each incident closure time into buckets like `Very Short`, `Short`, `Medium`, `Long`, and `Very Long`

---

## 📂 Project Structure

- `incident-resolution-predictor.ipynb`: Main notebook containing the full machine learning workflow.
- `incident_event_log.csv`: Can we downloaded from https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log
- `README.md`: This documentation file.

---

## 🧠 Objective

Build machine learning models to:
- Predict how long it will take to resolve an incident.
- Classify incidents into resolution time categories for triage.
---

## 🧠 Objective

Build machine learning models to:
- Predict how long it will take to resolve an incident.
- Classify incidents into resolution time categories for triage.

---

## ⚙️ Setup Instructions

Install dependencies (if running locally):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
```

> 💡 You can also run the notebook in **Google Colab**, which comes pre-installed with these libraries.

---

## 📊 Dataset Overview

- **Source**: UCI Machine Learning Repository
- **Rows**: ~141,000
- **Target Variables**:
  - `time_to_close_hours` (for regression)
  - `closure_class` (for classification)
- **Features**: Includes incident metadata such as category, urgency, timestamps, priority, assignment group, etc.

### 📌 Training Columns Used
The following columns were used for training the models:

```python
 incident_state
 active 
 reassignment_count 
 reopen_count
 sys_mod_count
 made_sla 
 location 
 category 
 impact
 urgency 
 priority 
 knowledge
 u_priority_confirmation
 closed_code
 time_to_close_hours 
 opened_at_hour
 opened_at_weekday  
 opened_at_month
 sys_updated_at_hour
 sys_updated_at_weekday
 sys_updated_at_month
```

These features include categorical, boolean, and derived time-based features.

### 📆 Date Column Transformation
To engineer temporal features, we transformed each date column using:

```python
for col in date_col_list:
    for df in [train_df, test_df]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_weekday"] = df[col].dt.weekday
        df[f"{col}_month"] = df[col].dt.month
```

This helped in extracting useful components like hour of the day, day of the week, and month of the year.

---

## 📘 Workflow Summary

### 1. 📥 Load Data
- Mounted Google Drive
- Read `incident_event_log.csv`
- Previewed dataset with `.head()`

### 2. 🔍 Exploratory Data Analysis (EDA)
- Checked distributions, missing values, and correlations
- Visualized categorical trends vs. resolution time
- Detected outliers using IQR/Z-score

### 3. 🧹 Data Cleaning
- Removed duplicates and irrelevant columns
- Filled missing values using statistical imputations
- Converted dates and created new features

### 4. 🛠 Feature Engineering
- Extracted `day of week`, `hour`, etc. from timestamps
- Applied `LabelEncoder` and `OneHotEncoder` where appropriate
- Binned the regression target into categorical classes for classification

---

## 📈 Model Training & Evaluation

### 🔍 Model Performance: Classification (Target = `closure_class`)

| Model                  | Accuracy | Macro Avg F1 | Weighted Avg F1 | Precision (Macro) | Recall (Macro) |
|------------------------|----------|--------------|------------------|--------------------|----------------|
| 🎯 Random Forest       | 0.782    | 0.68         | 0.78             | 0.77               | 0.65           |
| 📉 Logistic Regression | 0.519    | 0.43         | 0.55             | 0.46               | 0.47           |
| ⚡ XGBoost Classifier   | 0.736    | 0.68         | 0.73             | 0.78               | 0.64           |

### 🔢 Class-Wise Performance (Random Forest Example)

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Very Short   | 0.76      | 0.71   | 0.73     |
| Short        | 0.70      | 0.72   | 0.71     |
| Medium       | 0.81      | 0.89   | 0.84     |
| Long         | 0.85      | 0.74   | 0.79     |
| Very Long    | 0.74      | 0.22   | 0.34     |

#### 🔍 Classification Insights:
- **Random Forest** achieved the highest accuracy (78%) with balanced precision and recall across major classes.
- **Very Long** class had poor recall across all models — suggesting class imbalance should be addressed in future iterations.
- **XGBoost** performed competitively, slightly behind Random Forest, but better than Logistic Regression.
- **Logistic Regression** struggled due to its limitations with non-linearity and high-dimensional categorical data.

---

### 🔍 Model Performance: Regression (Target = `time_to_close_hours`)

| Model                  |   MAE   |     MSE    |   RMSE  |   R²   |
|------------------------|---------|------------|---------|--------|
| Linear Regression      | 215.96  | 235,966.51 | 485.76  | 0.48   |
| Random Forest          | 132.44  | 120,355.04 | 346.92  | 0.74   |
| Tuned Random Forest    | 131.76  | 119,516.18 | 345.71  | 0.74   |
| Gradient Boosting      | 168.95  | 162,388.16 | 402.97  | 0.64   |
| XGBoost Regressor      | 174.45  | 178,954.50 | 423.03  | 0.61   |
| Neural Network         | 189.34  | 200,430.39 | 447.69  | 0.56   |

#### 📌 Regression Insights:
- **Tuned Random Forest** achieved the best overall performance, with the lowest MAE/RMSE and the highest R² (0.74).
- **Neural Network** underperformed, likely needing more hyperparameter tuning or longer training.
- **Linear Regression** showed poor fit (R² = 0.48), highlighting the complexity and non-linearity of the target.

---

## ✅ How to Run

1. Open the notebook in **Google Colab**.
2. Mount your Google Drive with:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Update the `file_path` variable with your dataset path.
4. Run all notebook cells in order for a full end-to-end pipeline.