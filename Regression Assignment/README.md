# Regression Assignment – Insurance Charges Prediction

## Project Overview

This project is a Machine Learning regression assignment developed to predict medical insurance charges using personal and lifestyle information.

The objective is to develop multiple regression models, compare their performance using the **R² Score**, and select the best-performing model as the final model.

## Problem Statement

Develop a Machine Learning model that predicts insurance charges based on:

- Age
- BMI
- Number of Children
- Sex
- Smoking Status

This is a **Supervised Learning – Regression** problem because `charges` is a continuous numerical target variable.

## Dataset

The dataset contains **1,338 rows and 6 columns**.

| Column | Description |
|---|---|
| `age` | Age of the individual |
| `sex` | Gender of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependents |
| `smoker` | Smoking status |
| `charges` | Medical insurance charges |

### Target Variable

`charges`

### Input Features

`age`, `bmi`, `children`, `sex`, `smoker`

## Data Pre-processing

The categorical variables `sex` and `smoker` were converted into numerical values using One-Hot Encoding with `drop_first=True`.

```python
dataset = pd.get_dummies(dataset, drop_first=True)
dataset = dataset.astype(int)
```

After preprocessing:

| Original Value | Encoded Value |
|---|---:|
| Female | 0 |
| Male | 1 |
| No | 0 |
| Yes | 1 |

The final model input features were:

```text
age
bmi
children
sex_male
smoker_yes
```

## Train-Test Split

The dataset was split into training and testing data using:

```python
train_test_split(
    independent,
    dependent,
    test_size=0.30,
    random_state=0
)
```

- Training Data: **70%**
- Testing Data: **30%**
- Random State: **0**

## Machine Learning Models

Four regression approaches were evaluated.

### 1. Multiple Linear Regression

Best R² Score:

**0.7893**

### 2. Support Vector Regression (SVR)

The `kernel` and `C` parameters were tested.

Best configuration:

- Kernel: `linear`
- C: `3000`

Best R² Score:

**0.7590**

### 3. Decision Tree Regression

The `criterion`, `splitter`, and `max_features` parameters were tested.

Best configuration:

- Criterion: `squared_error`
- Splitter: `best`
- Max Features: `sqrt`

Best R² Score:

**0.7839**

### 4. Random Forest Regression

The `criterion`, `max_features`, and `n_estimators` parameters were tested.

Best configuration:

- Criterion: `absolute_error`
- Max Features: `sqrt`
- N Estimators: `100`

Best R² Score:

**0.8667**

## Model Comparison

| Model | Best Configuration | R² Score |
|---|---|---:|
| Multiple Linear Regression | Default | **0.7893** |
| Support Vector Regression | Linear, C=3000 | **0.7590** |
| Decision Tree Regression | squared_error, best, sqrt | **0.7839** |
| Random Forest Regression | absolute_error, sqrt, 100 | **0.8667** |

## Final Model

The **Random Forest Regressor** was selected as the final model because it achieved the highest R² Score among the tested models.

### Final Parameters

```python
RandomForestRegressor(
    criterion='absolute_error',
    max_features='sqrt',
    n_estimators=100,
    random_state=0
)
```

### Final R² Score

**0.8667**

## Project Structure

```text
Regression Assignment/
│
├── README.md
├── insurance_pre.csv
├── Regression_Assignment_Insurance_Charges.pdf
│
├── Models/
│   ├── 1. Multiple Linear Regression.ipynb
│   ├── 2. SVM_Regression_without_standardization.ipynb
│   ├── 3. Decision_Tree.ipynb
│   └── 4. Random_Forest_regression.ipynb
│
├── Automation/
│   ├── Automation_SVM.ipynb
│   ├── Automation_Decision_Tree.ipynb
│   └── Automation_Random_Forest_regression.ipynb
│
└── Final Model/
    ├── README.md
    ├── Random_Forest_regression_Phase-1_Model_Creation.ipynb
    ├── Random_Forest_regression_Phase-2_Deployment.ipynb
    └── final_Model_Insurance_Charges_Predict.sav
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- Pickle
- GitHub

## Evaluation Metric

The primary evaluation metric used in this project is the **R² Score**.

A higher R² Score indicates that the model explains a larger proportion of the variation in the target variable.

## Conclusion

Multiple regression algorithms were developed and evaluated for predicting medical insurance charges.

The final comparison showed that **Random Forest Regression achieved the highest R² Score of 0.8667**.

Therefore, the Random Forest Regressor with `criterion='absolute_error'`, `max_features='sqrt'`, and `n_estimators=100` was selected as the final model.

## Author

**Sanjay R**

Machine Learning – Regression Assignment
