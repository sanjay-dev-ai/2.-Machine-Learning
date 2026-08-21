# Regression Assignment – Insurance Charges Prediction

## Project Overview

This project is a Machine Learning regression project developed to predict medical insurance charges based on an individual's personal and lifestyle information.

The objective is to build and compare different regression models using the R² Score and select the model with the best prediction performance.

## Problem Statement

Develop a Machine Learning model that predicts insurance charges based on:

- Age
- BMI
- Number of Children
- Sex
- Smoking Status

This is a **Supervised Learning – Regression** problem because `charges` is a continuous numerical target variable.

## Dataset

The insurance dataset contains:

- **Rows:** 1,338
- **Columns:** 6

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

Categorical variables such as `sex` and `smoker` were converted into numerical values using One-Hot Encoding with `drop_first=True`.

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

The final input features were:

```text
age
bmi
children
sex_male
smoker_yes
```

## Train-Test Split

The dataset was divided using a 70:30 train-test split.

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

### 1. Multiple Linear Regression

Best R² Score:

**0.7893**

### 2. Support Vector Regression (SVR)

Best configuration:

- Kernel: `linear`
- C: `3000`

Best R² Score:

**0.7590**

### 3. Decision Tree Regression

Best configuration:

- Criterion: `squared_error`
- Splitter: `best`
- Max Features: `sqrt`

Best R² Score:

**0.7839**

### 4. Random Forest Regression

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

The **Random Forest Regressor** was selected because it achieved the highest R² score among all tested models.

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

## Prediction

The trained model was saved using Python's `pickle` module.

```python
import pickle

load_model = pickle.load(
    open('final_Model_Insurance_Charges_Predict.sav', 'rb')
)
```

The prediction program accepts:

- Age
- BMI
- Number of Children
- Sex (`male` / `female`)
- Smoking Status (`yes` / `no`)

Categorical inputs are automatically converted to the numerical values used during model training.

Example:

```text
Enter your Age: 23
Enter your BMI: 25
Enter No. of Children: 1
Enter your Sex (male/female): male
Are you a Smoker? (yes/no): no
```

## Project Structure

```text
Regression Assignment/
│
├── Regression_Assignment.ipynb
├── insurance_pre.csv
├── final_Model_Insurance_Charges_Predict.sav
├── Regression_Assignment_Report.pdf
└── README.md
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- Pickle
- GitHub

## Evaluation Metric

The primary evaluation metric is **R² Score**.

A higher R² score indicates better predictive performance.

## Conclusion

Four regression algorithms were developed and evaluated: Multiple Linear Regression, Support Vector Regression, Decision Tree Regression and Random Forest Regression.

After comparing the experimental R² scores, **Random Forest Regression achieved the highest score of 0.8667**.

Therefore, the Random Forest Regressor with `criterion='absolute_error'`, `max_features='sqrt'`, and `n_estimators=100` was selected as the final model for predicting insurance charges.

## Author

**Sanjay R**

Regression Assignment – Machine Learning
