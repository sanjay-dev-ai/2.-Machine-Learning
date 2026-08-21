# Final Model – Insurance Charges Prediction

## Overview

This folder contains the final selected Machine Learning model for the Insurance Charges Regression Assignment.

After comparing Multiple Linear Regression, Support Vector Regression, Decision Tree Regression, and Random Forest Regression, **Random Forest Regression** achieved the highest R² Score.

## Final Model

### Algorithm

**Random Forest Regressor**

### Final Parameters

```python
RandomForestRegressor(
    criterion='absolute_error',
    max_features='sqrt',
    n_estimators=100,
    random_state=0
)
```

### Performance

**R² Score: 0.8667**

This was the highest R² Score obtained among the evaluated models, so it was selected as the final model.

## Files in This Folder

### 1. Model Creation Notebook

`Random_Forest_regression_Phase-1_Model_Creation.ipynb`

This notebook contains the final model development process, including:

- Dataset loading
- Data preprocessing
- Train-test split
- Random Forest model creation
- Model training
- Prediction
- R² Score evaluation
- Saving the trained model

### 2. Deployment Notebook

`Random_Forest_regression_Phase-2_Deployment.ipynb`

This notebook loads the saved model and allows the user to enter values manually to predict insurance charges.

The user can enter:

- Age
- BMI
- Number of Children
- Sex
- Smoking Status

### 3. Saved Model

`final_Model_Insurance_Charges_Predict.sav`

This is the trained Random Forest model saved using Python's `pickle` module.

It can be loaded using:

```python
import pickle

load_model = pickle.load(
    open('final_Model_Insurance_Charges_Predict.sav', 'rb')
)
```

## User Input and Encoding

The original dataset contains categorical values for `sex` and `smoker`.

During preprocessing, these were converted using:

```python
dataset = pd.get_dummies(dataset, drop_first=True)
```

The resulting columns are:

```text
sex_male
smoker_yes
```

Therefore, the deployment notebook converts user-friendly inputs as follows:

| User Input | Model Value |
|---|---:|
| `female` | `sex_male = 0` |
| `male` | `sex_male = 1` |
| `no` | `smoker_yes = 0` |
| `yes` | `smoker_yes = 1` |

For example:

```text
Age: 23
BMI: 25
Children: 1
Sex: male
Smoker: no
```

is converted internally to:

```text
[23, 25, 1, 1, 0]
```

The trained model then predicts the estimated insurance charges.

## Final Model Justification

Random Forest Regression was selected because it produced the highest R² Score among the models tested.

| Model | Best R² Score |
|---|---:|
| Multiple Linear Regression | 0.7893 |
| Support Vector Regression | 0.7590 |
| Decision Tree Regression | 0.7839 |
| **Random Forest Regression** | **0.8667** |

Therefore, the Random Forest Regressor was chosen as the final model for insurance charge prediction.

## Project Flow

```text
Dataset
   ↓
Data Pre-processing
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Hyperparameter Testing
   ↓
Model Comparison
   ↓
Random Forest Selected
   ↓
Model Saved as .sav
   ↓
Deployment Notebook
   ↓
User Input
   ↓
Predicted Insurance Charges
```

## Author

**Sanjay R**

Machine Learning – Regression Assignment
