# Part 2 – Supervised Machine Learning

## Overview

This project builds and evaluates Regression and Classification models using `cleaned_data.csv`.

---

## Dataset

* File: `cleaned_data.csv`
* Features (X): All columns except target column.
* Regression Target (y_reg): Last numeric column in the dataset.
* Classification Target (y_clf): Created using:

```python
y_clf = (y_reg > y_reg.median()).astype(int)
```

---

## Data Preprocessing

### Encoding

* Categorical columns were encoded using One-Hot Encoding.
* `drop_first=True` was used to avoid duplicate dummy variables.

### Train-Test Split

* Training Data: 80%
* Testing Data: 20%
* Random State: 42

### Feature Scaling

* StandardScaler was fitted only on training data.
* This prevents data leakage from the test set.

---

## Regression Model

### Linear Regression

Metrics:

* Mean Squared Error (MSE): __________
* R² Score: __________

### Top Important Features

1. ---
2. ---
3. ---

### Ridge Regression

* Alpha = 1.0

| Model             | MSE  | R²   |
| ----------------- | ---- | ---- |
| Linear Regression | ____ | ____ |
| Ridge Regression  | ____ | ____ |

---

## Classification Model

### Logistic Regression

* Max Iterations = 1000
* Class Weight = Balanced (if needed)

### Confusion Matrix

Generated after running the model.

### Classification Metrics

* Accuracy: __________
* Precision: __________
* Recall: __________
* F1 Score: __________

---

## Precision and Recall

**Precision Formula**

Precision = TP / (TP + FP)

**Recall Formula**

Recall = TP / (TP + FN)

---

## ROC Curve and AUC

* ROC Curve plotted using `roc_curve()`
* AUC Score: __________

### AUC Meaning

AUC shows how well the model separates the two classes.

* 0.5 = Random
* 1.0 = Perfect Model

---

## Threshold Analysis

| Threshold | Precision | Recall | F1   |
| --------- | --------- | ------ | ---- |
| 0.30      | ____      | ____   | ____ |
| 0.40      | ____      | ____   | ____ |
| 0.50      | ____      | ____   | ____ |
| 0.60      | ____      | ____   | ____ |
| 0.70      | ____      | ____   | ____ |

### Best Threshold

Threshold with highest F1 Score: __________

---

## Regularization Experiment

| Model                        | Precision | Recall | AUC  |
| ---------------------------- | --------- | ------ | ---- |
| Logistic Regression (C=1.0)  | ____      | ____   | ____ |
| Logistic Regression (C=0.01) | ____      | ____   | ____ |

### C Parameter

* Large C = Less Regularization
* Small C = More Regularization

---

## Bootstrap AUC Difference

500 bootstrap samples were generated.

| Metric              | Value |
| ------------------- | ----- |
| Mean AUC Difference | ____  |
| Lower 95% CI        | ____  |
| Upper 95% CI        | ____  |

### Interpretation

* If CI excludes 0 → Difference is reliable.
* If CI includes 0 → Difference may not be significant.

---

## Conclusion

This project successfully implemented:

* Data Preprocessing
* Linear Regression
* Ridge Regression
* Logistic Regression
* ROC & AUC Analysis
* Threshold Analysis
* Regularization Comparison
* Bootstrap Confidence Interval Analysis
