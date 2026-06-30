import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv("data/cleaned_data.csv")

# Select target column
target_col = df.select_dtypes(include=np.number).columns[-1]

# Regression Target
y_reg = df[target_col]

# Classification Target
y_clf = (y_reg > y_reg.median()).astype(int)

# Features
X = df.drop(columns=[target_col])

# ======================================
# ENCODING
# ======================================

cat_cols = X.select_dtypes(include=['object']).columns

X = pd.get_dummies(
    X,
    columns=cat_cols,
    drop_first=True
)

feature_names = X.columns

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_clf,
    test_size=0.2,
    random_state=42
)

# ======================================
# SCALING
# ======================================

scaler = StandardScaler()

X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# ======================================
# LINEAR REGRESSION
# ======================================

lr = LinearRegression()

lr.fit(
    X_train_reg_scaled,
    y_train_reg
)

y_pred_reg = lr.predict(X_test_reg_scaled)

mse = mean_squared_error(
    y_test_reg,
    y_pred_reg
)

r2 = r2_score(
    y_test_reg,
    y_pred_reg
)

print("\nLINEAR REGRESSION")
print("MSE:", mse)
print("R2:", r2)

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": lr.coef_
})

print("\nCoefficients")
print(coef_df)

top3 = coef_df.reindex(
    coef_df.Coefficient.abs().sort_values(
        ascending=False
    ).index
).head(3)

print("\nTop 3 Features")
print(top3)

# RIDGE REGRESSION


ridge = Ridge(alpha=1.0)

ridge.fit(
    X_train_reg_scaled,
    y_train_reg
)

ridge_pred = ridge.predict(
    X_test_reg_scaled
)

ridge_mse = mean_squared_error(
    y_test_reg,
    ridge_pred
)

ridge_r2 = r2_score(
    y_test_reg,
    ridge_pred
)

print("\nRIDGE RESULTS")
print("MSE:", ridge_mse)
print("R2:", ridge_r2)

# CLASS IMBALANCE CHECK


print("\nClass Counts Before")
print(y_train_clf.value_counts())

ratio = y_train_clf.value_counts().min() / len(y_train_clf)

if ratio < 0.35:
    class_weight = "balanced"
else:
    class_weight = None

# LOGISTIC REGRESSION


log_model = LogisticRegression(
    max_iter=1000,
    class_weight=class_weight
)

log_model.fit(
    X_train_clf_scaled,
    y_train_clf
)

y_pred = log_model.predict(
    X_test_clf_scaled
)

y_prob = log_model.predict_proba(
    X_test_clf_scaled
)[:, 1]

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_test_clf,
    y_pred
))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test_clf,
    y_pred
))

auc = roc_auc_score(
    y_test_clf,
    y_prob
)

print("AUC:", auc)

# ROC CURVE


fpr, tpr, _ = roc_curve(
    y_test_clf,
    y_prob
)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC={auc:.3f})")
plt.savefig("outputs/roc_curve.png")
plt.show()


# THRESHOLD ANALYSIS


print("\nTHRESHOLD ANALYSIS")

thresholds = [0.30,0.40,0.50,0.60,0.70]

results=[]

for t in thresholds:

    pred = (y_prob >= t).astype(int)

    precision = precision_score(
        y_test_clf,
        pred
    )

    recall = recall_score(
        y_test_clf,
        pred
    )

    f1 = f1_score(
        y_test_clf,
        pred
    )

    results.append(
        [t,precision,recall,f1]
    )

threshold_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1"
    ]
)

print(threshold_df)

threshold_df.to_csv(
    "outputs/threshold_analysis.csv",
    index=False
)

# REGULARIZATION EXPERIMENT

log_small_c = LogisticRegression(
    C=0.01,
    max_iter=1000
)

log_small_c.fit(
    X_train_clf_scaled,
    y_train_clf
)

prob_small = log_small_c.predict_proba(
    X_test_clf_scaled
)[:,1]

pred_small = log_small_c.predict(
    X_test_clf_scaled
)

auc_small = roc_auc_score(
    y_test_clf,
    prob_small
)

print("\nC=0.01 AUC:", auc_small)


# BOOTSTRAP CI

diffs=[]

for _ in range(500):

    idx = np.random.choice(
        len(y_test_clf),
        size=len(y_test_clf),
        replace=True
    )

    y_boot = y_test_clf.iloc[idx]

    auc1 = roc_auc_score(
        y_boot,
        y_prob[idx]
    )

    auc2 = roc_auc_score(
        y_boot,
        prob_small[idx]
    )

    diffs.append(
        auc1 - auc2
    )

mean_diff = np.mean(diffs)

lower = np.percentile(
    diffs,
    2.5
)

upper = np.percentile(
    diffs,
    97.5
)

print("\nBootstrap Results")
print("Mean Difference:", mean_diff)
print("95% CI:", lower, upper)
