# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# pandas/numpy → data manipulation
import pandas as pd
import numpy as np

# seaborn/matplotlib → visualization
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn → machine learning pipeline (optional)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# make plots look better
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ============================================================
# 2. LOAD DATA
# ============================================================

# Replace filename with your actual Kaggle CSV
df = pd.read_csv("student_data.csv", sep=";")

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns)

print("\n--- First Rows ---")
print(df.head())


# ============================================================
# 3. BASIC DATA INFO & CLEANING
# ============================================================

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isna().sum())

# (Dataset typically has no missing values, but in case:)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================

print("\n--- Numerical Summary ---")
print(df.describe())

# Identify categorical features (you can adjust this)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("\n--- Categorical Columns ---")
print(categorical_cols)

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())


# ============================================================
# 5. UNIVARIATE ANALYSIS (Distribution Plots)
# ============================================================

# numerical distributions
df[num_cols].hist(bins=20, figsize=(14, 10))
plt.suptitle("Numerical Feature Distributions")
plt.show()

# categorical distributions
for col in categorical_cols:
    plt.figure()
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(f"Count Plot: {col}")
    plt.show()


# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================

corr = df[num_cols].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Between Numerical Features")
plt.show()


# ============================================================
# 7. RELATIONSHIP BETWEEN FEATURES & TARGET
# ============================================================

# Make sure the target column name is correct:
TARGET = "target"      # change to: "Target", "output", or whatever the column is named

if TARGET not in df.columns:
    raise ValueError("⚠️ Update TARGET variable to match the dataset's target column.")

# class balance
print("\n--- Target Class Distribution ---")
print(df[TARGET].value_counts(normalize=True))

# relationship numeric → target
for col in num_cols:
    plt.figure()
    sns.boxplot(data=df, x=TARGET, y=col)
    plt.title(f"{col} vs {TARGET}")
    plt.xticks(rotation=45)
    plt.show()

# relationship categorical → target
for col in categorical_cols:
    plt.figure()
    sns.countplot(data=df, x=col, hue=TARGET)
    plt.title(f"{col} vs {TARGET}")
    plt.xticks(rotation=45)
    plt.show()


# ============================================================
# 8. FEATURE ENGINEERING EXAMPLES
# ============================================================

# Example: pass rate for 1st/2nd semester units
if "curricular_units_1st_sem_enrolled" in df.columns:
    df["pass_rate_1st_sem"] = (
        df["curricular_units_1st_sem_approved"] /
        df["curricular_units_1st_sem_enrolled"].replace(0, np.nan)
    )

if "curricular_units_2nd_sem_enrolled" in df.columns:
    df["pass_rate_2nd_sem"] = (
        df["curricular_units_2nd_sem_approved"] /
        df["curricular_units_2nd_sem_enrolled"].replace(0, np.nan)
    )

print("\n--- Pass Rate Features Added ---")
print(df[["pass_rate_1st_sem", "pass_rate_2nd_sem"]].head())


# ============================================================
# 9. BASIC MACHINE LEARNING PIPELINE (Random Forest)
# ============================================================

# separate features + target
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# new list of numerical/categorical columns after feature engineering
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# model
model = RandomForestClassifier(random_state=42)

# complete pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# train model
clf.fit(X_train, y_train)

# predictions
y_pred = clf.predict(X_test)

# evaluation
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

# extract feature names after OneHotEncoder expansion
ohe = clf.named_steps["preprocessor"].named_transformers_["cat"]
ohe_names = ohe.get_feature_names_out(cat_cols)

feature_names = num_cols + list(ohe_names)
importances = clf.named_steps["model"].feature_importances_

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n--- Top 15 Most Important Features ---")
print(feat_imp.head(15))

# plot top features
plt.figure(figsize=(8, 6))
feat_imp.head(15).plot(kind="barh")
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.show()
