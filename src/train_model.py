import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/ciren_severity.csv"
MODEL_PATH = "models/ciren_severity_model.pkl"


print("=" * 70)
print("IMPROVED CIREN SEVERITY MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nTarget distribution:")
print(df["SEVERITY"].value_counts())


# ============================================================
# REMOVE HIGHLY INCOMPLETE FEATURES
# ============================================================

missing_percentage = df.isnull().mean() * 100

remove_columns = []

print("\nRemoving highly incomplete features:")

for column, percentage in missing_percentage.items():

    if percentage > 80:

        remove_columns.append(column)

        print(
            f"  {column}: "
            f"{percentage:.2f}% missing"
        )


# Never remove target
if "SEVERITY" in remove_columns:
    remove_columns.remove("SEVERITY")


df = df.drop(columns=remove_columns)


# ============================================================
# SELECT FEATURES
# ============================================================

TARGET = "SEVERITY"

# Remove identifiers and target-related columns
columns_to_remove = [
    TARGET,
    "SEVERITY_CODE",
    "CASENO",
    "CIRENID",
    "VEHNO",
    "CASEYEAR"
]

X = df.drop(
    columns=[
        c for c in columns_to_remove
        if c in df.columns
    ]
)

y = df[TARGET]


# ============================================================
# KEEP NUMERIC FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["int64", "float64"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# For the current CIREN dataset we use numeric features.
X = X[numeric_features]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records :", len(X_train))
print("Testing records  :", len(X_test))


print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)


# ============================================================
# BALANCED RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=500,

    random_state=42,

    # IMPORTANT:
    # Give more importance to MINOR and CRITICAL
    # instead of allowing SERIOUS to dominate.

    class_weight="balanced",

    max_features="sqrt",

    min_samples_leaf=2,

    n_jobs=-1
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([

    (
        "imputer",
        imputer
    ),

    (
        "classifier",
        model
    )
])


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BALANCED RANDOM FOREST")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)

print("\nTraining completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy          : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=2,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "MINOR",
    "SERIOUS",
    "CRITICAL"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("\nConfusion Matrix:")
print(
    "              Predicted"
)

print(
    "              MINOR  SERIOUS  CRITICAL"
)

for label, row in zip(labels, cm):

    print(
        f"{label:<12}",
        row
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"\n{MODEL_PATH}"
)

print("\nDONE")