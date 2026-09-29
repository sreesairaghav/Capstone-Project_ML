import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ciren_severity.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "ciren_severity_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CIREN SEVERITY MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "SEVERITY"


print("\nTarget distribution:")
print(df[TARGET].value_counts())


# ============================================================
# REMOVE TARGET / LEAKAGE COLUMNS
# ============================================================

LEAKAGE_COLUMNS = [
    "MAIS",
    "ISS",
    "FATAL",
    "SEVERITY",
    "SEVERITY_CODE",

    # identifiers
    "CASENO",
    "VEHNO",
    "CIRENID"
]


X = df.drop(
    columns=[
        column
        for column in LEAKAGE_COLUMNS
        if column in df.columns
    ]
)

y = df[TARGET]


# ============================================================
# REMOVE FEATURES WITH VERY HIGH MISSINGNESS
#
# A feature missing > 80% of the time is not useful for
# the first model.
# ============================================================

missing_ratio = X.isna().mean()

HIGH_MISSING_COLUMNS = missing_ratio[
    missing_ratio > 0.80
].index.tolist()


print("\nRemoving highly incomplete features:")

for column in HIGH_MISSING_COLUMNS:
    print(
        f"  {column}: "
        f"{missing_ratio[column] * 100:.2f}% missing"
    )


X = X.drop(
    columns=HIGH_MISSING_COLUMNS
)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    column
    for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

if constant_columns:

    print("\nRemoving constant features:")

    for column in constant_columns:
        print(" ", column)

    X = X.drop(
        columns=constant_columns
    )


# ============================================================
# IDENTIFY NUMERIC / CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["int64", "float64", "int32", "float32"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,

    # IMPORTANT:
    # Handles MINOR / SERIOUS / CRITICAL imbalance
    class_weight="balanced",

    random_state=42,
    n_jobs=-1
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


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
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[
            "MINOR",
            "SERIOUS",
            "CRITICAL"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "MINOR",
        "SERIOUS",
        "CRITICAL"
    ]
)

print("\nConfusion Matrix:")

print(
    "              Predicted"
)

print(
    "              MINOR  SERIOUS  CRITICAL"
)

for label, row in zip(
    ["MINOR", "SERIOUS", "CRITICAL"],
    cm
):

    print(
        f"{label:<12}",
        row
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_FILE
)


print("\n" + "=" * 70)

print("MODEL SAVED")

print(
    f"\n{MODEL_FILE}"
)

print("=" * 70)