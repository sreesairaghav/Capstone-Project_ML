"""
CAPSTONE PROJECT - ML MODEL EVALUATION
======================================

Evaluates:

1. Driving Behaviour Model
   NORMAL / SLOW / AGGRESSIVE

2. CIREN Crash Severity Model
   MINOR / SERIOUS / CRITICAL

Metrics:
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- Macro F1
- Weighted F1
- ROC-AUC
- Confusion Matrix
- Classification Report

IMPORTANT:
Models in this project were saved using JOBLIB.
Therefore joblib.load() is used instead of pickle.load().
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# UTILITY
# ============================================================

def load_model(path):

    print(f"\nLoading: {path}")

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found:\n{path}"
        )

    # IMPORTANT:
    # Models were created using joblib.dump()
    model = joblib.load(path)

    print(
        f"Loaded successfully: "
        f"{type(model).__name__}"
    )

    return model


def print_header(title):

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    model,
    X_test,
    model_name
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print_header(
        f"{model_name} - RESULTS"
    )

    print(
        f"\nAccuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision   : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Macro Recall      : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Macro F1          : "
        f"{f1_macro * 100:.2f}%"
    )

    print(
        f"Weighted F1       : "
        f"{f1_weighted * 100:.2f}%"
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = np.nan

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(
                X_test
            )

            classes = model.classes_

            if len(classes) == 2:

                roc_auc = roc_auc_score(
                    y_true,
                    probabilities[:, 1]
                )

            else:

                roc_auc = roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                    labels=classes
                )

            print(
                f"ROC-AUC           : "
                f"{roc_auc * 100:.2f}%"
            )

        except Exception as e:

            print(
                f"\nROC-AUC unavailable: {e}"
            )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n")
    print("CLASSIFICATION REPORT")
    print("-" * 75)

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("CONFUSION MATRIX")
    print("-" * 75)

    labels = sorted(
        pd.Series(y_true).unique()
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual {x}"
            for x in labels
        ],
        columns=[
            f"Predicted {x}"
            for x in labels
        ]
    )

    print(
        cm_df.to_string()
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Balanced Accuracy":
            balanced_accuracy,

        "Macro Precision":
            precision,

        "Macro Recall":
            recall,

        "Macro F1":
            f1_macro,

        "Weighted F1":
            f1_weighted,

        "ROC-AUC":
            roc_auc
    }


# ============================================================
# DRIVING BEHAVIOUR
# ============================================================

def evaluate_driving_model():

    print_header(
        "DRIVING BEHAVIOUR MODEL EVALUATION"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "driving_behavior_model.pkl"
    )

    data_path = os.path.join(
        DATA_DIR,
        "driving_features.csv"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_path
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading driving behaviour dataset..."
    )

    df = pd.read_csv(
        data_path
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # Existing project uses "Class"
    target = "Class"

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' "
            f"not found.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Features / target
    # --------------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # --------------------------------------------------------
    # Display class distribution
    # --------------------------------------------------------

    print("\nClass Distribution")
    print("-" * 50)

    print(
        y.value_counts()
    )

    print("\nClass Distribution (%)")

    print(
        (
            y.value_counts(
                normalize=True
            ) * 100
        ).round(2)
    )

    # --------------------------------------------------------
    # Train/Test split
    #
    # IMPORTANT:
    # This reproduces a stratified 80/20 evaluation split.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print("\nEvaluation Dataset")
    print("-" * 50)

    print(
        f"Training samples : "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples  : "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    return calculate_metrics(

        y_test,

        y_pred,

        model,

        X_test,

        "Driving Behaviour Random Forest"
    )


# ============================================================
# CIREN SEVERITY
# ============================================================

def evaluate_ciren_model():

    print_header(
        "CIREN CRASH SEVERITY MODEL EVALUATION"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "ciren_severity_model.pkl"
    )

    data_path = os.path.join(
        DATA_DIR,
        "ciren_severity.csv"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_path
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading CIREN dataset..."
    )

    df = pd.read_csv(
        data_path
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This EXACTLY follows train_model.py
    # --------------------------------------------------------

    TARGET = "SEVERITY"

    # --------------------------------------------------------
    # Remove highly incomplete columns
    #
    # This reproduces the training process.
    # --------------------------------------------------------

    missing_percentage = (
        df.isnull().mean() * 100
    )

    remove_columns = []

    for column, percentage in (
        missing_percentage.items()
    ):

        if percentage > 80:

            remove_columns.append(
                column
            )

    # Never remove target

    if TARGET in remove_columns:

        remove_columns.remove(
            TARGET
        )

    df = df.drop(
        columns=remove_columns
    )

    print(
        "\nHighly incomplete columns removed:"
    )

    for column in remove_columns:

        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # Remove target/identifier columns
    #
    # EXACTLY matching train_model.py
    # --------------------------------------------------------

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
            c
            for c in columns_to_remove
            if c in df.columns
        ]
    )

    y = df[TARGET]

    # --------------------------------------------------------
    # NUMERIC FEATURES
    #
    # EXACTLY matching train_model.py
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(

        include=[
            "int64",
            "float64"
        ]

    ).columns.tolist()

    X = X[
        numeric_features
    ]

    print(
        f"\nNumber of numeric features: "
        f"{len(numeric_features)}"
    )

    print(
        "\nNumeric features:"
    )

    for feature in numeric_features:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print(
        "\nSeverity Distribution"
    )

    print(
        y.value_counts()
    )

    print(
        "\nSeverity Distribution (%)"
    )

    print(
        (
            y.value_counts(
                normalize=True
            ) * 100
        ).round(2)
    )

    # --------------------------------------------------------
    # Split
    #
    # EXACTLY matching train_model.py
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples :",
        len(X_test)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # ciren_severity_model.pkl contains the entire
    # Pipeline:
    #
    # SimpleImputer
    #       ↓
    # RandomForest
    #
    # Therefore we DON'T manually impute here.
    #
    # pipeline.predict() handles it.
    # --------------------------------------------------------

    print(
        "\nGenerating CIREN predictions..."
    )

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    return calculate_metrics(

        y_test,

        y_pred,

        model,

        X_test,

        "CIREN Crash Severity Random Forest"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 75)

    print(
        "CAPSTONE PROJECT"
    )

    print(
        "MACHINE LEARNING MODEL EVALUATION"
    )

    print("=" * 75)

    results = []

    # ========================================================
    # DRIVING MODEL
    # ========================================================

    try:

        driving_result = (
            evaluate_driving_model()
        )

        results.append(
            driving_result
        )

    except Exception as e:

        print_header(
            "DRIVING MODEL ERROR"
        )

        print(
            repr(e)
        )

    # ========================================================
    # CIREN MODEL
    # ========================================================

    try:

        ciren_result = (
            evaluate_ciren_model()
        )

        results.append(
            ciren_result
        )

    except Exception as e:

        print_header(
            "CIREN MODEL ERROR"
        )

        print(
            repr(e)
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if len(results) > 0:

        print_header(
            "FINAL MODEL SCORE SUMMARY"
        )

        result_df = pd.DataFrame(
            results
        )

        percentage_columns = [

            "Accuracy",

            "Balanced Accuracy",

            "Macro Precision",

            "Macro Recall",

            "Macro F1",

            "Weighted F1",

            "ROC-AUC"
        ]

        display_df = result_df.copy()

        for column in percentage_columns:

            if column in display_df.columns:

                display_df[column] = (

                    display_df[column] * 100

                ).round(2)

        print(
            display_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        output_path = os.path.join(

            DATA_DIR,

            "model_evaluation_results.csv"
        )

        display_df.to_csv(

            output_path,

            index=False
        )

        print(
            f"\nResults saved to:"
        )

        print(
            output_path
        )

    else:

        print(
            "\nNo model evaluations completed."
        )

    print("\n")

    print("=" * 75)

    print(
        "EVALUATION COMPLETE"
    )

    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()