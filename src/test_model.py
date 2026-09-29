import joblib
import numpy as np
import pandas as pd


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = (
    "models/"
    "driving_behavior_model.pkl"
)

FEATURE_FILE = (
    "models/"
    "feature_columns.pkl"
)


model = joblib.load(
    MODEL_FILE
)

feature_columns = joblib.load(
    FEATURE_FILE
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = (
    "data/processed/"
    "driving_features.csv"
)

df = pd.read_csv(
    DATA_FILE
)


# ============================================================
# SELECT RANDOM SAMPLE
# ============================================================

sample = df.sample(
    1,
    random_state=None
)


actual_class = sample["Class"].iloc[0]


X_sample = sample.drop(
    columns=["Class"]
)


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(
    X_sample
)[0]


probabilities = model.predict_proba(
    X_sample
)[0]


classes = model.classes_


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 60)
print("DRIVING BEHAVIOUR PREDICTION")
print("=" * 60)
print()

print(
    "Actual class     :",
    actual_class
)

print(
    "Predicted class  :",
    prediction
)

print()

print("Prediction probabilities:")

for class_name, probability in zip(
    classes,
    probabilities
):

    print(
        f"{class_name:<12}: "
        f"{probability * 100:.2f}%"
    )

print()

if prediction == actual_class:

    print("Prediction: CORRECT")

else:

    print("Prediction: INCORRECT")

print()