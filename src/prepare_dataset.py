import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/raw/driving_behavior/train_motion_data.csv"

OUTPUT_FILE = "data/processed/driving_features.csv"


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("ACCIDENT DETECTION ML PROJECT")
print("STEP 1 - LOADING KAGGLE DRIVING DATA")
print("=" * 70)
print()

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)
print("Columns:")
print(df.columns.tolist())
print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "AccX",
    "AccY",
    "AccZ",
    "GyroX",
    "GyroY",
    "GyroZ",
    "Class",
    "Timestamp"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

df = df[required_columns].copy()


# ============================================================
# CONVERT SENSOR VALUES TO NUMERIC
# ============================================================

sensor_columns = [
    "AccX",
    "AccY",
    "AccZ",
    "GyroX",
    "GyroY",
    "GyroZ"
]

for column in sensor_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before = len(df)

df = df.dropna(
    subset=sensor_columns + ["Class"]
)

after = len(df)

print("Rows before cleaning:", before)
print("Rows after cleaning :", after)
print()


# ============================================================
# SORT BY TIMESTAMP
# ============================================================

df = df.sort_values(
    "Timestamp"
).reset_index(
    drop=True
)


# ============================================================
# ACCELERATION MAGNITUDE
# ============================================================

df["AccelMagnitude"] = np.sqrt(

    df["AccX"] ** 2
    + df["AccY"] ** 2
    + df["AccZ"] ** 2

)


# ============================================================
# GYROSCOPE MAGNITUDE
# ============================================================

df["GyroMagnitude"] = np.sqrt(

    df["GyroX"] ** 2
    + df["GyroY"] ** 2
    + df["GyroZ"] ** 2

)


# ============================================================
# JERK
# ============================================================

df["Jerk"] = (

    df["AccelMagnitude"]
    .diff()
    .fillna(0)

)


# ============================================================
# GYRO CHANGE
# ============================================================

df["GyroChange"] = (

    df["GyroMagnitude"]
    .diff()
    .fillna(0)

)


# ============================================================
# WINDOW CONFIGURATION
# ============================================================

WINDOW_SIZE = 20

STEP_SIZE = 10


# ============================================================
# FEATURE EXTRACTION
# ============================================================

feature_rows = []


for start in range(

    0,

    len(df) - WINDOW_SIZE + 1,

    STEP_SIZE

):

    window = df.iloc[
        start:start + WINDOW_SIZE
    ]


    # --------------------------------------------------------
    # Determine dominant class
    # --------------------------------------------------------

    class_label = (

        window["Class"]
        .mode()
        .iloc[0]

    )


    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    features = {}


    # ========================================================
    # ACCELERATION FEATURES
    # ========================================================

    for axis in ["AccX", "AccY", "AccZ"]:

        features[f"{axis}_mean"] = (
            window[axis].mean()
        )

        features[f"{axis}_std"] = (
            window[axis].std()
        )

        features[f"{axis}_min"] = (
            window[axis].min()
        )

        features[f"{axis}_max"] = (
            window[axis].max()
        )


    # ========================================================
    # ACCELERATION MAGNITUDE
    # ========================================================

    features["AccelMag_mean"] = (
        window["AccelMagnitude"].mean()
    )

    features["AccelMag_std"] = (
        window["AccelMagnitude"].std()
    )

    features["AccelMag_min"] = (
        window["AccelMagnitude"].min()
    )

    features["AccelMag_max"] = (
        window["AccelMagnitude"].max()
    )


    # ========================================================
    # JERK FEATURES
    # ========================================================

    features["Jerk_mean"] = (
        window["Jerk"].abs().mean()
    )

    features["Jerk_std"] = (
        window["Jerk"].std()
    )

    features["Jerk_max"] = (
        window["Jerk"].abs().max()
    )


    # ========================================================
    # GYROSCOPE FEATURES
    # ========================================================

    for axis in ["GyroX", "GyroY", "GyroZ"]:

        features[f"{axis}_mean"] = (
            window[axis].mean()
        )

        features[f"{axis}_std"] = (
            window[axis].std()
        )

        features[f"{axis}_min"] = (
            window[axis].min()
        )

        features[f"{axis}_max"] = (
            window[axis].max()
        )


    # ========================================================
    # GYROSCOPE MAGNITUDE
    # ========================================================

    features["GyroMag_mean"] = (
        window["GyroMagnitude"].mean()
    )

    features["GyroMag_std"] = (
        window["GyroMagnitude"].std()
    )

    features["GyroMag_min"] = (
        window["GyroMagnitude"].min()
    )

    features["GyroMag_max"] = (
        window["GyroMagnitude"].max()
    )


    # ========================================================
    # GYRO CHANGE
    # ========================================================

    features["GyroChange_mean"] = (
        window["GyroChange"].abs().mean()
    )

    features["GyroChange_max"] = (
        window["GyroChange"].abs().max()
    )


    # ========================================================
    # LABEL
    # ========================================================

    features["Class"] = class_label


    feature_rows.append(features)


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

feature_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# CLEAN FEATURE DATA
# ============================================================

feature_df = feature_df.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_df = feature_df.dropna()

feature_df = feature_df.reset_index(
    drop=True
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 70)
print()

print(
    "Original rows:",
    len(df)
)

print(
    "Feature windows:",
    len(feature_df)
)

print(
    "Number of ML features:",
    len(feature_df.columns) - 1
)

print()

print("CLASS DISTRIBUTION")
print("-" * 70)

print(
    feature_df["Class"].value_counts()
)

print()

print("CLASS PERCENTAGE")
print("-" * 70)

print(
    feature_df["Class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print()

print(
    "Saved file:",
    OUTPUT_FILE
)

print()