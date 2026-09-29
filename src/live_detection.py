import time
import random
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/driving_behavior_model.pkl"
FEATURE_FILE = "models/feature_columns.pkl"

WINDOW_SIZE = 20

SAMPLE_DELAY = 0.15


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LIVE VEHICLE SENSOR MONITOR")
print("=" * 70)
print()

print("Loading ML model...")

model = joblib.load(MODEL_FILE)

feature_columns = joblib.load(FEATURE_FILE)

print("Model loaded successfully!")

print()


# ============================================================
# SENSOR SIMULATOR
# ============================================================

def generate_sensor_data(mode):

    # --------------------------------------------------------
    # NORMAL DRIVING
    # --------------------------------------------------------

    if mode == "NORMAL":

        acc_x = np.random.normal(0, 0.35)
        acc_y = np.random.normal(0, 0.35)
        acc_z = np.random.normal(0, 0.35)

        gyro_x = np.random.normal(0, 0.05)
        gyro_y = np.random.normal(0, 0.05)
        gyro_z = np.random.normal(0, 0.05)


    # --------------------------------------------------------
    # SLOW DRIVING
    # --------------------------------------------------------

    elif mode == "SLOW":

        acc_x = np.random.normal(0, 0.15)
        acc_y = np.random.normal(0, 0.15)
        acc_z = np.random.normal(0, 0.15)

        gyro_x = np.random.normal(0, 0.025)
        gyro_y = np.random.normal(0, 0.025)
        gyro_z = np.random.normal(0, 0.025)


    # --------------------------------------------------------
    # AGGRESSIVE DRIVING
    # --------------------------------------------------------

    elif mode == "AGGRESSIVE":

        acc_x = np.random.normal(0, 1.2)
        acc_y = np.random.normal(0, 1.2)
        acc_z = np.random.normal(0, 1.2)

        gyro_x = np.random.normal(0, 0.20)
        gyro_y = np.random.normal(0, 0.20)
        gyro_z = np.random.normal(0, 0.20)


    else:

        acc_x = 0
        acc_y = 0
        acc_z = 0

        gyro_x = 0
        gyro_y = 0
        gyro_z = 0


    return {

        "AccX": acc_x,
        "AccY": acc_y,
        "AccZ": acc_z,

        "GyroX": gyro_x,
        "GyroY": gyro_y,
        "GyroZ": gyro_z

    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(sensor_window):

    df = pd.DataFrame(sensor_window)


    # --------------------------------------------------------
    # ACCELERATION MAGNITUDE
    # --------------------------------------------------------

    df["AccelMagnitude"] = np.sqrt(

        df["AccX"] ** 2
        + df["AccY"] ** 2
        + df["AccZ"] ** 2

    )


    # --------------------------------------------------------
    # GYROSCOPE MAGNITUDE
    # --------------------------------------------------------

    df["GyroMagnitude"] = np.sqrt(

        df["GyroX"] ** 2
        + df["GyroY"] ** 2
        + df["GyroZ"] ** 2

    )


    # --------------------------------------------------------
    # JERK
    # --------------------------------------------------------

    df["Jerk"] = (

        df["AccelMagnitude"]
        .diff()
        .fillna(0)

    )


    # --------------------------------------------------------
    # GYRO CHANGE
    # --------------------------------------------------------

    df["GyroChange"] = (

        df["GyroMagnitude"]
        .diff()
        .fillna(0)

    )


    features = {}


    # --------------------------------------------------------
    # ACCELERATION FEATURES
    # --------------------------------------------------------

    for axis in ["AccX", "AccY", "AccZ"]:

        features[f"{axis}_mean"] = (
            df[axis].mean()
        )

        features[f"{axis}_std"] = (
            df[axis].std()
        )

        features[f"{axis}_min"] = (
            df[axis].min()
        )

        features[f"{axis}_max"] = (
            df[axis].max()
        )


    # --------------------------------------------------------
    # ACCELERATION MAGNITUDE
    # --------------------------------------------------------

    features["AccelMag_mean"] = (
        df["AccelMagnitude"].mean()
    )

    features["AccelMag_std"] = (
        df["AccelMagnitude"].std()
    )

    features["AccelMag_min"] = (
        df["AccelMagnitude"].min()
    )

    features["AccelMag_max"] = (
        df["AccelMagnitude"].max()
    )


    # --------------------------------------------------------
    # JERK
    # --------------------------------------------------------

    features["Jerk_mean"] = (
        df["Jerk"].abs().mean()
    )

    features["Jerk_std"] = (
        df["Jerk"].std()
    )

    features["Jerk_max"] = (
        df["Jerk"].abs().max()
    )


    # --------------------------------------------------------
    # GYROSCOPE FEATURES
    # --------------------------------------------------------

    for axis in ["GyroX", "GyroY", "GyroZ"]:

        features[f"{axis}_mean"] = (
            df[axis].mean()
        )

        features[f"{axis}_std"] = (
            df[axis].std()
        )

        features[f"{axis}_min"] = (
            df[axis].min()
        )

        features[f"{axis}_max"] = (
            df[axis].max()
        )


    # --------------------------------------------------------
    # GYRO MAGNITUDE
    # --------------------------------------------------------

    features["GyroMag_mean"] = (
        df["GyroMagnitude"].mean()
    )

    features["GyroMag_std"] = (
        df["GyroMagnitude"].std()
    )

    features["GyroMag_min"] = (
        df["GyroMagnitude"].min()
    )

    features["GyroMag_max"] = (
        df["GyroMagnitude"].max()
    )


    # --------------------------------------------------------
    # GYRO CHANGE
    # --------------------------------------------------------

    features["GyroChange_mean"] = (
        df["GyroChange"].abs().mean()
    )

    features["GyroChange_max"] = (
        df["GyroChange"].abs().max()
    )


    return pd.DataFrame(
        [features],
        columns=feature_columns
    )


# ============================================================
# DISPLAY
# ============================================================

def display_prediction(
    sensor,
    prediction,
    probabilities
):

    print(
        "\033[2J\033[H",
        end=""
    )

    print("=" * 70)

    print(
        "🚗 LIVE VEHICLE ACCIDENT DETECTION SYSTEM"
    )

    print("=" * 70)

    print()

    print("LIVE MPU6050 SENSOR DATA")

    print("-" * 70)

    print(
        f"AccX : {sensor['AccX']:7.3f}   "
        f"AccY : {sensor['AccY']:7.3f}   "
        f"AccZ : {sensor['AccZ']:7.3f}"
    )

    print(
        f"GyroX: {sensor['GyroX']:7.3f}   "
        f"GyroY: {sensor['GyroY']:7.3f}   "
        f"GyroZ: {sensor['GyroZ']:7.3f}"
    )

    print()

    print("ML PREDICTION")

    print("-" * 70)

    print(
        f"Detected behaviour: {prediction}"
    )

    print()

    for class_name, probability in probabilities.items():

        print(
            f"{class_name:<12} "
            f"{probability * 100:6.2f}%"
        )

    print()

    print("-" * 70)

    if prediction == "AGGRESSIVE":

        print(
            "⚠️  WARNING: AGGRESSIVE DRIVING DETECTED"
        )

    elif prediction == "SLOW":

        print(
            "🐢 SLOW / LOW-INTENSITY DRIVING"
        )

    else:

        print(
            "✅ NORMAL DRIVING"
        )

    print("-" * 70)

    print()

    print(
        "Press CTRL+C to stop."
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SELECT SIMULATION MODE")
    print("=" * 70)
    print()

    print("1. NORMAL")
    print("2. SLOW")
    print("3. AGGRESSIVE")
    print("4. RANDOM")

    print()

    choice = input(
        "Enter choice: "
    ).strip()


    if choice == "1":

        mode = "NORMAL"

    elif choice == "2":

        mode = "SLOW"

    elif choice == "3":

        mode = "AGGRESSIVE"

    elif choice == "4":

        mode = "RANDOM"

    else:

        print(
            "Invalid choice."
        )

        return


    print()

    print(
        f"Simulation mode: {mode}"
    )

    print()

    print(
        "Collecting sensor samples..."
    )

    time.sleep(2)


    sensor_window = []


    try:

        while True:


            # ------------------------------------------------
            # SELECT CURRENT MODE
            # ------------------------------------------------

            if mode == "RANDOM":

                current_mode = random.choice(
                    [
                        "NORMAL",
                        "SLOW",
                        "AGGRESSIVE"
                    ]
                )

            else:

                current_mode = mode


            # ------------------------------------------------
            # GENERATE SENSOR READING
            # ------------------------------------------------

            sensor = generate_sensor_data(
                current_mode
            )


            sensor_window.append(
                sensor
            )


            # Keep only latest samples

            if len(sensor_window) > WINDOW_SIZE:

                sensor_window.pop(0)


            # ------------------------------------------------
            # WAIT UNTIL WINDOW IS FULL
            # ------------------------------------------------

            if len(sensor_window) < WINDOW_SIZE:

                print(
                    f"Collecting "
                    f"{len(sensor_window)}/"
                    f"{WINDOW_SIZE} samples..."
                )

                time.sleep(
                    SAMPLE_DELAY
                )

                continue


            # ------------------------------------------------
            # EXTRACT FEATURES
            # ------------------------------------------------

            features = extract_features(
                sensor_window
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                features
            )[0]


            probability_values = (
                model.predict_proba(
                    features
                )[0]
            )


            probabilities = dict(

                zip(

                    model.classes_,

                    probability_values

                )

            )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display_prediction(

                sensor,

                prediction,

                probabilities

            )


            time.sleep(
                SAMPLE_DELAY
            )


    except KeyboardInterrupt:

        print()
        print()
        print("=" * 70)
        print("LIVE SIMULATION STOPPED")
        print("=" * 70)
        print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()