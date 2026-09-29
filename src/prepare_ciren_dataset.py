import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CIREN_DIR = os.path.join(BASE_DIR, "data", "raw", "ciren")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

CASE_FILE = os.path.join(CIREN_DIR, "case.sas7bdat")
GV_FILE = os.path.join(CIREN_DIR, "gv.sas7bdat")
VE_FILE = os.path.join(CIREN_DIR, "ve.sas7bdat")

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ciren_severity.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CIREN DATA
# ============================================================

print("=" * 70)
print("CIREN SEVERITY DATASET PREPARATION")
print("=" * 70)

print("\nLoading case.sas7bdat...")
case = pd.read_sas(CASE_FILE)

print("Loading gv.sas7bdat...")
gv = pd.read_sas(GV_FILE)

print("Loading ve.sas7bdat...")
ve = pd.read_sas(VE_FILE)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

case.columns = case.columns.str.upper()
gv.columns = gv.columns.str.upper()
ve.columns = ve.columns.str.upper()


print("\nLoaded datasets:")
print(f"CASE : {case.shape}")
print(f"GV   : {gv.shape}")
print(f"VE   : {ve.shape}")


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

for df in [case, gv, ve]:

    for column in df.columns:

        if column not in ["CASENO"]:

            try:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )
            except Exception:
                pass


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_case = [
    "CASENO",
    "VEHNO",
    "MAIS",
    "ISS",
    "FATAL"
]

required_gv = [
    "CASENO",
    "VEHNO"
]

required_ve = [
    "CASENO",
    "VEHNO"
]


print("\nChecking required columns...")

for column in required_case:

    if column not in case.columns:
        raise ValueError(
            f"CASE dataset is missing required column: {column}"
        )

for column in required_gv:

    if column not in gv.columns:
        raise ValueError(
            f"GV dataset is missing required column: {column}"
        )

for column in required_ve:

    if column not in ve.columns:
        raise ValueError(
            f"VE dataset is missing required column: {column}"
        )

print("Required columns found.")


# ============================================================
# SELECT CASE / TARGET INFORMATION
# ============================================================

case_cols = [
    "CASENO",
    "VEHNO",
    "CIRENID",
    "MAIS",
    "ISS",
    "FATAL"
]

case_small = case[case_cols].copy()


# ============================================================
# SELECT GV CRASH FEATURES
#
# These are crash characteristics rather than injury outcomes.
# ============================================================

gv_features = [
    "CASENO",
    "VEHNO",

    # Delta-V / crash energy
    "DVEST",
    "DVTOTAL",
    "DVLAT",
    "DVLONG",
    "ENERGY",
    "IMPACTSP",

    # Crash / vehicle characteristics
    "ACCTYPE",
    "BODYTYPE",
    "CURBWGT",

    # Driver / pre-crash behaviour
    "MANEUVER",
    "PREEVENT",
    "PREMOVE",
    "PREILOC",
    "PREISTAB",

    # Rollover
    "ROLLOVER",
    "ROLLDIST",
    "ROLINTYP",

    # Road / environment
    "LANES",
    "LGTCOND",
    "SURCOND",
    "SURTYPE",
    "SPLIMIT",
    "TRAVELSP",
    "WEATHER",

    # Collision / traffic
    "RELINTER",
    "TRAFCONT",
    "TRAFFLOW",

    # Airbag deployment
    "BAGDEPFV",
    "BAGDEPOV"
]


gv_features = [
    column
    for column in gv_features
    if column in gv.columns
]

gv_small = gv[gv_features].copy()


# ============================================================
# SELECT VE VEHICLE DAMAGE FEATURES
# ============================================================

ve_features = [
    "CASENO",
    "VEHNO",

    # Crush profile
    "DVC1",
    "DVC2",
    "DVC3",
    "DVC4",
    "DVC5",
    "DVC6",
    "DVD",
    "DVL",

    # Deformation
    "EXTENT1",
    "EXTENT2",

    # Direction of force
    "PDOF1",
    "PDOF2",

    # Fire
    "FIRE"
]


ve_features = [
    column
    for column in ve_features
    if column in ve.columns
]

ve_small = ve[ve_features].copy()


# ============================================================
# REMOVE DUPLICATE VEHICLE RECORDS
#
# We want one subject vehicle record per CASE + VEHNO.
# ============================================================

gv_small = gv_small.drop_duplicates(
    subset=["CASENO", "VEHNO"],
    keep="first"
)

ve_small = ve_small.drop_duplicates(
    subset=["CASENO", "VEHNO"],
    keep="first"
)


# ============================================================
# MERGE
#
# CASE -> GV -> VE
#
# Key:
# CASENO + VEHNO
# ============================================================

print("\nMerging CASE + GV...")

df = case_small.merge(
    gv_small,
    on=["CASENO", "VEHNO"],
    how="left"
)

print(f"After CASE + GV: {df.shape}")


print("Merging VE...")

df = df.merge(
    ve_small,
    on=["CASENO", "VEHNO"],
    how="left"
)

print(f"After CASE + GV + VE: {df.shape}")


# ============================================================
# CLEAN MAIS TARGET
# ============================================================

df["MAIS"] = pd.to_numeric(
    df["MAIS"],
    errors="coerce"
)


# CIREN MAIS=9 represents unknown/not applicable category.
# It should not be used as a supervised severity class.
df = df[
    df["MAIS"].isin([1, 2, 3, 4, 5, 6])
].copy()


# ============================================================
# CREATE 3-CLASS SEVERITY TARGET
#
# MAIS 1-2 -> MINOR
# MAIS 3-4 -> SERIOUS
# MAIS 5-6 -> CRITICAL
# ============================================================

def create_severity(mais):

    if mais in [1, 2]:
        return "MINOR"

    elif mais in [3, 4]:
        return "SERIOUS"

    elif mais in [5, 6]:
        return "CRITICAL"

    return np.nan


df["SEVERITY"] = df["MAIS"].apply(
    create_severity
)


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================

df = df.dropna(
    subset=["SEVERITY"]
).copy()


# ============================================================
# HANDLE SPECIAL / NON-MEASUREMENT VALUES
#
# These values are commonly used in CIREN/NASS-style coded
# datasets to represent unknown, not reported or not applicable.
#
# We only replace them for continuous crash measurements.
# We DO NOT globally replace 97/98/99 because those can have
# legitimate categorical meanings depending on the variable.
# ============================================================

continuous_special_codes = {
    "DVEST": [777, 888, 998, 999],
    "DVTOTAL": [777, 888, 998, 999],
    "DVLAT": [777, 888, 998, 999],
    "DVLONG": [777, 888, 998, 999],
    "ENERGY": [777, 888, 998, 999],
    "IMPACTSP": [777, 888, 998, 999],
    "CURBWGT": [777, 888, 998, 999],
    "ROLLDIST": [777, 888, 998, 999],
    "TRAVELSP": [777, 888, 998, 999],
    "SPLIMIT": [777, 888, 998, 999],
    "PDOF1": [998, 999],
    "PDOF2": [998, 999]
}


for column, special_values in continuous_special_codes.items():

    if column in df.columns:

        df[column] = df[column].replace(
            special_values,
            np.nan
        )


# ============================================================
# CREATE DERIVED CRASH FEATURES
# ============================================================

# Absolute Delta-V components

if "DVLAT" in df.columns:
    df["ABS_DVLAT"] = df["DVLAT"].abs()

if "DVLONG" in df.columns:
    df["ABS_DVLONG"] = df["DVLONG"].abs()


# Delta-V magnitude from components

if "DVLAT" in df.columns and "DVLONG" in df.columns:

    df["DV_COMPONENT_MAG"] = np.sqrt(
        df["DVLAT"] ** 2 +
        df["DVLONG"] ** 2
    )


# ============================================================
# CREATE TARGET CODE
# ============================================================

severity_mapping = {
    "MINOR": 0,
    "SERIOUS": 1,
    "CRITICAL": 2
}

df["SEVERITY_CODE"] = df["SEVERITY"].map(
    severity_mapping
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

identifier_columns = [
    "CASENO",
    "VEHNO",
    "CIRENID"
]

target_columns = [
    "MAIS",
    "ISS",
    "FATAL",
    "SEVERITY",
    "SEVERITY_CODE"
]


feature_columns = [
    column
    for column in df.columns
    if column not in identifier_columns
    and column not in target_columns
]


final_columns = (
    identifier_columns +
    feature_columns +
    target_columns
)


df = df[final_columns]


# ============================================================
# REMOVE COMPLETELY EMPTY FEATURES
# ============================================================

empty_features = []

for column in feature_columns:

    if df[column].notna().sum() == 0:

        empty_features.append(column)


if empty_features:

    print("\nRemoving completely empty features:")

    for column in empty_features:
        print("  -", column)

    df = df.drop(
        columns=empty_features
    )


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("CIREN DATASET CREATED")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nRows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


print("\nSeverity distribution:")
print(
    df["SEVERITY"]
    .value_counts()
    .sort_index()
)

print("\nSeverity percentage:")
print(
    (
        df["SEVERITY"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print("\nMAIS distribution:")
print(
    df["MAIS"]
    .value_counts()
    .sort_index()
)


print("\nMissing values in selected features:")

missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(
    missing.head(20).round(2)
)


print("\nFirst 5 rows:")
print(
    df.head()
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)