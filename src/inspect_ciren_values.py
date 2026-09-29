import pandas as pd
import os

BASE = "data/raw/ciren"

CHECKS = {
    "case.sas7bdat": [
        "FATAL",
        "MAIS",
        "ISS"
    ],

    "gv.sas7bdat": [
        "DVTOTAL",
        "DVEST",
        "DVBASIS",
        "DVCONFID",
        "DVLAT",
        "DVLONG",
        "ENERGY",
        "IMPACTSP",
        "TRAVELSP",
        "MANEUVER",
        "PREMOVE",
        "ROLLOVER",
        "ROLLDIST"
    ],

    "ve.sas7bdat": [
        "DVC1",
        "DVC2",
        "DVC3",
        "DVC4",
        "DVC5",
        "DVC6",
        "DVD",
        "DVL",
        "EXTENT1",
        "EXTENT2",
        "PDOF1",
        "PDOF2",
        "FIRE"
    ]
}


for filename, columns in CHECKS.items():

    path = os.path.join(BASE, filename)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    df = pd.read_sas(path)

    for col in columns:

        if col not in df.columns:
            print(f"\n{col}: NOT FOUND")
            continue

        series = df[col]

        print(f"\n{col}")
        print("-" * 40)

        print("Non-null :", series.notna().sum())
        print("Missing  :", series.isna().sum())
        print("Unique   :", series.nunique())

        values = series.value_counts(dropna=False).head(15)

        print("Top values:")
        print(values.to_dict())