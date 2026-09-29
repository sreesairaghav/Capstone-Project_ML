import pandas as pd
import os

BASE = "data/raw/ciren"

files = [
    "case.sas7bdat",
    "event.sas7bdat",
    "acc_desc.sas7bdat",
    "injury.sas7bdat",
    "injuryranalysis.sas7bdat",
    "scenario.sas7bdat",
    "ve.sas7bdat",
    "vi.sas7bdat",
    "gv.sas7bdat",
]


for filename in files:

    path = os.path.join(BASE, filename)

    print("\n" + "=" * 90)
    print(filename)
    print("=" * 90)

    if not os.path.exists(path):
        print("FILE NOT FOUND")
        continue

    try:

        df = pd.read_sas(path)

        print("ROWS    :", df.shape[0])
        print("COLUMNS :", df.shape[1])

        print("\nCOLUMN NAMES:\n")

        for i, column in enumerate(df.columns, start=1):
            print(f"{i:3}. {column}")

    except Exception as e:

        print("ERROR:", e)