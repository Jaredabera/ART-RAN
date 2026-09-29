from pathlib import Path
import json
import platform

import joblib
import pandas as pd
import sklearn

from art_ran_commag.twin import COMMAGResponseModel


PREP = Path("prepared_v062")

DATA = PREP / "commag_wide.parquet"
OUT = PREP / "commag_twin_sklearn152.joblib"
VAL_OUT = PREP / "twin_validation_sklearn152.csv"
TEST_OUT = PREP / "twin_test_sklearn152.csv"
META_OUT = PREP / "twin_metadata_sklearn152.json"


print("[1/6] Loading COMMAG dataframe...")
df = pd.read_parquet(DATA)

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

print(
    f"train={len(train):,}, "
    f"val={len(val):,}, "
    f"test={len(test):,}"
)

print("[2/6] Training ExtraTrees empirical twin...")

twin = COMMAGResponseModel(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=2027,
    n_jobs=8,
)

twin.fit(train)

print("[3/6] Validation fidelity...")
val_score = twin.score(val)
print(val_score.to_string(index=False))

print("[4/6] Test fidelity...")
test_score = twin.score(test)
print(test_score.to_string(index=False))

print("[5/6] Saving rebuilt twin...")
twin.save(OUT)

val_score.to_csv(
    VAL_OUT,
    index=False,
)

test_score.to_csv(
    TEST_OUT,
    index=False,
)

metadata = {
    "python": platform.python_version(),
    "scikit_learn": sklearn.__version__,
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "max_features": 0.85,
    "random_state": 2027,
    "n_jobs": 8,
    "train_rows": int(len(train)),
    "val_rows": int(len(val)),
    "test_rows": int(len(test)),
}

META_OUT.write_text(
    json.dumps(metadata, indent=2)
)

print("[6/6] Reload test...")

reloaded = COMMAGResponseModel.load(OUT)

row = val.iloc[0]

buffer_state = [
    row["dl_buffer [bytes]_s0"],
    row["dl_buffer [bytes]_s1"],
    row["dl_buffer [bytes]_s2"],
]

actions = pd.read_csv(
    PREP / "action_catalog.csv"
).to_numpy()

tput, ratio = reloaded.predict_from_row(
    row,
    buffer_state,
    actions[0],
)

print("Reloaded twin: OK")
print("Example action:", actions[0])
print("Throughput:", tput)
print("Grant ratio:", ratio)
print("Saved:", OUT)
