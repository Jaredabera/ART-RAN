from pathlib import Path
import argparse
import pandas as pd
from art_ran_commag.twin import COMMAGResponseModel

p=argparse.ArgumentParser()
p.add_argument("--parquet",required=True)
p.add_argument("--out",default="prepared/commag_twin.joblib")
a=p.parse_args()

df=pd.read_parquet(a.parquet)
train=df[df.split=="train"]; val=df[df.split=="val"]; test=df[df.split=="test"]
m=COMMAGResponseModel().fit(train)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
m.save(a.out)
print("VALIDATION")
print(m.score(val).round(4).to_string(index=False))
print("\nTEST")
print(m.score(test).round(4).to_string(index=False))
