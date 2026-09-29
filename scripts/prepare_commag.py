from pathlib import Path
import argparse, json
from art_ran_commag.data import build_commag_wide, add_splits_and_labels, build_transitions, observed_action_catalog
from art_ran_commag.calibration import calibrate

p=argparse.ArgumentParser()
p.add_argument("--data-root",required=True)
p.add_argument("--out",default="prepared/commag_wide.parquet")
p.add_argument("--min-action-count",type=int,default=100)
a=p.parse_args()

df=build_commag_wide(a.data_root,a.out)
df,thr=add_splits_and_labels(df)
df=build_transitions(df)
df.to_parquet(a.out,index=False)

train=df[df.split=="train"]
actions,counts=observed_action_catalog(train,a.min_action_count)
# cal=calibrate(train)
cal=calibrate(train, action_catalog=actions)

Path(a.out).with_name("action_catalog.csv").write_text(
    "a0,a1,a2\n"+"\n".join(",".join(map(str,x)) for x in actions)
)
counts.to_csv(Path(a.out).with_name("action_counts.csv"),index=False)
Path(a.out).with_name("calibration.json").write_text(json.dumps({**cal,**thr},indent=2))
print(df.groupby(["split","scenario"]).size())
print("actions:",actions.tolist())
