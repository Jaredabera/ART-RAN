import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from art_ran_commag.twin import COMMAGResponseModel
from art_ran_commag.calibration import utilities

p=argparse.ArgumentParser()
p.add_argument("--parquet",required=True); p.add_argument("--twin",required=True)
p.add_argument("--actions",required=True); p.add_argument("--calibration",required=True)
p.add_argument("--n",type=int,default=1000)
a=p.parse_args()
df=pd.read_parquet(a.parquet)
val=df[df.split=="val"]
if len(val)>a.n: val=val.sample(a.n,random_state=2027)
twin=COMMAGResponseModel.load(a.twin)
actions=pd.read_csv(a.actions).to_numpy(int)
cal=json.loads(Path(a.calibration).read_text())
targets=np.asarray([cal[f"qos_target_s{s}"] for s in range(3)])
rows=[]
for _,r in val.iterrows():
    buffer=np.asarray([r[f"dl_buffer [bytes]_s{s}"] for s in range(3)],float)
    best=np.zeros(3); feasible=False
    dt=float(r.get("dt_sec",.25)); arrivals=np.asarray([r[f"arrival_bytes_s{s}"] for s in range(3)],float)
    for act in actions:
        tp,ra=twin.predict_from_row(r,buffer,act)
        service=tp*1e6/8.0*dt
        nb=np.maximum(buffer-service,0.0)+np.maximum(arrivals,0.0)
        u=utilities(tp,ra,nb,cal)
        best=np.maximum(best,u)
        feasible |= bool(np.all(u>=targets))
    rows.append((r["scenario"],*best,feasible))
out=pd.DataFrame(rows,columns=["scenario","best_embb","best_mmtc","best_urllc","joint_qos_feasible"])
print("QoS targets:",targets.round(4).tolist())
print(out.groupby("scenario").agg({
    "best_embb":"mean","best_mmtc":"mean","best_urllc":"mean","joint_qos_feasible":"mean"
}).round(3))
