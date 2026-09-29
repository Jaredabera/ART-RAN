from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

def feature_columns():
    cols=[]
    for s in range(3):
        cols += [
            f"log_buffer_s{s}", f"sum_requested_prbs_s{s}",
            f"dl_cqi_s{s}", f"dl_mcs_s{s}",
            f"tx_errors downlink (%)_s{s}",
            f"scheduling_policy_s{s}", f"action_s{s}"
        ]
    return cols

def target_columns():
    cols=[]
    for s in range(3):
        cols += [
            f"tx_brate downlink [Mbps]_s{s}",
            f"ratio_granted_req_s{s}"
        ]
    return cols

def make_xy(df: pd.DataFrame):
    x=df.copy()
    for s in range(3):
        x[f"log_buffer_s{s}"]=np.log1p(x[f"dl_buffer [bytes]_s{s}"].clip(lower=0))
        x[f"action_s{s}"]=x[f"slice_prb_s{s}"]
    X=x[feature_columns()].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
    Y=x[target_columns()].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
    return X,Y

class COMMAGResponseModel:
    """Predict action-sensitive immediate KPI response only.

    Queue evolution is handled by a trace-driven physical balance equation in
    COMMAGPolicyEnv. This prevents compounding next-buffer model error.
    """
    def __init__(self, n_estimators=300, min_samples_leaf=3, random_state=2027, n_jobs=-1):
        self.model=ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=n_jobs, max_features=.85
        )
    def fit(self,df):
        X,Y=make_xy(df); self.model.fit(X,Y); return self
    def predict_from_row(self,row,buffer_state,action):
        vals=[]
        for s in range(3):
            vals += [
                np.log1p(max(float(buffer_state[s]),0.0)),
                float(row[f"sum_requested_prbs_s{s}"]),
                float(row[f"dl_cqi_s{s}"]) if pd.notna(row[f"dl_cqi_s{s}"]) else 0.0,
                float(row[f"dl_mcs_s{s}"]) if pd.notna(row[f"dl_mcs_s{s}"]) else 0.0,
                float(row[f"tx_errors downlink (%)_s{s}"]) if pd.notna(row[f"tx_errors downlink (%)_s{s}"]) else 0.0,
                float(row[f"scheduling_policy_s{s}"]) if pd.notna(row[f"scheduling_policy_s{s}"]) else 0.0,
                float(action[s])
            ]
        y=self.model.predict(np.asarray(vals,float)[None,:])[0]
        throughput=np.asarray([y[0],y[2],y[4]],float).clip(0,None)
        ratio=np.asarray([y[1],y[3],y[5]],float).clip(0,1)
        return throughput,ratio
    def score(self,df):
        X,Y=make_xy(df); P=self.model.predict(X)
        names=target_columns(); rows=[]
        for i,n in enumerate(names):
            rows.append({
                "target":n,
                "MAE":float(mean_absolute_error(Y[:,i],P[:,i])),
                "R2":float(r2_score(Y[:,i],P[:,i])),
                "std":float(np.std(Y[:,i]))
            })
        return pd.DataFrame(rows)
    def save(self, path):
        joblib.dump(
            self.model,
            path,
            compress=3,
        )
    @classmethod
    def load(cls,path):
        payload=joblib.load(path)
        depth=0
        while isinstance(payload,cls):
            payload=payload.model; depth+=1
            if depth>16: raise RuntimeError("Excessive nested COMMAGResponseModel wrappers")
        if not hasattr(payload,"predict"):
            raise TypeError(f"Unsupported twin payload: {type(payload)!r}")
        obj=cls(); obj.model=payload; return obj
