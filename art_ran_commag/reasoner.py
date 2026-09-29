from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RAppPolicy:
    priority: np.ndarray
    min_alloc: np.ndarray
    max_change: int
    def vector(self,budget):
        return np.concatenate([
            self.priority.astype(np.float32),
            (self.min_alloc/max(float(budget),1.0)).astype(np.float32),
            np.asarray([self.max_change/max(float(budget),1.0)],np.float32)
        ])

class ReferenceReasoner:
    """Auditable clean Non-RT reasoner; replaceable by an LLM later."""
    def __init__(self,action_catalog,cal):
        self.actions=np.asarray(action_catalog,int); self.cal=cal
        self.floor=self.actions.min(axis=0)
    def propose(self,buffer_state,row,current_u):
        targets=np.asarray([self.cal[f"qos_target_s{s}"] for s in range(3)])
        deficits=np.maximum(targets-current_u,0)/np.maximum(targets,1e-6)
        req=np.asarray([row[f"sum_requested_prbs_s{s}"] for s in range(3)],float)
        req=req/np.maximum(np.asarray([self.cal[f"req_ref_s{s}"] for s in range(3)]),1e-6)
        buf=np.asarray(buffer_state,float)/np.maximum(np.asarray([self.cal[f"buffer_ref_s{s}"] for s in range(3)]),1e-6)
        critical=np.asarray([1.0,0.8,1.35])
        score=.45*critical + 1.15*deficits + .25*np.clip(req,0,2) + .20*np.tanh(buf)
        pr=np.clip(score,.05,None); pr=pr/pr.sum()
        mins=self.floor.copy()
        j=int(np.argmax(deficits+.20*req))
        candidate=mins.copy(); candidate[j]+=1
        if np.any(np.all(self.actions>=candidate[None,:],axis=1)):
            mins=candidate
        return RAppPolicy(pr.astype(np.float32),mins.astype(np.int32),self.cal["max_change_default"])
