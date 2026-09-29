import argparse
from pathlib import Path
import pandas as pd,matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument("--csv",required=True);p.add_argument("--out-dir",default=None);a=p.parse_args()
d=pd.read_csv(a.csv);out=Path(a.out_dir) if a.out_dir else Path(a.csv).parent/"figures";out.mkdir(parents=True,exist_ok=True)
plots=[("oracle_score_mean","oracle_score_se","Oracle-normalized action quality","oracle_score",100),
       ("regret_mean","regret_se","Immediate reward regret","regret",1),
       ("fcsr_mean","fcsr_se","Feasibility-conditioned QoS success (%)","fcsr",100),
       ("urllc_violation_mean","urllc_violation_se","URLLC empirical-QoS violation (%)","urllc_violation",100)]
for metric,se,ylabel,name,scale in plots:
    plt.figure(figsize=(7,4.2))
    for s,g in d.groupby("scenario"):
        g=g.sort_values("timesteps");x=g.timesteps/1000;y=scale*g[metric];ci=1.96*scale*g[se]
        plt.plot(x,y,marker="o",label=s.replace("_"," "));plt.fill_between(x,y-ci,y+ci,alpha=.12)
    plt.xlabel("Training steps (×10³)");plt.ylabel(ylabel);plt.grid(True,linestyle=":",linewidth=.6);plt.legend(ncol=2,frameon=False,fontsize=8);plt.tight_layout();plt.savefig(out/f"learning_{name}.pdf",bbox_inches="tight");plt.savefig(out/f"learning_{name}.png",dpi=400,bbox_inches="tight");plt.close()
print("Saved",out)
