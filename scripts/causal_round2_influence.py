"""Which zones drive the left tail of the zone bootstrap of design D (results/causal/round2/bootstrap_D.csv)?

Rebuilds the zone draws of each bootstrap replication from its seed (seed, 0, rep) as in causal_round2.py, correlates the
number of times each zone is drawn with the estimate, and re-estimates design D (with exposure terms) without each of
the three most influential zones. Output: results/causal/round2/bootstrap_influence.json.
Usage: python scripts/causal_round2_influence.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.utils.config import load_config
from src.data.dataset import prepare
from src.causal import switch_5min as s5
from src.utils.paths import project_root
cfg=load_config(_bootstrap.DEFAULT_CFG); root=project_root(cfg.paths.get('root')); P=prepare(cfg)
fm=s5.load_fivemin(os.path.join(root,'data','interim','fiveMin'))
zpos={int(z):i for i,z in enumerate(P.zones)}
S_zone=s5.zone_ring_exposure(fm.zone_P,P.ring_members,P.ref_lp)
inv=s5.invalid_mask(fm,P.frozen,np.array([zpos[int(z)] for z in fm.st_zone]))
T_HI=(P.split.train_end+1)*12-1; gz={int(z):int(g) for z,g in zip(P.zones,P.groups)}; cc=cfg.causal
spec=s5.Spec5(window=36,jump_threshold=cc.jump_threshold,exclude_holiday_pm1=cc.exclude_holiday_pm1)
pn,_=s5.build_panel_5min(fm,gz,zpos,S_zone,inv,T_HI,spec,cfg.data.context.day_start,cfg.data.context.night_start)
pn=pn[pn.tou].reset_index(drop=True); ZONES=np.unique(pn.zone)
out=os.path.join(root,'results','causal','round2'); bt=pd.read_csv(os.path.join(out,'bootstrap_D.csv'))
C=np.zeros((len(bt),len(ZONES)))
for r in bt.rep.astype(int):
    rng=np.random.default_rng([20261013,0,r]); d=rng.choice(ZONES,size=len(ZONES),replace=True)
    for z in d: C[r,np.searchsorted(ZONES,z)]+=1
y=bt.D_exp.values
corr=[np.corrcoef(C[:,j],y)[0,1] for j in range(len(ZONES))]
o=np.argsort(corr)[:5]
ev=pn[pn.x!=0].groupby('zone').size()
res={'events_total':int((pn.x!=0).sum()),'zones':[]}
for j in o: print('zone',ZONES[j],'corr',round(corr[j],2),'events',int(ev.get(ZONES[j],0)))
tail=y<np.quantile(y,0.1); print('mean count of top zone in tail vs rest', C[tail,o[0]].mean().round(2), C[~tail,o[0]].mean().round(2))
# leave-one-zone-out estimate for the top zones
for j in o[:3]:
    d=pn[pn.zone!=ZONES[j]]; r=s5.ppml_pair(d,['x','S1','S2','S3'],['dh','zh']); print('drop',ZONES[j],round(r['coef']['x'],3),round(r['se']['x'],3))
    res['zones'].append({'zone':int(ZONES[j]),'corr_count_beta':float(corr[j]),'events':int(ev.get(ZONES[j],0)),'beta_without':float(r['coef']['x']),'se_without':float(r['se']['x'])})
json.dump(res,open(os.path.join(out,'bootstrap_influence.json'),'w'),indent=1)
