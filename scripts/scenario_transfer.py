"""Energy disappearance versus capacity-limited shifting, conditional accounting scenarios.
Uses measured/inferred hourly energy and a repeated-day boundary (each zone-day repeats every 24 h).
Removed 18:00--22:00 energy is queued at 22:00, then allocated chronologically to the next
minimum-tariff non-evening hours within 24 h. No cross-zone transfer or recursive response.
Capacity proxy = pile count * training-period energy/duration power; not feeder capacity.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import _bootstrap
from src.data.dataset import prepare
from src.data.windows import WindowDataset
from src.utils.config import load_config

ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/scenarios/transfer');ap.add_argument('overrides',nargs='*');a=ap.parse_args()
cfg=load_config(_bootstrap.DEFAULT_CFG,a.overrides);P=prepare(cfg);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
t=WindowDataset(P,'test',L=int(cfg.data.L),H=int(cfg.data.H),stride=int(cfg.train.eval_stride)).origins+24
time=P.time[t];counts=pd.Series(1,index=time.normalize()).groupby(level=0).sum();t=t[np.isin(time.normalize(),counts.index[counts==24])]
assert np.array_equal(P.time[t].hour,np.tile(np.arange(24),len(t)//24))
mask=P.tou_mask(); nz=int(mask.sum()); nd=len(t)//24
E=P.raw['volume'][t][:,mask].astype(float).reshape(nd,24,nz)/1000
price=np.exp(P.lp[t][:,mask]).reshape(nd,24,nz)
cap=P.capacity[mask]*P.power_kw[mask]/1000
assert np.isfinite(E).all() and E.min()>=0
# Existing observations above the capacity proxy cannot absorb further energy.
headroom=np.maximum(cap[None,None,:]-E,0)
B=json.loads(Path('configs/anchor/fivemin_main.json').read_text())['own'][0]['beta']
betas={'anchor':B,**{f'arrival_mu{mu}':B/(1-mu/180*(1-np.exp(-180/mu))) for mu in [30,60,120]},'adaptive':-.343,'naive_1.58':-1.58}
order=np.r_[22,23,np.arange(18)]; rows=[];profs=[];allocs=[]; checks=[]
fact=E.sum(axis=(0,2))/nd
for label,b in betas.items():
    removed=np.zeros_like(E);removed[:,18:22]=E[:,18:22]*(1-np.exp(b*np.log(1.2)))
    lost=E-removed;added=np.zeros_like(E);unallocated=np.zeros((nd,nz))
    for day in range(nd):
        for z in range(nz):
            energy=removed[day,:,z].sum(); low=price[day,order,z].min()
            eligible=order[np.isclose(price[day,order,z],low,rtol=1e-6,atol=1e-8)]
            for h in eligible:
                take=min(energy,headroom[day,h,z]);added[day,h,z]=take;energy-=take
                if take>0:allocs.append(dict(elasticity=label,date=str(P.time[t[day*24]].date()),zone=int(P.zones[mask][z]),hour=int(h),MWh=float(take)))
                if energy<1e-12:break
            unallocated[day,z]=max(0,energy)
    shifted=lost+added
    err=float(np.max(np.abs(removed.sum(axis=1)-added.sum(axis=1)-unallocated)))
    capacity_violation=float(np.maximum(added-headroom,0).max())
    assert err<1e-9 and capacity_violation<1e-9 and np.min(shifted)>=0
    for mode,cf in [('disappear',lost),('shift',shifted)]:
        profile=cf.sum(axis=(0,2))/nd;d=profile-fact
        rows.append(dict(elasticity=label,beta=b,mode=mode,n_days=nd,n_zones=nz,removed_MWh_day=removed.sum()/nd,restored_MWh_day=added.sum()/nd if mode=='shift' else 0,unallocated_MWh_day=unallocated.sum()/nd if mode=='shift' else 0,net_MWh_day=d.sum(),evening_delta_MW=d[18:22].mean(),peak_MW=profile.max(),peak_hour=int(profile.argmax()),peak_delta_MW=profile.max()-fact.max(),fact_peak_MW=fact.max(),fact_peak_hour=int(fact.argmax())))
        profs.extend(dict(elasticity=label,mode=mode,hour=h,fact_MW=fact[h],cf_MW=profile[h],delta_MW=d[h]) for h in range(24))
    checks.append(dict(elasticity=label,conservation_error_MWh=err,capacity_violation_MW=capacity_violation,unallocated_MWh=float(unallocated.sum())))
pd.DataFrame(rows).to_csv(out/'transfer_summary.csv',index=False);pd.DataFrame(profs).to_csv(out/'transfer_profiles.csv',index=False);pd.DataFrame(allocs).to_csv(out/'transfer_allocations.csv',index=False)
meta={'rule':__doc__,'boundary':'Each observed zone-day is repeated periodically; hours 0--17 receive charging from the previous identical evening. Not a chronological simulation across different dates.', 'capacity_proxy_MW_total':float(cap.sum()),'baseline_hours_above_proxy':int((E>cap[None,None,:]+1e-9).sum()),'baseline_max_excess_MW':float(np.maximum(E-cap[None,None,:],0).max()),'n_days':nd,'n_zones':nz,'test_dates':[str(P.time[t[0]].date()),str(P.time[t[-1]].date())],'checks':checks,'interpretation':'Conditional point scenarios, not confidence bounds, observed behavioural redistribution, station-access feasibility or grid dispatch.'}
(out/'transfer_meta.json').write_text(json.dumps(meta,indent=2));print(pd.DataFrame(rows).round(5).to_string(index=False));print(json.dumps(meta,indent=2))
