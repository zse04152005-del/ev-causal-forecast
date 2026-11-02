"""Wait for a complete 999-rep bootstrap, then refresh dependent scenario and figure outputs."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--runs',required=True);args=ap.parse_args()
root=Path(__file__).resolve().parents[1];os.chdir(root)
while True:
 s=json.loads(Path('results/causal/round2/summary.json').read_text())
 if s.get('bootstrap',{}).get('D_exp',{}).get('reps')==999:break
 time.sleep(10)
b=s['D_exp']['beta'];se=s['bootstrap']['D_exp']['se'];g=1-(60/180)*(1-np.exp(-180/60))
z=pd.read_csv('results/causal/round2/bootstrap_D.csv');assert len(z)==999 and z.rep.nunique()==999 and np.isfinite(z[['D_exp','D_noexp']]).all().all()
raw=str(Path(args.data_dir).resolve());override='paths.data_dir='+raw
commands=[['scripts/make_paper_tables.py',override],['scripts/scenario_energy.py','--out','results/scenarios/energy',override]]
for dest,beta,sd in [('results/scenarios',b,se),('results/scenarios/steady_state_mu60',b/g,se/g),('results/scenarios/adaptive',-.343,se)]:
 commands.append(['scripts/simulate_scenarios.py','--runs',args.runs,'--out',dest,'--beta',str(beta),'--se',str(sd),override])
for cmd in commands:
 print('RUN',cmd,flush=True);subprocess.run([sys.executable,*cmd],check=True)
seed=pd.read_csv('results/comparisons/backbones/backbones_seeds.csv');seed=seed[seed.group.isin(['r2_proj0','r2_stid_learn0','r2_lstm_learn0'])].copy();assert len(seed)==30
seed['anchor_beta']=b;seed['anchor_se_boot999']=se;seed['e_hat']=np.maximum((seed.beta_learned-b)**2-se**2,0);seed['w_hat']=seed.e_hat/(seed.e_hat+se**2);seed['beta_adaptive']=np.minimum(seed.w_hat*min(b,0)+(1-seed.w_hat)*seed.beta_learned,0)
seed.to_csv('results/comparisons/backbones/adaptive_boot999.csv',index=False)
assert seed.w_hat.max()==0,'Adaptive conclusions must be revised'
# Direct propagation of untruncated bootstrap: separate from the normal/sign-constrained scenarios.
em=[]
for label,fac in [('anchor',1),('arrival_mu60',g)]:
 effect=100*np.expm1(z.D_exp.to_numpy()/fac*np.log(1.2));em.append({'elasticity':label,'q025_pct':float(np.quantile(effect,.025)),'q975_pct':float(np.quantile(effect,.975)),'q05_pct':float(np.quantile(effect,.05)),'q95_pct':float(np.quantile(effect,.95))})
Path('results/scenarios/energy/empirical_bootstrap_evening.json').write_text(json.dumps(em,indent=2))
subprocess.run([sys.executable,'figures/scripts/fig_results.py','results/comparisons','figures/results','results/scenarios','results/EDA','results/causal/5min',raw],check=True)
Path('results/causal/round3/dependents_complete.json').write_text(json.dumps({'reps':999,'beta':b,'se':se,'bootstrap_sha256':hashlib.sha256(Path('results/causal/round2/bootstrap_D.csv').read_bytes()).hexdigest(),'adaptive_max_weight':float(seed.w_hat.max()),'commands':commands},indent=2))
print('ALL DEPENDENTS COMPLETE',flush=True)
