"""Audited Delta^SD FLCI from saved event-study coefficients (no re-estimation).
Reference: HonestDiD/R/flci.R, https://github.com/asheshrambachan/HonestDiD.
Uses deterministic folded-normal quantiles and an epigraph for the L1 bias support.
Intervals target the no-exposure event-study lag average, not the main anchor.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize, stats, linalg


def fit_setup(t,b,V,base=-1):
    grid=np.arange(t.min(),t.max()+1); n=len(grid); bp=np.where(grid==base)[0][0]
    post=t>=0; pre=t<base; l=post.astype(float)/post.sum()
    g=t[pre]-base; h=l@(t-base); w0=h*g/(g@g); Q=linalg.null_space(g[None,:])
    C=np.zeros((len(t),Q.shape[1])); C[pre]=-Q
    c0=l.copy(); c0[pre]=-w0
    D=np.diff(np.eye(n),n=2,axis=0)
    def full(c):
        cf=np.zeros(n); cf[np.searchsorted(grid,t)]=c; cf[bp]=-c.sum()
        return cf
    T=np.linalg.solve(D@D.T,D)
    a0=T@full(c0); A=np.column_stack([T@full(c) for c in C.T])
    # Check the support function against the original constrained linear programme.
    errs=[]
    for z in np.random.default_rng(17).normal(size=(12,Q.shape[1])):
        cf=full(c0+C@z)
        eq=np.eye(n)[bp:bp+1]
        r=optimize.linprog(-cf,A_ub=np.r_[D,-D],b_ub=np.ones(2*(n-2)),A_eq=eq,b_eq=[0],bounds=[(None,None)]*n,method='highs')
        assert r.success
        errs.append(abs(-r.fun-np.abs(a0+A@z).sum()))
        assert abs(cf.sum())<1e-10 and abs(cf@grid)<1e-9
    assert max(errs)<1e-7
    zmin=-np.linalg.solve(C.T@V@C,C.T@V@c0)
    return c0,C,a0,A,zmin,max(errs)


def cv(x):
    return optimize.brentq(lambda q:stats.norm.cdf(q-x)-stats.norm.cdf(-q-x)-.95,0,x+10,xtol=1e-12)


def calculate(t,b,V,Ms):
    c0,C,a0,A,zmin,err=fit_setup(t,b,V); nz=C.shape[1]; nu=len(a0)
    mat=np.r_[np.c_[-A,np.eye(nu)],np.c_[A,np.eye(nu)]]
    constraint=optimize.LinearConstraint(mat,np.r_[a0,-a0],np.inf)
    prev=zmin; rows=[]; weights=[]
    for M in Ms:
        def obj(x):
            c=c0+C@x[:nz]; sd=np.sqrt(c@V@c); bias=M*x[nz:].sum()
            return sd*cv(max(0,bias)/sd)
        if M==0:
            z=zmin; success=True; disagreement=0.
        else:
            fits=[]
            for z0 in (prev,zmin,np.zeros(nz)):
                x0=np.r_[z0,np.abs(a0+A@z0)+1e-10]
                r=optimize.minimize(obj,x0,method='SLSQP',constraints=[constraint],options={'ftol':1e-11,'maxiter':1500})
                violation=max(0,float(np.max(constraint.lb-mat@r.x)))
                if r.success and violation<1e-7: fits.append(r)
            if not fits: raise RuntimeError(f'No converged solution at M={M}')
            best=min(fits,key=lambda r:r.fun); z=best.x[:nz]; success=best.success
            disagreement=max(r.fun for r in fits)-best.fun
            assert disagreement<1e-4,(M,disagreement)
        prev=z; c=c0+C@z; sd=float(np.sqrt(c@V@c)); bias=float(M*np.abs(a0+A@z).sum()); half=sd*cv(bias/sd); th=float(c@b)
        rows.append(dict(M=float(M),theta_hat=th,half_length=half,lo=th-half,hi=th+half,sd=sd,bias_bound=bias,converged=bool(success),multistart_spread=disagreement))
        weights.append(c.tolist())
    return pd.DataFrame(rows),dict(support_lp_max_error=err,weights=weights)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',default='results/causal/round3');a=ap.parse_args(); p=Path(a.dir)
    df=pd.read_csv(p/'event_study_k1.csv'); d=df[(df.direction=='all')&(df.bin!=-1)].sort_values('k')
    t=d.bin.to_numpy(); b=d.beta.to_numpy();V=np.load(p/'event_vcov_all.npy')
    assert np.max(np.abs(d.se.to_numpy()**2-np.diag(V)))<1e-9,'Stale covariance'
    assert np.linalg.eigvalsh(V).min()>0
    Ms=[0,.001,.002,.005,.01,.015,.02,.025,.03,.04,.05,.075,.1,.15,.2]
    hs,qa=calculate(t,b,V,Ms);hs.to_csv(p/'honest_sd.csv',index=False)
    l=(t>=0).astype(float);l/=l.sum();th=float(l@b);se=float(np.sqrt(l@V@l))
    s=json.loads((p/'summary.json').read_text());s['honest']={'target':'No-exposure event-study mean of twelve post-switch 15-min coefficients; base bin -1. Different estimand from the whole-window anchor with exposures.', 'restriction':'DeltaSD: absolute second differences bounded by M; M=0 allows linear trends', 'theta_hat_parallel_trends':th,'se_parallel_trends':se,'ci_parallel_trends':[th-1.96*se,th+1.96*se], 'pooled':-1.58,'flci':hs.to_dict('records'),'first_grid_M_containing_pooled':next((float(r.M) for r in hs.itertuples() if r.lo<=-1.58<=r.hi),None),'first_grid_M_containing_zero':next((float(r.M) for r in hs.itertuples() if r.lo<=0<=r.hi),None),'lead_max_abs_second_difference':float(np.max(np.abs(np.diff(np.r_[b[t<-1],0],n=2)))),'validation':qa}
    (p/'summary.json').write_text(json.dumps(s,indent=2));print(hs.to_string(index=False));print('LP support error:',qa['support_lp_max_error'])
if __name__=='__main__':main()
