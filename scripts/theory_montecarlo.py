"""Monte Carlo checks of Theorems 1 and 2 (Supplementary Material, Tables S2 and S3).

Panel: Z zones x days x 24 h. log y = g0(zone, hour) + a_day + beta0 * D + eps.
D = A * s_z(hour) + V, where s_z is a fixed TOU schedule (+1 in high-price hours, -1 otherwise) and V is the
residual (non-schedule) price variation, with corr(V, eps) = r.  The forecaster class G is either
  'full'     : zone x hour + day effects (contains g0 and the schedule)        -> only the confounding term
  'additive' : zone + hour + day effects (misses the zone-specific profile)   -> adds the approximation term
The loss-based elasticity is the least-squares coefficient of D given G (squared log loss).
Checks: (i) mean of beta_L against beta0 + <MD, M(eps+g0)> / ||MD||^2 computed on a large draw;
        (ii) sd of beta_L against sigma_u / (sqrt(n) sd(MD)) with u = M eps - b MD;
        (iii) Theorem 2: empirical MSE of w*beta_A + (1-w)*beta_L over a grid of w against the formula and w*.

Usage (repository root): python scripts/theory_montecarlo.py [results/theory/mc_t1_t2.json]   (CPU, a few minutes)
"""
import json
import os
import sys
from statistics import NormalDist

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                                                          "theory", "mc_t1_t2.json")
rng = np.random.default_rng(20261013)
Z, DAYS, H = 30, 60, 24
beta0, A = -0.5, 0.08


def make(sd_v, r, align, cls, rng, struct_seed=1):
    srng = np.random.default_rng(struct_seed)                      # zone structure fixed across replications
    hours = np.arange(H)
    peak = srng.integers(0, H, Z)                                      # zone-specific demand peak
    prof = np.cos(2 * np.pi * (hours[None, :] - peak[:, None]) / 24)  # [Z, H] zone x hour profile
    g0 = 0.6 * prof
    # schedule: high price in the 10 hours ranked by score = align * profile + (1 - |align|) * noise
    sched = np.empty((Z, H))
    for z in range(Z):
        score = align * prof[z] + (1 - abs(align)) * srng.standard_normal(H)
        hi = np.argsort(-score)[:10]
        s = -np.ones(H); s[hi] = 1; sched[z] = s
    aday = 0.2 * srng.standard_normal(DAYS)
    e = rng.standard_normal((DAYS, Z, H))
    v = r * e + np.sqrt(1 - r ** 2) * rng.standard_normal((DAYS, Z, H))
    eps = 0.3 * e; V = sd_v * v
    D = A * sched[None] + V
    Y = g0[None] + aday[:, None, None] + beta0 * D + eps
    return Y, D, eps, g0, A * np.broadcast_to(sched, D.shape).copy()


def resid(X, cls):
    """M_G X for X [DAYS, Z, H] by alternating projections (exact for these crossed designs)."""
    R = X - X.mean()
    for _ in range(50):
        if cls == "full":
            R = R - R.mean(axis=0, keepdims=True)                   # zone x hour
        else:
            R = R - R.mean(axis=(0, 2), keepdims=True)              # zone
            R = R - R.mean(axis=(0, 1), keepdims=True)              # hour
        R = R - R.mean(axis=(1, 2), keepdims=True)                  # day
    return R


def beta_L(Y, D, cls):
    MD = resid(D, cls); MY = resid(Y, cls)
    return (MD * MY).sum() / (MD * MD).sum(), MD


out = {"T1": [], "T2": {}}
for cls in ("full", "additive"):
    for align in (0.0, 1.0):
        for r in (0.0, 0.1):
            for sd_v in (0.005, 0.02):
                est = []
                for rep in range(200):
                    Y, D, eps, g0, S = make(sd_v, r, align, cls, np.random.default_rng(rep))
                    b, MD = beta_L(Y, D, cls)
                    est.append(b)
                # population prediction (Theorem 1): only the noise is random, the schedule S and g0 are fixed
                Y, D, eps, g0, S = make(sd_v, r, align, cls, np.random.default_rng(0))
                n = D.size
                k = Z * H + DAYS - 1 if cls == "full" else Z + H + DAYS - 2
                MS = resid(S, cls); Mg = resid(np.broadcast_to(g0, D.shape).copy(), cls)
                EMD2 = (MS * MS).sum() + n * sd_v ** 2 * (1 - k / n)      # E||MD||^2
                conf = n * r * 0.3 * sd_v * (1 - k / n) / EMD2             # E<MD, M eps> / E||MD||^2
                appr = (MS * Mg).sum() / EMD2
                sd_pred = 0.3 * np.sqrt(1 - r ** 2) / np.sqrt(EMD2)
                est = np.array(est)
                out["T1"].append(dict(G=cls, align=align, r=r, sd_v=sd_v, R2=float(1 - resid(D, cls).var() / D.var()),
                                      mean_beta_L=float(est.mean()), sd_beta_L=float(est.std()),
                                      predicted_mean=float(beta0 + conf + appr), conf_term=float(conf), predicted_sd=float(sd_pred),
                                      approx_term=float(appr)))
# Theorem 2: learned estimators from three settings, anchor ~ N(beta0, s_A^2) independent of the learned estimate
cells = {"unconfounded_scarce": dict(sd_v=0.005, r=0.0, align=1.0, cls="full"),
         "mildly_confounded_scarce": dict(sd_v=0.005, r=0.01, align=1.0, cls="full"),
         "misspecified_aligned": dict(sd_v=0.005, r=0.0, align=1.0, cls="additive")}
for cname, cell in cells.items():
    bl = np.array([beta_L(*make(cell["sd_v"], cell["r"], cell["align"], cell["cls"], np.random.default_rng(rep + 1000))[:2], cell["cls"])[0]
                   for rep in range(400)])
    b_L, v_L = bl.mean() - beta0, bl.var()
    for s_A in (0.1, 0.3, 0.6):
        q = np.array([NormalDist().inv_cdf((k + 0.5) / 400) for k in range(400)])   # exact N(0,1) quantiles
        ba = beta0 + s_A * np.random.default_rng(7).permutation(q / q.std())
        ws = np.linspace(0, 1, 101)
        mse = np.array([np.mean((w * ba + (1 - w) * bl - beta0) ** 2) for w in ws])
        w_star = (b_L ** 2 + v_L) / (b_L ** 2 + v_L + s_A ** 2)
        mse_star = (b_L ** 2 + v_L) * s_A ** 2 / (b_L ** 2 + v_L + s_A ** 2)
        out["T2"][f"{cname}|s_A={s_A}"] = dict(b_L=float(b_L), v_L=float(v_L), w_star_formula=float(w_star),
                               w_star_empirical=float(ws[mse.argmin()]), mse_at_w_star_formula=float(mse_star),
                               mse_min_empirical=float(mse.min()), mse_anchor=float(mse[-1]), mse_anchor_formula=float(s_A ** 2),
                               mse_learned=float(mse[0]), mse_learned_formula=float(b_L ** 2 + v_L), mse_price_blind=float(beta0 ** 2))
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
for row in out["T1"]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
for k, v in out["T2"].items():
    print("s_A", k, {kk: round(vv, 4) for kk, vv in v.items()})
