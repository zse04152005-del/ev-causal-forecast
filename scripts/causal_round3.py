"""Round-3 checks of the switch-event design (pre-submission review, 2026-10-28; items A4 and A5).

Stages (comma-separated with --stages; default all):
  event      event study of design D with the last 15 min before the switch (k = -1) as the base, for all events and by
             direction; the leads and lags get a joint zone-clustered covariance (stacked influence functions), and the
             ten leads are tested jointly (Wald, chi2 with 10 degrees of freedom)
  honest     Rambachan and Roth (2023) sensitivity of the average lag (0-180 min) of the event study for all events:
             fixed-length confidence intervals under the smoothness restriction Delta^SD(M) (second differences of the
             violation of parallel trends bounded by M per 15-min bin, linear trends through k = -1 allowed at M = 0),
             and the breakdown values of M at which the interval first contains the pooled slope (-1.58) or zero
  jackknife  design D (with and without exposure terms) leaving out each zone in turn
  offset     design D with the pseudo-count added to both window counts set to 0, 0.01, 0.1 (main) and 1 pile-hour
  variation  where the identifying variation comes from: the residual of the switch size after the fixed effects of
             design D, split into station-slot cells whose switch size stays at its usual value, departs from it for
             at least seven consecutive days (persistent departures; not verified operator reforms) or for fewer days (transient departures); and a
             balance test that regresses the switch size, with the same fixed effects, on the demand in the same
             pre-window one and seven days earlier and on the current pre-window demand
Outputs go to results/causal/round3/ (CSV and a JSON summary). The bootstrap with 999 replications is the boot stage of
causal_round2.py with --reps 999 (resumes from any replications already saved).
Usage: python scripts/causal_round3.py [--stages all] [--workers 4] [--data-dir data/interim/fiveMin]
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from scipy import optimize, stats

import _bootstrap  # noqa: F401
from src.causal import switch_5min as s5
from src.causal.switch_did import DESIGNS
from src.data.dataset import prepare
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--data-dir", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--stages", default="all")
ap.add_argument("--window", type=int, default=36)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--pooled", type=float, default=-1.58, help="pooled price-demand slope (Fig. 5c)")
ap.add_argument("--base-bin", type=int, default=1, help="event study: base = the n-th 15-min bin before the switch")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, list(a.overrides))
root = project_root(cfg.paths.get("root"))
out = ensure_dir(a.out or os.path.join(root, "results", "causal", "round3"))
stages = {"event", "honest", "jackknife", "offset", "variation"} if a.stages == "all" else set(a.stages.split(","))
t0 = time.time()
P = prepare(cfg)
fm = s5.load_fivemin(a.data_dir or os.path.join(root, "data", "interim", "fiveMin"))
zpos = {int(z): i for i, z in enumerate(P.zones)}
S_zone = s5.zone_ring_exposure(fm.zone_P, P.ring_members, P.ref_lp)
inv = s5.invalid_mask(fm, P.frozen, np.array([zpos[int(z)] for z in fm.st_zone]))
T_HI = (P.split.train_end + 1) * 12 - 1
gz = {int(z): int(g) for z, g in zip(P.zones, P.groups)}
cc = cfg.causal
spec = s5.Spec5(window=a.window, jump_threshold=cc.jump_threshold, exclude_holiday_pm1=cc.exclude_holiday_pm1)
pn, info = s5.build_panel_5min(fm, gz, zpos, S_zone, inv, T_HI, spec, cfg.data.context.day_start, cfg.data.context.night_start)
pn = pn[pn.tou].reset_index(drop=True)
FE_D = DESIGNS["D"]["fe"]
XS_E = ["x", "S1", "S2", "S3"]
SPECS = (("D_noexp", ["x"]), ("D_exp", XS_E))
_sp = os.path.join(out, "summary.json")
summary = json.load(open(_sp, encoding="utf-8")) if os.path.exists(_sp) else {}
summary.update({"panel_rows": int(len(pn)), "events": int((pn.x != 0).sum()), "zones": int(pn.zone.nunique())})
print(f"面板 {len(pn):,} 行，事件 {summary['events']:,}，{time.time() - t0:.0f}s", flush=True)


def save():
    json.dump(summary, open(_sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


def est(df, xs, pseudo=0.1):
    r = s5.ppml_pair(df, xs, FE_D, pseudo=pseudo)
    return r["coef"]["x"], r["se"]["x"], r["n"]


# ============================================================ event study, base k = -1
KS = list(range(-33, 36, 3))           # 15-min bins: k/3 = -11, ..., 11; bin -1 is [t-15 min, t)
WIDTH = 3
BASE = -WIDTH * a.base_bin
B0 = BASE // WIDTH
TAG = "" if a.base_bin == 1 else f"_base{a.base_bin}"   # outputs of other bases get a suffix
LEADS = [k for k in KS if k < BASE]
LAGS = [k for k in KS if k >= 0]


def event_study(pe):
    """Each bin against the base bin; returns the coefficients and their joint zone-clustered covariance."""
    ks = [k for k in KS if k != BASE]
    b, sums, labels = [], [], None
    for k in ks:
        d = pe.assign(Apost=pe[f"A{k}"], Apre=pe[f"A{BASE}"])
        r = s5.ppml_pair(d, ["x"], FE_D, pseudo=0.1 * WIDTH / a.window, return_if=True)
        lab, sc = r["if_by_cluster"]
        b.append(r["coef"]["x"])
        sums.append(pd.Series(sc[:, 0], index=lab))
    S = pd.concat(sums, axis=1).fillna(0.0).to_numpy()              # [G, K]
    G = S.shape[0]
    V = S.T @ S * G / (G - 1)
    return ks, np.array(b), V, G


if "event" in stages:
    pe = s5.add_path_counts(pn, fm, inv, KS, width=WIDTH)
    ev = {}
    rows = []
    for name, keep in (("all", np.ones(len(pe), bool)), ("up", (pe.x >= 0).to_numpy()), ("down", (pe.x <= 0).to_numpy())):
        ks, b, V, G = event_study(pe[keep])
        se = np.sqrt(np.diag(V))
        il = [ks.index(k) for k in LEADS]
        ia = [ks.index(k) for k in LAGS]
        bl, Vl = b[il], V[np.ix_(il, il)]
        wald = float(bl @ np.linalg.solve(Vl, bl))
        lag_mean = float(b[ia].mean())
        lag_se = float(np.sqrt(np.ones(len(ia)) @ V[np.ix_(ia, ia)] @ np.ones(len(ia))) / len(ia))
        ev[name] = {"ks": ks, "b": b, "V": V, "clusters": G}
        summary.setdefault("event" + TAG, {})[name] = {
            "base": f"bin {B0}", "clusters": int(G), "events": int((pe[keep].x != 0).sum()),
            "leads_wald": wald, "leads_df": len(il), "leads_p": float(stats.chi2.sf(wald, len(il))),
            "lead_mean": float(bl.mean()), "lag_mean": lag_mean, "lag_mean_se": lag_se,
            "lag_mean_minus_lead_mean": float(b[ia].mean() - bl.mean())}
        for k, bb, ss in zip(ks, b, se):
            rows.append({"direction": name, "k": k, "bin": k // WIDTH, "minutes": k * 5, "beta": bb, "se": ss})
        rows.append({"direction": name, "k": BASE, "bin": B0, "minutes": BASE * 5, "beta": 0.0, "se": 0.0})
        np.save(os.path.join(out, f"event_vcov_{name}{TAG}.npy"), V)
        print(f"[event] {name}: leads Wald {wald:.1f} (df {len(il)}, p {summary['event' + TAG][name]['leads_p']:.3g}), "
              f"mean lag {lag_mean:+.3f} ({lag_se:.3f})  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).sort_values(["direction", "k"]).to_csv(os.path.join(out, f"event_study_k1{TAG}.csv"), index=False)
    save()


# ============================================================ audited sensitivity from saved event outputs
if "honest" in stages:
    import subprocess, sys
    if a.base_bin != 1:
        raise ValueError("The audited FLCI currently requires --base-bin 1")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "honest_sensitivity.py"), "--dir", out], check=True)
    summary = json.load(open(_sp, encoding="utf-8"))


# ============================================================ leave one zone out
def jk_one(z):
    d = pn[pn.zone != z]
    return {"zone": int(z), **{name: est(d, xs)[0] for name, xs in SPECS}}


if "jackknife" in stages:
    zones = np.unique(pn.zone)
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            jk = pool.map(jk_one, zones)
    else:
        jk = [jk_one(z) for z in zones]
    jk = pd.DataFrame(jk)
    jk["events_left_out"] = jk.zone.map(pn[pn.x != 0].groupby("zone").size()).fillna(0).astype(int)
    jk.to_csv(os.path.join(out, "jackknife_zone.csv"), index=False)
    full = {name: est(pn, xs)[0] for name, xs in SPECS}
    G = len(jk)
    res = {}
    for name, _ in SPECS:
        v = jk[name].to_numpy()
        res[name] = {"full": full[name], "min": float(v.min()), "max": float(v.max()),
                     "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)),
                     "se_jackknife": float(np.sqrt((G - 1) / G * ((v - v.mean()) ** 2).sum())),
                     "zone_min": int(jk.zone[np.argmin(v)]), "zone_max": int(jk.zone[np.argmax(v)]),
                     "share_negative": float((v < 0).mean()), "zones": int(G)}
        print(f"[jackknife] {name}: {full[name]:+.3f}, range {res[name]['min']:+.3f} to {res[name]['max']:+.3f}, "
              f"jackknife s.e. {res[name]['se_jackknife']:.3f}", flush=True)
    summary["jackknife"] = res
    save()


# ============================================================ offset of the window counts
if "offset" in stages:
    res = {}
    for ps in (0.0, 0.01, 0.1, 1.0):
        for name, xs in SPECS:
            b, se, n = est(pn, xs, pseudo=ps)
            res[f"{name}|{ps}"] = {"pseudo": ps, "spec": name, "beta": b, "se": se, "n": n}
            print(f"[offset] {name} pseudo {ps}: {b:+.3f} ({se:.3f}) n {n}", flush=True)
    summary["offset"] = res
    save()


# ============================================================ source of the identifying variation
if "variation" in stages:
    r = pn.x.to_numpy(dtype=float).copy()
    keys = [pn.dh.to_numpy(), pn.zh.to_numpy()]
    for _ in range(100):
        for k in keys:
            r = r - pd.Series(r).groupby(k).transform("mean").to_numpy()
    d = pn[["station", "slot", "day", "x", "zone"]].copy()
    d["r2"] = r ** 2
    xr = d.x.round(3)
    mode = xr.groupby([d.station, d.slot]).transform(lambda s: s.value_counts().index[0])
    d["dev"] = (xr - mode).abs() > cc.jump_threshold
    # runs of consecutive panel days (gaps of up to 2 days allowed) with the same departing value
    d = d.sort_values(["station", "slot", "day"])
    d["xr"] = xr.loc[d.index]
    kind = np.full(len(d), "usual", dtype=object)
    for _, g in d[d.dev].groupby(["station", "slot"]):
        days, vals, idx = g.day.to_numpy(), g.xr.to_numpy(), g.index.to_numpy()
        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or days[i] - days[i - 1] > 3 or vals[i] != vals[start]:
                span = days[i - 1] - days[start] + 1
                kind[d.index.get_indexer(idx[start:i])] = "persistent departure" if span >= 7 else "transient"
                start = i
    d["kind"] = kind
    tot = d.r2.sum()
    share = d.groupby("kind").r2.sum() / tot
    counts = d.groupby("kind").size()
    ev_counts = d[d.x != 0].groupby("kind").size()
    # balance: switch size on lagged and current pre-window demand, same fixed effects as design D
    W = a.window
    Y = np.where(inv, 0.0, np.clip(fm.dur.astype(np.float64), 0, None))
    cy = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(Y, 0)])
    cbad = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(inv.astype(np.float64), 0)])
    st, t = pn.station.to_numpy(), pn.t.to_numpy()
    bal = pn[["x", "zone", "dh", "zh"]].copy()
    for lag, lab in ((0, "lag0"), (1, "lag1d"), (7, "lag7d")):
        tt = t - lag * s5.STEPS_PER_DAY
        ok = tt - W >= 0
        tt_ = np.where(ok, tt, W)
        cnt = cy[tt_, st] - cy[tt_ - W, st]
        bad = (cbad[tt_, st] - cbad[tt_ - W, st]) > 0
        bal[lab] = np.where(ok & ~bad, np.log(cnt + 0.1), np.nan)
    bal = bal.dropna().reset_index(drop=True)
    xs_b = ["lag0", "lag1d", "lag7d"]
    sd_x = {c: float(bal[c].std()) for c in xs_b}
    rb = s5.fe_ols_multi(bal, "x", xs_b, FE_D)
    summary["variation_source"] = {
        "share_residual_variance": {k: float(v) for k, v in share.items()},
        "rows": {k: int(v) for k, v in counts.items()}, "events": {k: int(v) for k, v in ev_counts.items()},
        "station_slot_cells_with_schedule_change": int(d[d.kind == "persistent departure"].groupby(["station", "slot"]).ngroups),
        "station_slot_cells": int(d.groupby(["station", "slot"]).ngroups),
        "balance": {"n": rb["n"], "coef": rb["coef"], "se": rb["se"],
                    "coef_per_sd": {c: rb["coef"][c] * sd_x[c] for c in xs_b},
                    "p": {c: float(2 * stats.norm.sf(abs(rb["coef"][c] / rb["se"][c]))) for c in xs_b},
                    "sd_x_residual_scale": float(np.sqrt((r ** 2).mean()))},
    }
    print(json.dumps(summary["variation_source"], ensure_ascii=False, indent=1, default=float), flush=True)
    save()
print(f"完成 {time.time() - t0:.0f}s → {out}")
