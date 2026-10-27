"""Multi-backbone anchoring (pre-submission check): is the anchored price term independent of the backbone?

For each backbone, the price-blind model, the same model with the anchored price term (cut feedback, Eq. 6 of the paper)
and, optionally, the model with a sign-constrained elasticity learned from zero (for adaptive anchoring, Eq. 10) are
trained with the same tuned settings and the same seeds:

| Backbone | price-blind | anchored | learned from zero |
|---|---|---|---|
| Graph WaveNet (CPA-STGNN) | v2_gwn_blind | v2_main_A0 | r2_proj0 |
| STID | r2_stid | r2_stid_anchor | r2_stid_learn0 |
| LSTM | r2_lstm | r2_lstm_anchor | r2_lstm_learn0 |

Reported per backbone: test MAE / RMSE / CRPS (x100, mean and standard deviation over the seeds present for both models),
the relative difference in MAE (anchored minus price-blind), a paired t-test across seeds, the Diebold-Mariano test on
the loss per forecast origin (all stored horizons, Holm-adjusted over the backbones), the coverage of the 90% interval
before and after online calibration, the elasticity implied by the anchored model (5% price rise in the TOU zones), and,
where the learned runs exist, the learned elasticity, the plug-in weight w and the adaptive elasticity of Eq. 10.

Usage (repository root, CPU only; the loss series come from scripts/loss_series.py, the learned elasticities from
scripts/learned_elasticity.py, see scripts/run_backbones.ps1 -Phase dm,summary):
    python scripts/summarize_backbones.py --runs results/runs \
        --dm results/comparisons/dm/loss_r2.npz results/comparisons/dm/loss_r2m.npz results/comparisons/backbones/loss_bb.npz \
        --le results/comparisons/learned_elasticity.csv results/comparisons/backbones/learned_elasticity_bb.csv
Outputs in results/comparisons/backbones/: backbones_seeds.csv (one row per run), backbones_summary.csv (one row per
backbone) and tab_backbones.tex (table for the Supplementary Material).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap  # noqa: F401
from src.eval.stats import dm_test

BACKBONES = [("Graph WaveNet", "v2_gwn_blind", "v2_main_A0", "r2_proj0"),
             ("STID", "r2_stid", "r2_stid_anchor", "r2_stid_learn0"),
             ("LSTM", "r2_lstm", "r2_lstm_anchor", "r2_lstm_learn0")]
METRICS = {"MAE": "test_MAE", "RMSE": "test_RMSE", "CRPS": "test_CRPS_q", "PICP_raw": "test_PICP90_raw",
           "PICP_aci": "test_PICP90_aci_mean"}

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--dm", nargs="*", default=[], help="loss_*.npz files; later files override earlier keys")
ap.add_argument("--le", nargs="*", default=[], help="learned_elasticity csv files")
ap.add_argument("--anchor", default="configs/anchor/fivemin_main.json")
ap.add_argument("--boot", default="results/causal/round2/summary.json", help="zone-bootstrap s.e. of the anchor")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--out", default="results/comparisons/backbones")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

beta_a = float(json.load(open(a.anchor, encoding="utf-8"))["own"][0]["beta"])
s_a = float(json.load(open(a.boot, encoding="utf-8"))["bootstrap"]["D_exp"]["se"])
print(f"anchor beta_A = {beta_a:.4f}, bootstrap s.e. s_A = {s_a:.4f}")


def read_run(group: str, s: int):
    d = os.path.join(a.runs, f"{group}_s{s}")
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        return None
    sm = json.load(open(p, encoding="utf-8"))
    row = {"group": group, "seed": s, **{k: sm.get(v) for k, v in METRICS.items()},
           "epochs": sm.get("epochs_run"), "seconds": sm.get("seconds"), "readout": sm.get("arc_elasticity_tou_mean")}
    pp = os.path.join(d, "price_params.json")
    if row["readout"] is None and os.path.exists(pp):
        js = json.load(open(pp, encoding="utf-8"))
        if js.get("mode") == "cut" and js.get("beta_tou_by_group_ctx") is not None:   # CPA-STGNN: below the cap = beta
            row["readout"] = float(np.nanmean(np.asarray(js["beta_tou_by_group_ctx"], dtype=float)))
    return row


rows = [r for _, *gs in BACKBONES for g in gs for s in range(a.seeds) if (r := read_run(g, s)) is not None]
seeds = pd.DataFrame(rows)
if seeds.empty:
    raise SystemExit(f"no runs found under {a.runs}")
le = pd.concat([pd.read_csv(f) for f in a.le if os.path.exists(f)], ignore_index=True) if a.le else pd.DataFrame()
if not le.empty:
    le["seed"] = le.run.str.extract(r"_s(\d+)$").astype(int)
    seeds = seeds.merge(le[["group", "seed", "beta_weighted"]].rename(columns={"beta_weighted": "beta_learned"}),
                        on=["group", "seed"], how="left")
seeds.to_csv(os.path.join(a.out, "backbones_seeds.csv"), index=False)

L = {}
for f in a.dm:
    if os.path.exists(f):
        z = np.load(f)
        L.update({k: z[k] for k in z.files})
    else:
        print(f"note: {f} not found")


def adaptive(beta_l: float):
    e = max((beta_l - beta_a) ** 2 - s_a ** 2, 0.0)
    w = e / (e + s_a ** 2)
    return w, min(w * min(beta_a, 0.0) + (1 - w) * beta_l, 0.0)


out = []
for name, blind, anch, learn in BACKBONES:
    b = seeds[seeds.group == blind].set_index("seed")
    c = seeds[seeds.group == anch].set_index("seed")
    common = sorted(set(b.index) & set(c.index))
    r = {"backbone": name, "blind": blind, "anchored": anch, "seeds": len(common)}
    if common:
        for m in ("MAE", "RMSE", "CRPS"):
            r[f"{m}_blind"], r[f"{m}_blind_sd"] = 100 * b.loc[common, m].mean(), 100 * b.loc[common, m].std()
            r[f"{m}_anch"], r[f"{m}_anch_sd"] = 100 * c.loc[common, m].mean(), 100 * c.loc[common, m].std()
            r[f"d{m}_pct"] = 100 * (c.loc[common, m].mean() / b.loc[common, m].mean() - 1)
        r["t_p_MAE"] = float(stats.ttest_rel(c.loc[common, "MAE"], b.loc[common, "MAE"]).pvalue) if len(common) > 1 else np.nan
        r["PICP_raw_anch"], r["PICP_aci_anch"] = c.loc[common, "PICP_raw"].mean(), c.loc[common, "PICP_aci"].mean()
        r["readout_anch"] = c.loc[common, "readout"].mean()
    for m in ("mae", "crps"):
        ka, kb = f"{anch}|{m}_o", f"{blind}|{m}_o"
        if ka in L and kb in L:
            x, y = np.nanmean(L[ka], 1), np.nanmean(L[kb], 1)
            d = dm_test(x, y, h=24)                       # d = anchored - blind; negative = anchored better
            r[f"dm_{m}"], r[f"dm_p_{m}"] = d["dm"], d["p"]
            r[f"share_origins_anch_better_{m}"] = float(np.mean(x < y))
            r[f"dm_seeds_{m}"] = (len(L[f"{anch}|seed_mae"]), len(L[f"{blind}|seed_mae"]))
    lr = seeds[seeds.group == learn]
    if "beta_learned" in seeds and lr.beta_learned.notna().any():
        bl = lr.beta_learned.dropna()
        ad = np.array([adaptive(v) for v in bl])
        r.update({"learn_seeds": len(bl), "beta_learned": bl.mean(), "beta_learned_sd": bl.std(),
                  "w_hat_mean": ad[:, 0].mean(), "w_hat_max": ad[:, 0].max(), "beta_adaptive": ad[:, 1].mean(),
                  "MAE_learn": 100 * lr.MAE.mean()})
    out.append(r)
summ = pd.DataFrame(out)
for m in ("mae", "crps"):                                 # Holm over the backbones that have a DM test
    col = f"dm_p_{m}"
    if col in summ:
        ok = summ[col].notna()
        p = summ.loc[ok, col].values
        o, adj, run = np.argsort(p), np.empty(len(p)), 0.0
        for rank, i in enumerate(o):
            run = max(run, min(1.0, (len(p) - rank) * p[i]))
            adj[i] = run
        summ.loc[ok, f"{col}_holm"] = adj
summ.to_csv(os.path.join(a.out, "backbones_summary.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(summ.round(4).T.to_string())


def fmt(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def pfmt(p):
    return "--" if p is None or not np.isfinite(p) else ("$<0.001$" if p < 0.001 else f"{p:.3f}")


lines = [r"\begin{table}[h]", r"\centering\footnotesize",
         r"\caption{Price-blind and anchored versions of three backbones (errors $\times100$, means over seeds; "
         r"$\Delta$MAE: anchored minus price-blind).}", r"\label{tab:backbones}",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r"Backbone & Seeds & MAE, blind & MAE, anchored & $\Delta$MAE (\%) & DM $p$ & $t$-test $p$ & Implied $\beta$ \\",
         r"\midrule"]
for _, r in summ.iterrows():
    dm = r.get("dMAE_pct", np.nan)
    lines.append(f"{r.backbone} & {int(r.seeds)} & {fmt(r.get('MAE_blind'))} & {fmt(r.get('MAE_anch'))} & "
                 f"{'--' if not np.isfinite(dm) else f'${dm:+.1f}$'} & {pfmt(r.get('dm_p_mae_holm', np.nan))} & "
                 f"{pfmt(r.get('t_p_MAE', np.nan))} & {fmt(r.get('readout_anch')).replace('-', '$-$', 1) if fmt(r.get('readout_anch')) != '--' else '--'} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\par\smallskip{\footnotesize DM: Diebold--Mariano test on the MAE per forecast origin, Holm-adjusted over the "
          r"backbones; $t$-test: paired across seeds. Implied $\beta$: response of the forecasts to a 5\% price rise in "
          r"the TOU zones.}", r"\end{table}"]
with open(os.path.join(a.out, "tab_backbones.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwritten: {a.out}/backbones_seeds.csv, backbones_summary.csv, tab_backbones.tex")
