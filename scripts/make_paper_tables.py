"""生成论文表 1（数据描述统计）与表 2（因果估计与稳健性）的 LaTeX 与 CSV。

表 1：由逐小时小区级数据直接计算（src.data.dataset.prepare，口径与训练、评价完全相同）
表 2：读取 results/causal/5min/ 下的 spec_curve.csv、placebo.csv、panel_supp.csv、boot_cd_diff_summary.json
      （由 scripts/estimate_causal_5min.py、boot_cd_diff.py、estimate_panel_supp.py 生成）

用法（在仓库根目录下）：
    python scripts/make_paper_tables.py [--out results/tables] [--causal-dir <目录>] [paths.data_dir=<数据目录>]
输出：tab_data.tex / .csv、tab_causal.tex / .csv。表里的数字不要手改，结果更新后重跑本脚本。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None)
ap.add_argument("--causal-dir", default=None)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(_bootstrap.DEFAULT_CFG, a.overrides)
root = project_root(cfg.paths.get("root"))
res = resolve(cfg.paths.results_dir, root)
out = ensure_dir(a.out or os.path.join(root, "results", "tables"))
cdir = a.causal_dir or os.path.join(res, "causal", "5min")

# ====================================================================== 表 1：数据描述统计
P = prepare(cfg)
T, N = P.y.shape
price = np.exp(P.lp)
day_range = P.lp.reshape(T // 24, 24, N).max(1) - P.lp.reshape(T // 24, 24, N).min(1)      # 日内对数价差
TYPES = [("Time-of-use", P.pricing == 2), ("Weakly varying", P.pricing == 1), ("Fixed", P.pricing == 0),
         ("All", np.ones(N, bool))]
rowsA = {}
for name, m in TYPES:
    v, y = P.valid[:, m], P.y[:, m]
    rowsA[name] = {"zones": int(m.sum()), "piles": int(P.capacity[m].sum()), "piles_median": float(np.median(P.capacity[m])),
                   "util_mean": float(y[v].mean()), "price_mean": float(price[:, m].mean()),
                   "day_range_median_pct": float(np.median(day_range[:, m]) * 100),
                   "imputed_pct": float(P.frozen[:, m].mean() * 100)}
PARTS = [("Training", "train"), ("Validation", "val"), ("Test", "test")]
rowsB = {}
for name, part in PARTS:
    s, e = P.split.part_range(part)
    sl = slice(s, e + 1)                                    # part_range 的终点是闭区间
    rowsB[name] = {"start": P.time[s], "end": P.time[e], "days": int(round((e + 1 - s) / 24)),
                   "util_mean": float(P.y[sl][P.valid[sl]].mean()), "imputed_pct": float(P.frozen[sl].mean() * 100)}
rowsB["All"] = {"start": P.time[0], "end": P.time[-1], "days": T // 24, "util_mean": float(P.y[P.valid].mean()),
                "imputed_pct": float(P.frozen.mean() * 100)}


def fmt_piles(x):
    return f"{x:.1f}" if x % 1 else f"{int(x)}"


def period(s, e):
    f = lambda t: f"{t.day} {t.strftime('%b')}"                 # 年份见表题
    return f"{s.day}--{f(e)}" if s.month == e.month else f"{f(s)}--{f(e)}"


A_ = [rowsA[n] for n, _ in TYPES]
B_ = [rowsB[n] for n in ("Training", "Validation", "Test", "All")]
L = ["\\begin{table}", "\\tblcaption{tab:data}{Descriptive statistics of the hourly zone-level data by tariff type.}",
     "{\\tblfont", "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccc@{}}", "\\toprule",
     " & Time-of-use & Weak & Fixed & All \\\\", "\\midrule",
     "Zones & " + " & ".join(f"{r['zones']}" for r in A_) + " \\\\",
     "Charging piles & " + " & ".join(f"{r['piles']:,}".replace(",", "{,}") for r in A_) + " \\\\",
     "Utilisation rate, mean & " + " & ".join(f"{r['util_mean']:.3f}" for r in A_) + " \\\\",
     "Price, mean (CNY/kWh) & " + " & ".join(f"{r['price_mean']:.2f}" for r in A_) + " \\\\",
     "Within-day price range (\\%) & " + " & ".join(f"{r['day_range_median_pct']:.1f}" for r in A_) + " \\\\",
     "Imputed zone-hours (\\%) & " + " & ".join(f"{r['imputed_pct']:.1f}" for r in A_) + " \\\\",
     "\\bottomrule", "\\end{tabular*}\\par}",
     # no table note: the definitions are in Section 4.1
     "\\end{table}"]
open(os.path.join(out, "tab_data.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
pd.concat([pd.DataFrame(rowsA).T.assign(panel="tariff"), pd.DataFrame(rowsB).T.assign(panel="sample")]) \
    .to_csv(os.path.join(out, "tab_data.csv"))
print(pd.DataFrame(rowsA).T.round(3).to_string())
print(pd.DataFrame(rowsB).T.to_string())

# ====================================================================== 表 2：因果估计与稳健性
sc = pd.read_csv(os.path.join(cdir, "spec_curve.csv"))
pl = pd.read_csv(os.path.join(cdir, "placebo.csv"))
ps = pd.read_csv(os.path.join(cdir, "panel_supp.csv"))
bt = json.load(open(os.path.join(cdir, "boot_cd_diff_summary.json"), encoding="utf-8"))


def pick(design, W, est="PPML", donut=0, outcome="duration", exposures=False):
    r = sc[(sc.design == design) & (sc.W_min == W) & (sc.est == est) & (sc.donut_min == donut) & (sc.outcome == outcome)
           & (sc.exposures == exposures)]
    assert len(r) == 1, (design, W, est, donut, outcome, exposures, len(r))
    return r.iloc[0]


def cell(r):
    return f"${r.beta:.2f}$ ({r.se:.2f})"


SPEC = [("$W=30$ min", dict(W=30)), ("$W=60$ min", dict(W=60)), ("$W=120$ min", dict(W=120)),
        ("$W=180$ min", dict(W=180)),
        (None, None),
        ("Exposure terms (anchor)", dict(W=180, exposures=True)),
        ("Log-ratio OLS", dict(W=180, est="OLS_log_eps0.01")),
        ("Without first 10 min", dict(W=180, donut=10)),
        ("Outcome: occupancy", dict(W=180, outcome="occupancy")),
        ("Outcome: energy", dict(W=180, outcome="volume"))]
L = ["\\begin{table}", "\\tblcaption{tab:causal}{Own-price elasticity estimated from price-switch events (standard errors in parentheses).}",
     "{\\tblfont", "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lccr@{}}", "\\toprule",
     " & Design C & Design D & Events \\\\", "\\midrule"]
flat = []
for name, kw in SPEC:
    if kw is None:
        L.append("\\addlinespace[2pt]")
        continue
    c, d = pick("C", **kw), pick("D", **kw)
    ev = f"{int(d.n_events):,}".replace(",", "{,}")
    L.append(f"{name} & {cell(c)} & {cell(d)} & {ev} \\\\")
    flat.append({"spec": name, "beta_C": c.beta, "se_C": c.se, "beta_D": d.beta, "se_D": d.se, "events": int(d.n_events),
                 "n_C": int(c.n), "n_D": int(d.n)})
# 安慰剂：切换时刻平移后的"假切换"（60 分钟窗口）
for des in ("C", "D"):
    q = pl[pl.design == des]
    z = (q.beta / q.se).abs()
    flat.append({"spec": f"placebo_{des}", "min": q.beta.min(), "max": q.beta.max(), "n_sig_5pct": int((z > 1.96).sum()),
                 "n": len(q)})
qc, qd = pl[pl.design == "C"], pl[pl.design == "D"]
rng = lambda q: f"${q.beta.min():.2f}$ to ${q.beta.max():.2f}$"
sig = lambda q: f"{int(((q.beta / q.se).abs() > 1.96).sum())} of {len(q)}"
L += ["\\addlinespace[2pt]",
      f"Placebo, range & {rng(qc)} & {rng(qd)} & \\\\",
      f"Placebo, significant & {sig(qc)} & {sig(qd)} & \\\\"]
# 逐小时面板的补充估计
r1 = ps[(ps["method"].str.startswith("PPML")) & (ps["sample"] == "TOU zones")].iloc[0]
r2 = ps[ps["method"].str.startswith("DML")].iloc[0]
L += ["\\addlinespace[2pt]",
      f"Hourly panel, PPML & \\multicolumn{{2}}{{c}}{{{cell(r1)}}} & \\\\",
      f"Hourly panel, DML & \\multicolumn{{2}}{{c}}{{{cell(r2)}}} & \\\\"]
flat += [{"spec": "panel_ppml", "beta_D": r1.beta, "se_D": r1.se, "n_D": int(r1.n)},
         {"spec": "panel_dml", "beta_D": r2.beta, "se_D": r2.se, "n_D": int(r2.n)},
         {"spec": "bootstrap_180", "se_C": bt["boot_se_C"], "se_D": bt["boot_se_D"], "diff_D_minus_C": bt["diff_D_minus_C"],
          "diff_ci_lo": bt["diff_ci95"][0], "diff_ci_hi": bt["diff_ci95"][1]}]
L += ["\\bottomrule", "\\end{tabular*}\\par}",
      "\\tblnote{$W$: half-window. Design C: date-by-slot fixed effects; design D: also station-by-slot fixed effects.}",
      "\\end{table}"]
open(os.path.join(out, "tab_causal.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
pd.DataFrame(flat).to_csv(os.path.join(out, "tab_causal.csv"), index=False)
print(pd.DataFrame(flat).round(3).to_string())
print("→", out)
