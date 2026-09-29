"""E-SS 预实验：在已知真值的半合成数据上检验识别设计（不需要 PyTorch）。

网格：内生性 ρ ∈ {0, 0.5, 1} × 时刻表跨日变动（无 / 20% 不执行分时 + 20% 平移 3 小时）× 若干随机种子；
每格用设计 A–D 估计弹性，与真值（默认 −0.8）比较。结果写入 05_实验结果/因果估计/ess_pilot.csv。

用法：python scripts/pilot_semisynthetic.py [--seeds 4] [--beta -0.8]
"""
import argparse
import os

import pandas as pd

import _bootstrap  # noqa: F401
from src.causal.switch_did import PanelSpec, build_panel, estimate_design
from src.data.dataset import prepare
from src.eval.semisynthetic import make_semisynthetic, to_prepared
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--beta", type=float, default=-0.8)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
P = prepare(cfg)
rows = []
for rho in (0.0, 0.5, 1.0):
    for label, flat, shift in (("无跨日变动", 0.0, 0.0), ("20%不执行+20%平移", 0.2, 0.2)):
        for seed in range(a.seeds):
            S = make_semisynthetic(P, rho=rho, betas_by_group=(a.beta,) * 3, flat_day_prob=flat, day_shift_prob=shift,
                                   seed=seed)
            Q = to_prepared(P, S)
            panel = build_panel(Q.raw["duration"], Q.lp, Q.pricing, Q.groups, Q.time, Q.ring_members,
                                valid=~Q.frozen, t_range=(0, Q.split.train_end), spec=PanelSpec())
            for d in "ABCD":
                r = estimate_design(panel, d, exposures=False)
                rows.append({"rho": rho, "variation": label, "seed": seed, "design": d, "beta_hat": r["coef"]["x"],
                             "se": r["se"]["x"], "truth": a.beta, "n_events": r["n_events"]})
df = pd.DataFrame(rows)
out = ensure_dir(os.path.join(resolve(cfg.paths.results_dir, project_root(cfg.paths.get("root"))), "因果估计"))
df.to_csv(os.path.join(out, "ess_pilot.csv"), index=False)
summ = (df.groupby(["rho", "variation", "design"])
        .agg(mean=("beta_hat", "mean"), sd=("beta_hat", "std"), mean_se=("se", "mean"))
        .assign(bias=lambda x: x["mean"] - a.beta))
with pd.option_context("display.width", 140):
    print(summ.round(3).to_string())
print(f"\n真值 β = {a.beta}；已写出 {os.path.join(out, 'ess_pilot.csv')}")
