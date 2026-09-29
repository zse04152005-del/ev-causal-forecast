"""用训练期的分时电价切换估计锚定值（设计文档 6.2–6.4），写出锚定 JSON。

当前用 GitHub 版逐小时数据，得到的是"小时数据可行性锚定"（source = hourly_train）；
阶段 6 换成 5 分钟数据后用同一脚本（加 --source 与数据路径）生成正式锚定。

用法：python scripts/estimate_anchors.py [--out configs/anchor/hourly_train.json] [覆盖项 ...]
"""
import argparse
import datetime
import os


import _bootstrap  # noqa: F401
from src.causal.anchors import merge_levels, save_json
from src.causal.switch_did import PanelSpec, build_panel, estimate_cells, estimate_design, event_path
from src.data.dataset import prepare
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--out", default=os.path.join(_bootstrap.CODE_DIR, "configs", "anchor", "hourly_train.json"))
ap.add_argument("--source", default="hourly_train")
ap.add_argument("--no-event-path", action="store_true")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
cc = cfg.causal
P = prepare(cfg)
Y = P.raw[cc.outcome]
spec = PanelSpec(window=int(cfg.anchor.window_hours), jump_threshold=cc.jump_threshold, eps=cc.eps_pile_hours,
                 exclude_holiday_pm1=cc.exclude_holiday_pm1)
t_range = (0, P.split.train_end)
panel = build_panel(Y, P.lp, P.pricing, P.groups, P.time, P.ring_members, valid=~P.frozen, t_range=t_range, spec=spec,
                    day_start=cfg.data.context.day_start, night_start=cfg.data.context.night_start)
print(f"窗口观测 {len(panel):,} 个，其中本区切换 {(panel.x != 0).sum():,} 个（训练期、干净窗口、无冻结值）")

designs = {}
for d in "ABCD":
    r = estimate_design(panel, d)
    designs[d] = r
    ex = "  ".join(f"δ{k[1:]}={r['coef'][k]:+.3f}({r['se'][k]:.3f})" for k in r["coef"] if k != "x")
    print(f"设计 {d}: β = {r['coef']['x']:+.3f} (se {r['se']['x']:.3f})  事件 {r['n_events']:,}  "
          f"小区 {r['n_event_zones']}  FE 后剩余价格变动 {r['resid_var_share_x0']:.3f}  {ex}")

est = estimate_cells(panel, P.G, 4, design=cc.design)
own = merge_levels(est, P.G, 4, cfg.anchor.min_zones, cfg.anchor.max_se)
print("\n格子锚定（功能区 × 情境）：")
for e in own:
    print(f"  G{e['group']} {e['context']:<14} β = {e['beta']:+.3f} (se {e['se']:.3f})  来自 {e['level']:<7} "
          f"小区 {e['n_zones']:>3}  事件 {e['n_events']:>5}")
cross = est.attrs["cross"]
rings = [f"{lo}-{hi}km" for lo, hi in cfg.data.rings_km]
out = {
    "version": datetime.date.today().isoformat(), "source": a.source, "target": cfg.data.target,
    "price": cfg.data.price, "estimand": f"window_{spec.window}h_log_mean", "design": cc.design,
    "outcome": cc.outcome, "t_range": [str(P.time[t_range[0]]), str(P.time[t_range[1]])],
    "own": own,
    "own_levels": est.to_dict(orient="records"),
    "cross": [{"ring": r, "delta": float(cross[f"S{k + 1}"][0]), "se": float(cross[f"S{k + 1}"][1])}
              for k, r in enumerate(rings)],
    "shift": {"gamma": 0.0, "se": 1.0},
    "lag_weights": None,
    "designs": {d: {"beta": r["coef"]["x"], "se": r["se"]["x"], "n_events": r["n_events"]} for d, r in designs.items()},
}
if not a.no_event_path:
    ep = event_path(Y, P.lp, P.pricing, P.time, ~P.frozen, t_range, eps=cc.eps_pile_hours)
    out["event_path"] = ep.to_dict(orient="records")
    print("\n事件研究（相对切换前 1 小时）：" + "  ".join(f"k={int(r.k)}:{r.beta:+.2f}" for r in ep.itertuples()))
save_json(out, a.out)
res = ensure_dir(os.path.join(resolve(cfg.paths.results_dir, project_root(cfg.paths.get("root"))), "因果估计"))
est.to_csv(os.path.join(res, f"anchor_levels_{a.source}.csv"), index=False)
print(f"\n已写出 {a.out}")
