"""阶段 6：站点级 5 分钟数据上的因果估计、稳健性与锚定文件（设计文档 6.2–6.4、16.2–16.3；研究大纲 3.3.2）。

分阶段运行（都只用训练期 2022-09-01 至 2023-01-05，避免测试期信息进入锚定）：
    --stages spec      设计 A–D × 窗口 × 甜甜圈 × 结果口径 的 PPML 与对数比 OLS 估计 → spec_curve.csv
    --stages event     事件研究（设计 C、D）→ event_study_{C,D}.csv
    --stages placebo   安慰剂（把切换时刻前后平移）→ placebo.csv
    --stages cells     格子（功能区 × 情境）三层估计 + 合并规则 → anchor JSON、anchor_levels_fivemin_train.csv
    --stages cluster   标准误对比（小区聚类 / 日期聚类 / 双向）→ cluster_compare.csv
    --stages all       以上全部

用法（云端）：
    python scripts/estimate_causal_5min.py --data-dir <fiveMin 目录> --d1 <UrbanEV 逐小时数据目录> --out <输出目录> --stages spec
"""
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.causal import switch_5min as s5
from src.causal.anchors import merge_levels, save_json
from src.causal.switch_did import DESIGNS, estimate_design
from src.data.dataset import prepare
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--data-dir", default=None, help="5 分钟 npz 目录（默认 02_数据/interim/fiveMin）")
ap.add_argument("--d1", default=None, help="覆盖 paths.data_dir（UrbanEV 逐小时数据目录）")
ap.add_argument("--out", default=None, help="结果目录（默认 05_实验结果/因果估计/5min）")
ap.add_argument("--stages", default="spec")
ap.add_argument("--anchor-out", default=os.path.join(_bootstrap.CODE_DIR, "configs", "anchor", "fivemin_train.json"))
ap.add_argument("--controls", type=int, default=80, help="设计 A/B 用的固定电价对照站点数（内存受限时调小）")
ap.add_argument("--main-window", type=int, default=36)
ap.add_argument("--main-design", default="D")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
ov = list(a.overrides) + ([f"paths.data_dir={a.d1}"] if a.d1 else [])
cfg = load_config(a.config, ov)
root = project_root(cfg.paths.get("root"))
data_dir = a.data_dir or os.path.join(root, "02_数据", "interim", "fiveMin")
out_dir = ensure_dir(a.out or os.path.join(root, "05_实验结果", "因果估计", "5min"))
stages = {"spec", "event", "placebo", "cells", "cluster"} if a.stages == "all" else set(a.stages.split(","))

t0 = time.time()
P = prepare(cfg)
fm = s5.load_fivemin(data_dir)
assert list(fm.zone_ids) == list(P.zones), "5 分钟小区表的列顺序与 prepare 不一致"
zpos = {int(z): i for i, z in enumerate(P.zones)}
S_zone = s5.zone_ring_exposure(fm.zone_P, P.ring_members, P.ref_lp)
zone_index = np.array([zpos[int(z)] for z in fm.st_zone])
inv = s5.invalid_mask(fm, P.frozen, zone_index)
T_HI = (P.split.train_end + 1) * 12 - 1                          # 训练期最后一个 5 分钟点
gz = {int(z): int(g) for z, g in zip(P.zones, P.groups)}
cc = cfg.causal
day_start, night_start = cfg.data.context.day_start, cfg.data.context.night_start
print(f"载入完成 {time.time() - t0:.0f}s；训练期 {T_HI + 1:,} 个 5 分钟点；无效点占 {inv[:T_HI + 1].mean():.3f}", flush=True)


def panel(W=36, donut=0, outcome="duration", controls=0, shift=0, eps=0.01):
    spec = s5.Spec5(window=W, donut=donut, outcome=outcome, n_fixed_controls=controls, placebo_shift=shift, eps=eps,
                    jump_threshold=cc.jump_threshold, exclude_holiday_pm1=cc.exclude_holiday_pm1)
    return s5.build_panel_5min(fm, gz, zpos, S_zone, inv, T_HI, spec, day_start, night_start)


def write_rows(name, rows):
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, name), index=False)


# ---------------------------------------------------------------- spec
if "spec" in stages:
    spec_file = os.path.join(out_dir, "spec_curve.csv")
    rows = pd.read_csv(spec_file).to_dict("records") if os.path.exists(spec_file) else []      # 可续跑（内存不足被杀后）
    done = {(r["W_min"], r["donut_min"], r["outcome"]) for r in rows}
    grid = [(W, d, "duration") for W in (6, 12, 24, 36) for d in (0, 2)] + \
           [(W, 0, o) for W in (12, 36) for o in ("occupancy", "volume")]
    for W, donut, outcome in grid:
        if (W * 5, donut * 5, outcome) in done:
            continue
        pn, info = panel(W, donut, outcome, controls=a.controls)
        n_ev = int((pn.x != 0).sum())
        for dsg in "ABCD":
            sub = pn[pn.tou] if DESIGNS[dsg]["tou_only"] else pn
            for expo in (False, True):
                xs = ["x"] + (["S1", "S2", "S3"] if expo else [])
                r = s5.ppml_pair(sub, xs, DESIGNS[dsg]["fe"])
                row = {"est": "PPML", "design": dsg, "W_min": W * 5, "donut_min": donut * 5, "outcome": outcome,
                       "exposures": expo, "beta": r["coef"]["x"], "se": r["se"]["x"], "n": r["n"], "n_events": n_ev}
                row.update({f"delta{k[1:]}": r["coef"][k] for k in xs[1:]})
                row.update({f"delta{k[1:]}_se": r["se"][k] for k in xs[1:]})
                rows.append(row)
            r = estimate_design(sub.assign(J=sub.J), dsg, exposures=False)
            rows.append({"est": "OLS_log_eps0.01", "design": dsg, "W_min": W * 5, "donut_min": donut * 5, "outcome": outcome,
                         "exposures": False, "beta": r["coef"]["x"], "se": r["se"]["x"], "n": r["n"], "n_events": n_ev})
        write_rows("spec_curve.csv", rows)
        del pn
        gc.collect()
        print(f"[spec] W={W * 5}min donut={donut * 5}min {outcome}  {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- event
if "event" in stages:
    ks = list(range(-33, 36, 3))
    pn, _ = panel(36, 0, "duration")
    pn = s5.add_path_counts(pn, fm, inv, ks)
    for dsg in "CD":
        es = s5.event_study_ppml(pn, ks, DESIGNS[dsg]["fe"])
        es.insert(0, "design", dsg)
        es.to_csv(os.path.join(out_dir, f"event_study_{dsg}.csv"), index=False)
        print(f"[event] {dsg} 完成 {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- placebo
if "placebo" in stages:
    rows = []
    for W in (12,):
        for shift in (-48, -36, -24, 24, 36, 48):
            pn, _ = panel(W, 0, "duration", shift=shift)
            for dsg in "CD":
                r = s5.ppml_pair(pn, ["x"], DESIGNS[dsg]["fe"])
                rows.append({"W_min": W * 5, "shift_min": shift * 5, "design": dsg, "beta": r["coef"]["x"],
                             "se": r["se"]["x"], "n": r["n"], "n_events": int((pn.x != 0).sum())})
            write_rows("placebo.csv", rows)
            print(f"[placebo] shift {shift * 5} min  {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- cluster
if "cluster" in stages:
    rows = []
    pn, _ = panel(a.main_window, 0, "duration")
    for dsg in "CD":
        for cl in (("zone",), ("station",), ("day",), ("zone", "day")):
            r = s5.ppml_pair(pn, ["x"], DESIGNS[dsg]["fe"], clusters=cl)
            rows.append({"design": dsg, "clusters": "+".join(cl), "n_clusters": r["n_clusters"], "beta": r["coef"]["x"],
                         "se": r["se"]["x"]})
    write_rows("cluster_compare.csv", rows)
    print(pd.DataFrame(rows).to_string(), flush=True)

# ---------------------------------------------------------------- cells → 锚定
if "cells" in stages:
    pn, info = panel(a.main_window, 0, "duration")
    est = s5.estimate_cells_ppml(pn, P.G, 4, design=a.main_design)
    est.to_csv(os.path.join(out_dir, "anchor_levels_fivemin_train.csv"), index=False)
    own = merge_levels(est, P.G, 4, cfg.anchor.min_zones, cfg.anchor.max_se, require_negative=True)   # 决策：正号格子并入上一层级
    for e in own:
        print(f"  G{e['group']} {e['context']:<14} β = {e['beta']:+.3f} (se {e['se']:.3f})  来自 {e['level']:<7} "
              f"小区 {e['n_zones']:>3}  事件 {e['n_events']:>5}", flush=True)
    cross = est.attrs["cross"]
    rings = [f"{lo}-{hi}km" for lo, hi in cfg.data.rings_km]
    designs = {}
    for dsg in "CD":
        r = s5.ppml_pair(pn, ["x"], DESIGNS[dsg]["fe"])
        designs[dsg] = {"beta": r["coef"]["x"], "se": r["se"]["x"], "n": r["n"]}
    out = {"version": time.strftime("%Y-%m-%d"), "source": "fivemin_train", "target": cfg.data.target, "price": cfg.data.price,
           "estimand": f"window_{a.main_window * 5 // 60}h_log_mean_ppml", "design": a.main_design, "outcome": "duration",
           "t_range": [str(P.time[0]), str(P.time[P.split.train_end])], "own": own, "own_levels": est.to_dict(orient="records"),
           "cross": [{"ring": r, "delta": float(cross[f"S{k + 1}"][0]), "se": float(cross[f"S{k + 1}"][1])}
                     for k, r in enumerate(rings)],
           "shift": {"gamma": 0.0, "se": 1.0}, "lag_weights": None, "designs": designs}
    save_json(out, a.anchor_out)
    city = est[(est.level == "city")].iloc[0]                       # 备选：不分格子，全部用全市估计（弱识别时更稳）
    pooled = dict(out, own=[{"group": "all", "context": "all", "beta": float(min(city["beta"], 0.0)), "se": float(city["se"]),
                             "n_zones": int(city["n_zones"]), "n_events": int(city["n_events"]), "level": "city"}])
    save_json(pooled, a.anchor_out.replace(".json", "_pooled.json"))
    # 主实验锚定（2026-09-29 决策）：全市统一弹性 + δ = 0（溢出估计不稳定）+ γ 关闭；原始 δ 估计保留在 cross_estimated
    main = dict(pooled, source="fivemin_main", cross_estimated=pooled["cross"],
                cross=[{"ring": r, "delta": 0.0, "se": 0.2} for r in rings],
                note="全市统一弹性（设计 D、±3 小时、PPML、训练期）；溢出 δ 置 0，γ 不启用（阶段 6 决策，2026-09-29）")
    save_json(main, os.path.join(os.path.dirname(a.anchor_out), "fivemin_main.json"))
    print(f"已写出 {a.anchor_out}", flush=True)
print(f"全部完成 {time.time() - t0:.0f}s")
