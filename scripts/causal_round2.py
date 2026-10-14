"""Round-2 checks of the switch-event design (second review round, 2026-10-13).

Stages (comma-separated with --stages; default all):
  main      reproduce the main estimates (design D, half-window 180 min, with and without exposure terms)
  boot      zone-cluster bootstrap of design D (with and without exposures)
  ri        randomisation inference: within each (station, slot) the switch sizes x are permuted across days
            (the exposure terms move with their rows); the share of |beta_perm| >= |beta_hat| is the p-value
  direction design D separately for price increases and decreases (rows with the other sign dropped)
  event     event study of design D by direction (5-min leads and lags in 15-min steps)
  variation characterise the identifying variation: events, station-days, daily switch sequences, minutes,
            share of station-slot cells with variation in x, residual variance of x after the fixed effects
  periods   stability: Sep-Oct vs Nov-Jan, excluding December 2022 (epidemic exit wave), and excluding the
            week of the National Day holiday is already done by the holiday filter (+-1 day)
Outputs go to results/causal/round2/ (CSV and a JSON summary).
Usage: python scripts/causal_round2.py [--stages all] [--reps 200] [--workers 2] [--max-seconds 0] [--data-dir data/interim/fiveMin]
boot and ri are resumable: rerunning the same command continues from the reps already saved.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

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
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--window", type=int, default=36)
ap.add_argument("--seed", type=int, default=20261013)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--max-seconds", type=float, default=0, help="stop boot/ri after this many seconds (rerun to resume)")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, list(a.overrides))
root = project_root(cfg.paths.get("root"))
out = ensure_dir(a.out or os.path.join(root, "results", "causal", "round2"))
stages = {"main", "boot", "ri", "direction", "event", "variation", "periods"} if a.stages == "all" else set(a.stages.split(","))
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
FE_D, FE_C = DESIGNS["D"]["fe"], DESIGNS["C"]["fe"]
XS_E = ["x", "S1", "S2", "S3"]
_sp = os.path.join(out, "summary.json")
summary = json.load(open(_sp, encoding="utf-8")) if os.path.exists(_sp) else {}
summary.update({"panel_rows": int(len(pn)), "events": int((pn.x != 0).sum()), "stations": int(pn.station.nunique()),
           "zones": int(pn.zone.nunique())})
print(f"面板 {len(pn):,} 行，事件 {summary['events']:,}，{time.time() - t0:.0f}s", flush=True)


def est(df, xs, fe=FE_D):
    r = s5.ppml_pair(df, xs, fe)
    return r["coef"]["x"], r["se"]["x"], r["n"]


def save():
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)


if "main" in stages:
    for name, xs in (("D_noexp", ["x"]), ("D_exp", XS_E)):
        b, se, n = est(pn, xs)
        summary[name] = {"beta": b, "se_cluster_zone": se, "n": n}
        print(f"[main] {name}: {b:+.3f} ({se:.3f})", flush=True)
    save()

def boot_rep(rep):
    """One zone-cluster bootstrap draw; the generator is seeded by (seed, rep), so results do not depend on chunking."""
    rng = np.random.default_rng([a.seed, 0, rep])
    draw = rng.choice(ZONES, size=len(ZONES), replace=True)
    parts = []
    for i, z in enumerate(draw):
        g = BY_ZONE[z].copy()
        g["zone"] = i
        g["station"] = g["station"] * 1000 + i
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    d["zh"] = d["station"] * s5.STEPS_PER_DAY + d["slot"]
    return {"rep": rep, "D_noexp": est(d, ["x"])[0], "D_exp": est(d, XS_E)[0]}


def ri_rep(rep):
    """One permutation: within each (station, slot) cell the switch sizes x are reassigned across days."""
    rng = np.random.default_rng([a.seed, 1, rep])
    perm = np.arange(len(pn))
    for ix in IDX_LISTS:
        perm[ix] = rng.permutation(ix)
    d = pn.copy()
    d["x"] = pn.x.to_numpy()[perm]
    return {"rep": rep, "D_noexp": est(d, ["x"])[0], "D_exp": est(d, XS_E)[0]}


def run_reps(name, fn, fname):
    """Resumable: reps already in the CSV are skipped; stops early after --max-seconds (rerun to continue)."""
    path = os.path.join(out, fname)
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["rep", "D_noexp", "D_exp"])
    todo = [r for r in range(a.reps) if r not in set(done.rep.astype(int))]
    rows = done.to_dict("records")
    batch = max(a.workers, 1) * 2
    pool = mp.get_context("fork").Pool(a.workers) if a.workers > 1 else None
    try:
        for i in range(0, len(todo), batch):
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                break
            chunk = todo[i: i + batch]
            rows += pool.map(fn, chunk) if pool else [fn(r) for r in chunk]
            pd.DataFrame(rows).sort_values("rep").to_csv(path, index=False)
            print(f"[{name}] {len(rows)}/{a.reps}  {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool:
            pool.close()
    df = pd.DataFrame(rows)
    return df if len(df) >= a.reps else None


if "boot" in stages:
    ZONES = np.unique(pn.zone)
    BY_ZONE = {z: g for z, g in pn.groupby("zone")}
    bt = run_reps("boot", boot_rep, "bootstrap_D.csv")
    if bt is not None:
        summary["bootstrap"] = {k: {"se": float(bt[k].std(ddof=1)), "q025": float(bt[k].quantile(0.025)),
                                    "se_iqr": float((bt[k].quantile(0.75) - bt[k].quantile(0.25)) / 1.349),
                                    "skew": float(bt[k].skew()),
                                    "q975": float(bt[k].quantile(0.975)), "reps": int(len(bt))}
                                for k in ("D_noexp", "D_exp")}
        save()

if "ri" in stages:
    IDX_LISTS = [np.asarray(v) for v in pn.groupby(["station", "slot"]).indices.values() if len(v) > 1]
    ri = run_reps("ri", ri_rep, "randomisation_D.csv")
    if ri is not None:
        b_hat = {k: est(pn, xs)[0] for k, xs in (("D_noexp", ["x"]), ("D_exp", XS_E))}
        summary["randomisation"] = {k: {"beta_hat": b_hat[k],
                                        "p_two_sided": float((np.abs(ri[k]) >= abs(b_hat[k])).mean()),
                                        "p_two_sided_plus1": float(((np.abs(ri[k]) >= abs(b_hat[k])).sum() + 1) / (len(ri) + 1)),
                                        "sd_null": float(ri[k].std(ddof=1)), "reps": int(len(ri))} for k in b_hat}
        save()

if "direction" in stages:
    res = {}
    for name, keep in (("up", pn.x >= 0), ("down", pn.x <= 0)):
        sub = pn[keep]
        b, se, n = est(sub, ["x"])
        res[name] = {"beta": b, "se": se, "n": n, "events": int((sub.x != 0).sum())}
        print(f"[direction] {name}: {b:+.3f} ({se:.3f}) events {res[name]['events']}", flush=True)
    summary["direction"] = res
    save()

if "event" in stages:
    ks = list(range(-33, 36, 3))
    pe = s5.add_path_counts(pn, fm, inv, ks)
    frames = []
    for name, keep in (("all", np.ones(len(pe), bool)), ("up", pe.x >= 0), ("down", pe.x <= 0)):
        es = s5.event_study_ppml(pe[keep], ks, FE_D)
        es.insert(0, "direction", name)
        frames.append(es)
        print(f"[event] {name} {time.time() - t0:.0f}s", flush=True)
    pd.concat(frames).to_csv(os.path.join(out, "event_study_D_direction.csv"), index=False)

if "variation" in stages:
    ev = pn[pn.x != 0]
    t_ev = pd.DatetimeIndex(fm.time[ev.t.to_numpy()])
    seq = ev.assign(date=t_ev.normalize()).groupby(["station", "date"]).slot.apply(lambda s: tuple(sorted(s)))
    station_days = pn.assign(date=pd.DatetimeIndex(fm.time[pn.t.to_numpy()]).normalize())[["station", "date"]].drop_duplicates()
    cell = pn.groupby(["station", "slot"]).x
    share_var = float((cell.nunique() > 1).mean())
    # residual variance of x after the two sets of fixed effects of design D (alternating projections, unweighted)
    r = pn.x.to_numpy(dtype=float).copy()
    keys = [pn.dh.to_numpy(), pn.zh.to_numpy()]
    for _ in range(50):
        for k in keys:
            m = pd.Series(r).groupby(k).transform("mean").to_numpy()
            r = r - m
    minutes = (t_ev.minute.to_numpy())
    summary["variation"] = {
        "events": int(len(ev)), "event_station_days": int(len(seq)), "panel_station_days": int(len(station_days)),
        "distinct_daily_switch_sequences": int(seq.nunique()),
        "share_top10_sequences": float(seq.value_counts().head(10).sum() / len(seq)),
        "share_station_slot_cells_with_variation_in_x": share_var,
        "var_x": float(pn.x.var()), "var_x_residual_after_FE_D": float(r.var()),
        "share_residual": float(r.var() / pn.x.var()),
        "share_events_on_hour_or_half_hour": float(np.isin(minutes, (0, 30)).mean()),
        "share_increases": float((ev.x > 0).mean()),
        "median_abs_x": float(ev.x.abs().median()),
        "events_by_hour": {int(h): int(c) for h, c in pd.Series(t_ev.hour).value_counts().sort_index().items()},
        "holiday_pm1_excluded": bool(cc.exclude_holiday_pm1),
    }
    print(json.dumps(summary["variation"], ensure_ascii=False, indent=1, default=float), flush=True)
    save()

if "periods" in stages:
    date = pd.DatetimeIndex(fm.time[pn.t.to_numpy()]).normalize()
    res = {}
    masks = {"Sep-Oct": date < pd.Timestamp("2022-11-01"), "Nov-train_end": date >= pd.Timestamp("2022-11-01"),
             "excl_Dec2022": ~((date >= pd.Timestamp("2022-12-01")) & (date < pd.Timestamp("2023-01-01")))}
    for name, m in masks.items():
        sub = pn[np.asarray(m)]
        b, se, n = est(sub, ["x"])
        res[name] = {"beta": b, "se": se, "n": n, "events": int((sub.x != 0).sum())}
        print(f"[periods] {name}: {b:+.3f} ({se:.3f}) events {res[name]['events']}", flush=True)
    summary["periods"] = res
    save()
print(f"完成 {time.time() - t0:.0f}s → {out}")
