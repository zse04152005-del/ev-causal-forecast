"""阶段 5.1：用站点级 5 分钟数据复核价格切换（把小时数据上的判断在更细的时间尺度上重做）。

输入（02_数据/interim/fiveMin/，由 D3 压缩包中的 station-processed/features 转成 npz）：
    station_{e_price,s_price,duration,occupancy,volume}.npz   time [T]、cols [N 站点号]、X [T, N] float32
    station_inf.csv（站点 → 小区 TAZID、桩数）、station_information_zone.csv（主数据使用的 1,362 个站）
输出（02_数据/processed/audit/fiveMin/）：
    station_pricing.csv     每个站点的切换次数、有切换的天数、时刻表类型、:05 延迟占比、瞬时尖峰次数、桩数、小区
    switch_minutes.csv      切换发生在一天中的哪一分钟（训练期，主数据站点）
    clean_events.csv        不同窗口宽度 × 甜甜圈下的干净切换事件数（按情境）
    schedule_types.csv      时刻表类型（一天内的切换时刻 + 价格水平序列）及其站点数、天数
    fiveMin_audit.json      汇总数字
用法：python check13_5min.py [数据目录]  （默认 ../../data_fiveMin，即 ev_project/data_fiveMin）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data_fiveMin")
OUT = os.path.join(HERE, "..", "data_audit", "fiveMin")
os.makedirs(OUT, exist_ok=True)
STEP_MIN = 5
T_TRAIN_END = 127 * 288 - 1            # 训练期最后一个 5 分钟点：2023-01-05 23:55（与小时数据的 train_end 对齐）


def load(name):
    z = np.load(os.path.join(DATA, name + ".npz"), allow_pickle=True)
    return z["time"], z["cols"].astype(int), z["X"]


time, cols, E = load("station_e_price")
_, _, S = load("station_s_price")
_, _, D = load("station_duration")
_, _, O = load("station_occupancy")
P = E + S                                                            # 总价 = 电价 + 服务费（与主数据 price: total 口径一致）
inf = pd.read_csv(os.path.join(DATA, "station_inf.csv")).set_index("station_id")
main_ids = set(pd.read_csv(os.path.join(DATA, "station_information_zone.csv")).station_id)
zone_static = pd.read_csv(os.path.join(HERE, "..", "data_audit", "zone_static.csv")).set_index("zone")
t = pd.DatetimeIndex(time)
T, N = P.shape
in_main = np.array([c in main_ids for c in cols])
summary: dict = {"T": int(T), "N_stations": int(N), "N_main": int(in_main.sum()), "train_end_step": T_TRAIN_END,
                 "train_end_time": str(time[T_TRAIN_END])}

# ---------------------------------------------------------------- 数据质量（站点级）
neg = D < -1e-6
summary["negative_duration_cells"] = int(neg.sum())
summary["negative_duration_stations"] = int(neg.any(axis=0).sum())
summary["negative_duration_min"] = float(D.min())
summary["duration_gt_capacity_cells"] = int((D > (inf.loc[cols, "charge_count"].values[None, :] / 12 + 1e-6)).sum())

# ---------------------------------------------------------------- 价格切换
dP = np.zeros((T, N), dtype=bool)
dP[1:] = np.abs(P[1:] - P[:-1]) > 1e-6                              # dP[t]：t-1 → t 价格变了
tr = np.arange(T) <= T_TRAIN_END
mod = (t.hour * 60 + t.minute).values
date = t.normalize()
n_days_train = int(len(np.unique(date[tr])))
sw_all = dP.sum(axis=0)
sw_train = (dP & tr[:, None]).sum(axis=0)
# 有切换的天数
day_id = (date - date[0]).days.values
days_with_sw = np.array([len(np.unique(day_id[dP[:, j] & tr])) for j in range(N)])

# 瞬时尖峰：价格变了又在 ≤ 2 个点（10 分钟）内变回原值（例如 16:30 → 16:35 涨 0.1 → 16:40 回落）
spike = np.zeros((T, N), dtype=bool)
for lag in (1, 2):
    back = np.zeros((T, N), dtype=bool)
    back[: T - lag - 1] = dP[: T - lag - 1] & (np.abs(P[lag + 1:] - P[: T - lag - 1]) < 1e-6)[: T - lag - 1] & dP[lag: T - 1]
    spike |= back
spike_cnt = (spike & tr[:, None]).sum(axis=0)

# 每天的价格序列签名（切换时刻 + 切换后价格），用于时刻表分型
sig_day = {}
for j in np.flatnonzero(sw_train > 0):
    rows = []
    for d in np.unique(day_id[tr]):
        m = (day_id == d) & tr
        idx = np.flatnonzero(m)
        ch = idx[dP[idx, j]]
        rows.append((d, tuple((int(mod[c]), round(float(P[c, j]), 2)) for c in ch)))
    sig_day[j] = rows

# 站点类型：TOU = 训练期内 ≥ 50% 的天有切换（与小区级定义一致）；weak = 有过切换但 < 50%；fixed = 从无切换
typ = np.where(days_with_sw >= 0.5 * n_days_train, "TOU", np.where(sw_train > 0, "weak", "fixed"))
# 是否落在标准时刻（0、5 的整数倍分钟 → 看 :00/:30 与 +5 分钟延迟）
station_rows = []
for j in range(N):
    m = mod[dP[:, j] & tr]
    station_rows.append({
        "station": int(cols[j]), "zone": int(inf.loc[cols[j], "TAZID"]), "piles": int(inf.loc[cols[j], "charge_count"]),
        "in_main": bool(in_main[j]), "n_switch_train": int(sw_train[j]), "days_with_switch": int(days_with_sw[j]),
        "type": typ[j], "share_on_half_hour": float(np.mean(m % 30 == 0)) if len(m) else np.nan,
        "share_plus5": float(np.mean(m % 30 == 5)) if len(m) else np.nan, "spikes": int(spike_cnt[j]),
    })
st = pd.DataFrame(station_rows)
st.to_csv(os.path.join(OUT, "station_pricing.csv"), index=False)

sm = st[st.in_main]
summary["stations_main"] = {k: int((sm.type == k).sum()) for k in ("TOU", "weak", "fixed")}
summary["stations_all"] = {k: int((st.type == k).sum()) for k in ("TOU", "weak", "fixed")}
summary["piles_main"] = {k: int(sm.loc[sm.type == k, "piles"].sum()) for k in ("TOU", "weak", "fixed")}

# 与小区级（小时数据）分型对照
zt = sm.groupby("zone").type.agg(lambda s: "TOU" if (s == "TOU").any() else ("weak" if (s == "weak").any() else "fixed"))
cmp = pd.crosstab(zone_static.pricing.reindex(zt.index), zt, rownames=["hourly_zone"], colnames=["station_5min"])
summary["zone_class_crosstab"] = json.loads(cmp.to_json())
summary["zones_with_TOU_station"] = int((zt == "TOU").sum())

# 一天中的切换时刻
sw_min = pd.Series(mod[(dP & tr[:, None] & in_main[None, :]).nonzero()[0]])
cnt = sw_min.value_counts().sort_index()
pd.DataFrame({"minute_of_day": cnt.index, "hhmm": [f"{m // 60:02d}:{m % 60:02d}" for m in cnt.index],
              "count": cnt.values}).to_csv(os.path.join(OUT, "switch_minutes.csv"), index=False)
summary["switch_events_train_main"] = int(cnt.sum())
summary["share_switch_on_half_hour"] = float(cnt[cnt.index % 30 == 0].sum() / cnt.sum())
summary["share_switch_plus5"] = float(cnt[cnt.index % 30 == 5].sum() / cnt.sum())
summary["share_switch_other"] = float(cnt[(cnt.index % 30 != 0) & (cnt.index % 30 != 5)].sum() / cnt.sum())
summary["spikes_train_main"] = int(spike_cnt[in_main].sum())

# 时刻表分型：按"一天里的切换时刻序列（忽略价格水平）"归类
type_rows = {}
for j, rows in sig_day.items():
    if not in_main[j]:
        continue
    for d, sig in rows:
        key = tuple(m for m, _ in sig)
        type_rows.setdefault(key, {"days": 0, "stations": set()})
        type_rows[key]["days"] += 1
        type_rows[key]["stations"].add(cols[j])
sched = pd.DataFrame([{"switch_times": " ".join(f"{m // 60:02d}:{m % 60:02d}" for m in k), "n_switch": len(k),
                       "station_days": v["days"], "n_stations": len(v["stations"])} for k, v in type_rows.items()])
sched = sched.sort_values("station_days", ascending=False)
sched.to_csv(os.path.join(OUT, "schedule_types.csv"), index=False)
summary["n_daily_schedule_types_main"] = int(len(sched))
summary["top5_schedule_types_share_of_station_days"] = float(sched.station_days.head(5).sum() / sched.station_days.sum())
summary["station_days_with_no_switch_zero_switch_day_share_TOU"] = float(
    1 - sm[sm.type == "TOU"].days_with_switch.sum() / (n_days_train * max((sm.type == "TOU").sum(), 1)))

# ---------------------------------------------------------------- 独立时刻表（同一运营商克隆）：站点之间的价格路径相关性
tou_j = np.flatnonzero((typ == "TOU") & in_main)
tr_idx = np.flatnonzero(tr)
Pk = P[np.ix_(tr_idx, tou_j)]
codes = pd.factorize(pd.Series([hash(Pk[:, k].round(2).tobytes()) for k in range(Pk.shape[1])]))[0]
summary["distinct_full_price_paths_TOU_main"] = int(codes.max() + 1)         # 训练期整条价格路径完全相同的站点归为一类
summary["largest_identical_path_group"] = int(np.bincount(codes).max())
st.loc[st.station.isin(cols[tou_j]), "path_group"] = pd.Series(codes, index=st.index[st.station.isin(cols[tou_j])]).values
st.to_csv(os.path.join(OUT, "station_pricing.csv"), index=False)

# ---------------------------------------------------------------- 干净事件计数（不同窗口 × 甜甜圈）
# 干净事件：t 时刻本站价格跳变（|Δ ln p| > 0.01），且 [t-W-d, t-1] 内与 [t, t+W-1+d] 内没有其他跳变；
# 过滤瞬时尖峰（跳变后 ≤ 10 分钟内变回）；窗口内无冻结（此处先用"占用与时长同时不变且非零"的粗判，正式版在 switch_5min 里）。
lp = np.log(P)
x = np.zeros_like(lp)
x[1:] = lp[1:] - lp[:-1]
jump = np.abs(x) > 0.01
other = dP.copy()


def clean_counts(W: int, donut: int):
    c = np.cumsum(np.vstack([np.zeros((1, N)), dP.astype(np.int64)]), axis=0)   # c[k] = sum dP[:k]
    n = {}
    t_ix = np.arange(W + donut + 1, T - W - donut)
    ok_pre = (c[t_ix] - c[t_ix - W - donut + 1]) == 0                # dP[t-W-donut+1 .. t-1] 无变化（t-1 之前的窗口内价格不变）
    ok_post = (c[t_ix + W + donut] - c[t_ix + 1]) == 0                # dP[t+1 .. t+W+donut-1] 无变化
    ev = jump[t_ix] & ok_pre & ok_post & ~spike[t_ix] & (t_ix <= T_TRAIN_END - W - donut)[:, None] & in_main[None, :]
    return t_ix, ev


rows = []
ctx_of = None
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("calendar_", os.path.join(HERE, "..", "code", "src", "data", "calendar.py"))
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
ctx_all = cal.context_index(t)
hol = cal.holiday_window(t, 1)
for W in (6, 12, 24, 36):
    for donut in (0, 1, 2):
        t_ix, ev = clean_counts(W, donut)
        ev &= ~hol[t_ix][:, None]
        for c in range(4):
            m = ev & (ctx_all[t_ix] == c)[:, None]
            rows.append({"W_steps": W, "W_min": W * STEP_MIN, "donut_steps": donut, "context": cal.CONTEXT_NAMES[c],
                         "events": int(m.sum()), "stations": int(m.any(axis=0).sum())})
        rows.append({"W_steps": W, "W_min": W * STEP_MIN, "donut_steps": donut, "context": "all",
                     "events": int(ev.sum()), "stations": int(ev.any(axis=0).sum())})
ce = pd.DataFrame(rows)
ce.to_csv(os.path.join(OUT, "clean_events.csv"), index=False)
summary["clean_events_all"] = {f"W{r.W_min}min_d{r.donut_steps}": int(r.events) for r in ce[ce.context == "all"].itertuples()}

json.dump(summary, open(os.path.join(OUT, "fiveMin_audit.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1)[:6000])
print(ce[ce.context == "all"].to_string())
print(sched.head(12).to_string())
