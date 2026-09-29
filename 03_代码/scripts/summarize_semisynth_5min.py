"""汇总 5 分钟半合成验证（semisynthetic_5min.py 的输出），并反推与真实估计相符的"到达弹性"。

对每个平均会话时长 m 与窗口 W：
- 截距 a_W、斜率 b_W：合成存量（stock）上设计 D 的估计对真值 β 的线性回归（跨种子平均）
- 窗口口径真值比 r_W = truth 的估计 / β（无噪声时的"存量稀释"比例）
- 噪声衰减比 = (stock 斜率) / (truth 斜率)
- 反推到达弹性：β_arr(W) = (真实估计_W − a_W) / b_W，标准误 = 真实 se_W / |b_W|；再对各窗口做逆方差加权合并，
  并给出各窗口之间是否一致的卡方检验（窗口间估计相关，检验偏松，只作参考）

用法：python scripts/summarize_semisynth_5min.py <结果目录>   → semisynth_5min_summary.csv、semisynth_5min_implied.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "05_实验结果", "因果估计", "5min")
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(res, "semisynth_5min_*.csv"))
                if not f.endswith(("_summary.csv", "_implied.csv"))], ignore_index=True)
real = df[(df.outcome == "real")].drop_duplicates(["W_min", "design"]).set_index(["design", "W_min"])
syn = df[df.outcome.isin(["stock", "flow", "truth"]) & df.design.isin(["C", "D"])]

rows = []
for (m, dsg, W), g in syn.groupby(["mean_min", "design", "W_min"]):
    rec = {"mean_min": m, "design": dsg, "W_min": W}
    for oc in ("stock", "flow", "truth"):
        q = g[g.outcome == oc].groupby("beta_true").beta_hat.agg(["mean", "std", "count"])
        if len(q) >= 2:
            slope, icpt = np.polyfit(q.index.to_numpy(), q["mean"].to_numpy(), 1)
            rec[f"{oc}_slope"], rec[f"{oc}_icpt"] = slope, icpt
        for b, r in q.iterrows():
            rec[f"{oc}_b{b:+.1f}"] = r["mean"]
            rec[f"{oc}_b{b:+.1f}_sd"] = r["std"]
    if "stock_slope" in rec and "truth_slope" in rec:
        rec["noise_attenuation"] = rec["stock_slope"] / rec["truth_slope"]
    rec["real_beta"] = real.loc[(dsg, W), "beta_hat"] if (dsg, W) in real.index else np.nan
    rec["real_se"] = real.loc[(dsg, W), "se"] if (dsg, W) in real.index else np.nan
    rows.append(rec)
summ = pd.DataFrame(rows).sort_values(["design", "mean_min", "W_min"])
summ.to_csv(os.path.join(res, "semisynth_5min_summary.csv"), index=False)

imp = []
for (m, dsg), g in summ[summ.design == "D"].groupby(["mean_min", "design"]):
    b = (g.real_beta - g.stock_icpt) / g.stock_slope
    se = g.real_se / g.stock_slope.abs()
    w = 1 / se ** 2
    pooled = float((w * b).sum() / w.sum())
    chi2 = float((w * (b - pooled) ** 2).sum())
    for W, bi, si, rt in zip(g.W_min, b, se, g.truth_slope):
        imp.append({"mean_min": m, "W_min": W, "implied_arrival_beta": bi, "se": si, "truth_ratio": rt})
    imp.append({"mean_min": m, "W_min": "pooled", "implied_arrival_beta": pooled, "se": float(np.sqrt(1 / w.sum())),
                "chi2_between_windows": chi2, "df": len(g) - 1})
imp = pd.DataFrame(imp)
imp.to_csv(os.path.join(res, "semisynth_5min_implied.csv"), index=False)
pd.set_option("display.width", 200)
cols = ["design", "mean_min", "W_min", "truth_slope", "stock_slope", "flow_slope", "stock_icpt", "noise_attenuation", "real_beta"]
print(summ[[c for c in cols if c in summ]].round(3).to_string(index=False))
print()
print(imp.round(3).to_string(index=False))
