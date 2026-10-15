"""由逐样本结果（evaluate_pipeline.py --save-samples）统计：按故障类型、按过渡电阻分段的区段准确率与测距误差，以及测距误差的分位数。"""
import json
from pathlib import Path

import numpy as np

from simulate_dataset import FAULT_TYPES

RF_BINS = (0.0, 1.0, 10.0, 100.0, 300.0, np.inf)
METHODS = ("本方案", "硬串联", "物理穷举")
case_dir = Path("results/improved/ieee34")
cls = np.array(json.load(open(case_dir / "evaluation.json"))["equivalence_classes"])
S = np.load(case_dir / "samples.npz")
type_name = np.array([n for n, _ in FAULT_TYPES])

out = {}
for sc in sorted({k.split("/")[0] for k in S.files}):
    ft, rf = type_name[S[f"{sc}/ftype"]], S[f"{sc}/rf"]
    res = {}
    for m in METHODS:
        ok = cls[S[f"{sc}/{m}/sec"]] == cls[S[f"{sc}/sec_true"]]
        err = S[f"{sc}/{m}/d_err"]
        groups = {f"故障类型 {t}": ft == t for t in ("SLG", "LL", "LLG", "LLLG")}
        groups.update({f"Rf {lo:g}–{hi:g} Ω": (rf >= lo) & (rf < hi) for lo, hi in zip(RF_BINS[:-1], RF_BINS[1:])})
        res[m] = dict(
            分位数km={f"P{q}": float(np.percentile(err, q)) for q in (50, 75, 90, 95, 99)},
            分组={g: dict(n=int(mask.sum()), class_acc=float(ok[mask].mean()), d_mean=float(err[mask].mean()), d_median=float(np.median(err[mask])))
                  for g, mask in groups.items() if mask.any()})
    out[sc] = res

# 过渡电阻连续曲线：域内（Rf ≤ 300 Ω，训练范围内）与高阻外推（300–1000 Ω）拼接
curve = {}
for sc, bins in (("域内", [(0, 1), (1, 10), (10, 100), (100, 300)]), ("高阻外推", [(300, 500), (500, 1000.01)])):
    rf, t = S[f"{sc}/rf"], cls[S[f"{sc}/sec_true"]]
    for lo, hi in bins:
        m = (rf >= lo) & (rf < hi)
        curve[f"Rf {lo:g}–{min(hi, 1000):g} Ω"] = dict(n=int(m.sum()), **{
            mth: dict(class_acc=float((cls[S[f"{sc}/{mth}/sec"]] == t)[m].mean()), d_mean=float(S[f"{sc}/{mth}/d_err"][m].mean())) for mth in METHODS})
out["过渡电阻曲线"] = curve

# 阶段一（硬串联的区段即神经网络 Top-1）与物理穷举的对错互补性
comp = {}
for sc in [k for k in out if k != "过渡电阻曲线"]:
    t = cls[S[f"{sc}/sec_true"]]
    a, b = cls[S[f"{sc}/硬串联/sec"]] == t, cls[S[f"{sc}/物理穷举/sec"]] == t
    comp[sc] = dict(两者都对=float((a & b).mean()), 仅神经网络对=float((a & ~b).mean()), 仅物理对=float((~a & b).mean()), 两者都错=float((~a & ~b).mean()),
                    神经网络错时物理对=float(b[~a].mean()), 物理错时神经网络对=float(a[~b].mean()))
out["互补性"] = comp
(case_dir / "breakdown.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

sc = "域内"
print("场景", sc)
for g, v in out[sc]["本方案"]["分组"].items():
    h = out[sc]["硬串联"]["分组"][g]
    print(f"{g:18s} n={v['n']:5d} 本方案 {v['class_acc']*100:5.1f}% MAE {v['d_mean']:.2f} 中位 {v['d_median']:.2f} | 硬串联 {h['class_acc']*100:5.1f}% MAE {h['d_mean']:.2f}")
print("分位数", out[sc]["本方案"]["分位数km"])
for g, v in out["过渡电阻曲线"].items():
    print(g, v["n"], {m: round(v[m]["class_acc"] * 100, 1) for m in METHODS})
