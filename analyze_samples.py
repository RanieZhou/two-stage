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
(case_dir / "breakdown.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

sc = "域内"
print("场景", sc)
for g, v in out[sc]["本方案"]["分组"].items():
    h = out[sc]["硬串联"]["分组"][g]
    print(f"{g:18s} n={v['n']:5d} 本方案 {v['class_acc']*100:5.1f}% MAE {v['d_mean']:.2f} 中位 {v['d_median']:.2f} | 硬串联 {h['class_acc']*100:5.1f}% MAE {h['d_mean']:.2f}")
print("分位数", out[sc]["本方案"]["分位数km"])
