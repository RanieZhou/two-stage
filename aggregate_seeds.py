"""汇总多随机种子的结果：每个场景给出各方法类别准确率与本方案测距误差的均值与标准差（ddof=1）。

种子 0 为 results/improved/<case>/ 下的主结果，其余种子在 seeds/ 下；种子只改变各神经网络的初始化与小批量抽样，测试样本相同。
"""
import json
from pathlib import Path

import numpy as np

OURS, S1_TOP1, HARD = "本方案(联合打分+候选扩展)", "阶段一Top-K类别召回", "基准1 硬串联"
SKIP = {"thresholds", "equivalence_classes", "k", "config", "耗时_ms每样本"}


def stat(v):
    v = np.array(v, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=len(v), values=v.tolist())


for case in ("ieee34", "ieee123"):
    root = Path("results/improved") / case
    ours = [json.load(open(root / "evaluation.json"))] + [json.load(open(p)) for p in sorted((root / "seeds").glob("s*/evaluation.json"))]
    gcn = {m: [json.load(open(root / f"{m if m == 'gcn_delta' else 'gcn_paper'}.json"))["gcn_delta" if m == "gcn_delta" else "gcn"]]
              + [json.load(open(p))[m] for p in sorted((root / "seeds").glob(f"{m}_s*.json"))] for m in ("gcn", "gcn_delta")}
    out = {}
    for sc in [k for k in ours[0] if k not in SKIP]:
        out[sc] = {
            "本方案": stat([r[sc][OURS]["class_acc"] for r in ours]),
            "阶段一Top-1": stat([r[sc][S1_TOP1]["1"] for r in ours]),
            "硬串联": stat([r[sc][HARD]["class_acc"] for r in ours]),
            "本方案测距误差均值km": stat([r[sc][OURS]["d_mean"] for r in ours]),
        }
        for name, m in (("GCN(论文输入)", "gcn"), ("GCN(突变量输入)", "gcn_delta")):
            if sc in gcn[m][0]:
                out[sc][name] = stat([g[sc]["class_acc"] for g in gcn[m]])
    (root / "multi_seed.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(case, "种子数", {k: v["n"] for k, v in next(iter(out.values())).items()})
    for sc, o in out.items():
        print(f"{sc:14s}", " | ".join(f"{k} {v['mean']*100 if 'km' not in k else v['mean']:.2f}±{v['std']*100 if 'km' not in k else v['std']:.2f}" for k, v in o.items()))
