"""汇总多随机种子的结果：每个场景给出各方法类别准确率与本方案测距误差的均值与标准差（ddof=1）。

种子 0 为 results/improved/<case>/ 下的主结果，其余种子在 seeds/ 下（IEEE 123 的 GCN 在种子 0 时未运行三个模型偏差场景，这些场景只含种子 1 起的结果）；种子只改变各神经网络的初始化与小批量抽样，测试样本相同。
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
            vals = [g[sc]["class_acc"] for g in gcn[m] if sc in g]
            if vals:
                out[sc][name] = stat(vals)
    # GAT（论文输入、突变量输入）、PCA+SVM、PCA+随机森林：种子 0 为主结果，其余在 seeds/；块稀疏贝叶斯无随机性，单次
    extra = {"GAT(论文输入)": ("gat_paper", "gcn"), "GAT(突变量输入)": ("gat_delta", "gcn_delta")}
    for name, (stem, mode) in extra.items():
        files = [root / f"{stem}.json"] + sorted((root / "seeds").glob(f"{stem}_s*.json"))
        runs = [json.load(open(f))[mode] for f in files if f.exists()]
        for sc, o in out.items():
            vals = [r[sc]["class_acc"] for r in runs if sc in r]
            if vals:
                o[name] = stat(vals)
    for name, m in (("PCA+SVM", "pca_svm"), ("PCA+随机森林", "pca_rf")):
        files = [root / "baselines_ml.json"] + sorted((root / "seeds").glob("pca_s*.json"))
        runs = [json.load(open(f))[m] for f in files if f.exists()]
        for sc, o in out.items():
            vals = [r[sc]["class_acc"] for r in runs if sc in r]
            if vals:
                o[name] = stat(vals)
    if (root / "baselines_bsbl.json").exists():
        bs = json.load(open(root / "baselines_bsbl.json"))
        for sc, o in out.items():
            if sc in bs:
                o["块稀疏贝叶斯"] = stat([bs[sc]["class_acc"]])
    (root / "multi_seed.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(case, "种子数", {k: v["n"] for k, v in next(iter(out.values())).items()})
    for sc, o in out.items():
        print(f"{sc:14s}", " | ".join(f"{k} {v['mean']*100 if 'km' not in k else v['mean']:.2f}±{v['std']*100 if 'km' not in k else v['std']:.2f}" for k, v in o.items()))
