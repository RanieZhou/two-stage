"""同分布内的分档对比：按故障类型、故障位置、过渡电阻、量测噪声档位，统计各方法的区段准确率（等价类 / 精确）与测距误差。

输入（均为 IEEE 34，种子 0）：
  results/improved/ieee34/samples.npz                 本方案、硬串联（神经网络 Top-1 + 物理定位）、物理穷举的逐样本结果（evaluate_pipeline.py --save-samples）
  results/improved/ieee34/stage1_swap/<名>/samples.npz 外部分类器作阶段一时的逐样本结果：硬串联 = 外部分类器 Top-1 + 物理定位
  results/improved/ieee34/baselines_{gcn,gat,ml}_samples.npz  外部分类器自身的逐样本预测区段；baselines_bsbl_samples.npz  块稀疏贝叶斯学习的区段与测距误差
外部分类器只输出区段，“区段中点”为取预测区段中点的测距误差，“+物理定位”为其 Top-1 区段接本方案阶段二的测距误差。
"""
import json
from pathlib import Path

import numpy as np

from ieee34_bus_impedance import CASES, Network
from simulate_dataset import FAULT_TYPES

R = Path("results/improved/ieee34")
cls = np.array(json.load(open(R / "evaluation.json"))["equivalence_classes"])
net = Network(case=CASES["ieee34"])
ours = np.load(R / "samples.npz")
gcn, gat, ml, bsbl = (np.load(R / f"baselines_{n}_samples.npz") for n in ("gcn", "gat", "ml", "bsbl"))
EXT = {"GCN 论文输入": ("gcn_paper", gcn, "gcn/{}/sec"), "GCN 突变量输入": ("gcn_delta", gcn, "gcn_delta/{}/sec"),
       "GAT 论文输入": ("gat_paper", gat, "gcn/{}/sec"), "GAT 突变量输入": ("gat_delta", gat, "gcn_delta/{}/sec"),
       "PCA+SVM": ("pca_svm", ml, "pca_svm/{}/sec"), "PCA+随机森林": ("pca_rf", ml, "pca_rf/{}/sec")}
swap = {name: np.load(R / "stage1_swap" / tag / "samples.npz") for name, (tag, _, _) in EXT.items()}
type_name = np.array([n for n, _ in FAULT_TYPES])
mid = lambda sec: np.array([net.fault_distance_km(s, 0.5) for s in sec])


def method_stats(scs, mask_of):
    """scs: 场景名列表（样本拼接）；mask_of(场景名) 返回该场景内属于本分组的布尔掩码。返回 {方法: 指标}。"""
    parts = {}
    for sc in scs:
        m = mask_of(sc)
        t = ours[f"{sc}/sec_true"][m]
        d_true = ours[f"{sc}/d_true"][m]
        rec = {"本方案": (ours[f"{sc}/本方案/sec"][m], ours[f"{sc}/本方案/d_err"][m], None),
               "DNN Top-1": (ours[f"{sc}/硬串联/sec"][m], ours[f"{sc}/硬串联/d_err"][m], None)}
        for name, (tag, src, key) in EXT.items():
            sec = src[key.format(sc)][:len(m)][m]   # 外部分类器的评测不截取，取与本方案相同的前若干个样本
            rec[name] = (sec, swap[name][f"{sc}/硬串联/d_err"][m], np.abs(mid(sec) - d_true))
        rec["块稀疏贝叶斯学习"] = (bsbl[f"bsbl/{sc}/sec"][:len(m)][m], bsbl[f"bsbl/{sc}/d_err"][:len(m)][m], None)
        for k, (sec, d_phys, d_mid) in rec.items():
            p = parts.setdefault(k, dict(t=[], sec=[], dp=[], dm=[]))
            p["t"].append(t); p["sec"].append(sec); p["dp"].append(d_phys); p["dm"].append(d_mid)
    out = {}
    for k, p in parts.items():
        t, sec = np.concatenate(p["t"]), np.concatenate(p["sec"])
        out[k] = dict(n=int(len(t)), class_acc=float((cls[sec] == cls[t]).mean()), section_acc=float((sec == t).mean()), d_mean=float(np.concatenate(p["dp"]).mean()))
        if p["dm"][0] is not None:
            out[k]["d_mean_midpoint"] = float(np.concatenate(p["dm"]).mean())
    return out


groups = {}
for t in ("SLG", "LL", "LLG", "LLLG"):
    groups[f"故障类型 {t}"] = method_stats(["域内"], lambda sc, t=t: type_name[ours[f"{sc}/ftype"]] == t)
groups["故障类型 全部"] = method_stats(["域内"], lambda sc: np.ones(len(ours[f"{sc}/sec_true"]), bool))
for lo, hi, lab in ((0.0, 0.35, "近端（α<0.35）"), (0.35, 0.65, "中段（0.35≤α<0.65）"), (0.65, 1.0, "远端（α≥0.65）")):
    groups[f"位置 {lab}"] = method_stats(["域内"], lambda sc, lo=lo, hi=hi: (ours[f"{sc}/alpha"] >= lo) & (ours[f"{sc}/alpha"] < hi))
for lo, hi in ((0, 1), (1, 10), (10, 100), (100, 300), (300, 500), (500, 1000.01)):
    sc = "域内" if hi <= 300 else "高阻外推"
    groups[f"Rf {lo:g}–{min(hi, 1000):g} Ω"] = method_stats([sc], lambda s, lo=lo, hi=hi: (ours[f"{s}/rf"] >= lo) & (ours[f"{s}/rf"] < hi))
for lab, sc in (("0.5", "噪声×0.5"), ("1", "域内"), ("2", "噪声×2"), ("3", "噪声×3"), ("5", "噪声×5")):
    groups[f"噪声 ×{lab}"] = method_stats([sc], lambda s: np.ones(len(ours[f"{s}/sec_true"]), bool))

(R / "tier1.json").write_text(json.dumps(groups, ensure_ascii=False, indent=1))
for g, v in groups.items():
    print(f"{g:22s}", " | ".join(f"{k} {x['class_acc']*100:.1f}/{x['section_acc']*100:.1f}/{x['d_mean']:.2f}" for k, x in v.items()))
