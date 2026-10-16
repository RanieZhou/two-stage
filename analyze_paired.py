"""核心场景下本方法与“其余方法中均值最高者”的配对比较（IEEE 34，种子 0，同一批 5000 个测试样本）。

最高者按 multi_seed.json 的类准确率均值确定（GCN、GAT 的两种输入都参与，即对基线取测试集上较好的输入，对本方法偏保守）。
对类准确率和精确区段准确率分别给出：差值（百分点）、配对 bootstrap 95% 置信区间（2000 次）、McNemar 精确检验的双侧 p 值。
输出 results/improved/ieee34/paired_tests.json。
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

R = Path("results/improved/ieee34")
CORE5 = ["域内", "负荷偏移", "高阻外推", "噪声×3", "个别线路阻抗±20%"]
MS = {"gcn_p": "GCN(论文输入)", "gcn_d": "GCN(突变量输入)", "gat_p": "GAT(论文输入)", "gat_d": "GAT(突变量输入)", "svm": "PCA+SVM", "rf": "PCA+随机森林", "bsbl": "块稀疏贝叶斯"}
TAG = {"gcn_p": "gcn_paper", "gcn_d": "gcn_delta", "gat_p": "gat_paper", "gat_d": "gat_delta", "svm": "pca_svm", "rf": "pca_rf"}

cls = np.array(json.load(open(R / "evaluation.json"))["equivalence_classes"])
ours = np.load(R / "samples.npz")
swaps = {k: np.load(R / "stage1_swap" / v / "samples.npz") for k, v in TAG.items()}
bsbl = np.load(R / "baselines_bsbl_samples.npz")
ms = json.load(open(R / "multi_seed.json"))
rng = np.random.default_rng(0)


def paired(a, b):
    """a、b 为同一批样本上两种方法是否判对的布尔数组。"""
    n = len(a)
    idx = rng.integers(0, n, (2000, n))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return dict(diff=float((a.mean() - b.mean()) * 100), lo=float(np.percentile(diffs, 2.5) * 100), hi=float(np.percentile(diffs, 97.5) * 100), p=float(p), ours_only=n10, other_only=n01)


out = {}
for sc in CORE5:
    kb = max(MS, key=lambda k: ms[sc][MS[k]]["mean"])
    st = ours[f"{sc}/sec_true"]
    n = len(st)
    other = bsbl[f"bsbl/{sc}/sec"][:n] if kb == "bsbl" else swaps[kb][f"{sc}/硬串联/sec"]
    mine = ours[f"{sc}/本方案/sec"]
    out[sc] = dict(best_other=kb, n=n,
                   ours_class=float((cls[mine] == cls[st]).mean()), other_class=float((cls[other] == cls[st]).mean()),
                   ours_exact=float((mine == st).mean()), other_exact=float((other == st).mean()),
                   class_test=paired(cls[mine] == cls[st], cls[other] == cls[st]), exact_test=paired(mine == st, other == st))
    r = out[sc]
    print(sc, kb, f"类 {r['ours_class']*100:.1f} vs {r['other_class']*100:.1f}", {k: round(v, 3) for k, v in r["class_test"].items()}, "| 精确", {k: round(v, 3) for k, v in r["exact_test"].items() if k in ("diff", "lo", "hi", "p")})
(R / "paired_tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
