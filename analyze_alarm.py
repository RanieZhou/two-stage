"""失配告警与置信度的可靠性（逐样本，种子 0）：告警占比、告警样本中的错误率（告警精确率）、全部错误中被告警的比例（告警召回率），
以及置信度对判对与否的区分能力（AUROC）。错误按等价类判定。用法：python analyze_alarm.py [--case ieee34|ieee123]，输出 results/improved/<case>/alarm.json。
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["ieee34", "ieee123"], default="ieee34")
a = ap.parse_args()
R = Path("results/improved") / a.case
cls = np.array(json.load(open(R / "evaluation.json"))["equivalence_classes"])
S = np.load(R / "samples.npz")
scs = sorted({k.split("/")[0] for k in S.files})


def auroc(score, positive):
    """score 越大越应判为 positive 类的 AUROC（秩和法，并列取平均秩）。"""
    if positive.all() or not positive.any():
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        m = score == v
        ranks[m] = ranks[m].mean()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


out = {}
for sc in scs:
    if f"{sc}/本方案/rule" not in S.files:
        continue
    ok = cls[S[f"{sc}/本方案/sec"]] == cls[S[f"{sc}/sec_true"]]
    rule, conf = S[f"{sc}/本方案/rule"], S[f"{sc}/本方案/conf"]
    alarm, expand = rule == 2, rule == 1
    out[sc] = dict(
        n=len(ok), 准确率=float(ok.mean()), 告警占比=float(alarm.mean()), 扩展占比=float(expand.mean()),
        告警样本错误率=float((~ok[alarm]).mean()) if alarm.any() else None,
        告警召回率=float(alarm[~ok].mean()) if (~ok).any() else None,
        扩展样本准确率=float(ok[expand].mean()) if expand.any() else None,
        置信度AUROC=auroc(conf, ok),
        低置信占比=float((conf < 0.6).mean()), 低置信样本错误率=float((~ok[conf < 0.6]).mean()) if (conf < 0.6).any() else None,
        低置信召回率=float((conf < 0.6)[~ok].mean()) if (~ok).any() else None,
    )
(R / "alarm.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for sc, v in out.items():
    f = lambda x: "-" if x is None else f"{x * 100:.1f}"
    print(f"{sc:16s} 准确率{f(v['准确率'])} 告警{f(v['告警占比'])}% 告警错误率{f(v['告警样本错误率'])} 召回{f(v['告警召回率'])} | 低置信{f(v['低置信占比'])}% 错误率{f(v['低置信样本错误率'])} 召回{f(v['低置信召回率'])} AUROC {v['置信度AUROC']:.3f}")
