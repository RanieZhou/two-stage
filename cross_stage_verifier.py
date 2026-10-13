"""阶段三：区段等价类、贝叶斯联合打分与候选扩展。"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def equivalence_classes(pen, sec_true, delta=4.0, frac=0.5):
    """由物理证据定义区段等价类。

    真实故障在 s 时，若在不少于 frac 的样本里 s' 的惩罚残差不比 s 大出 delta，则 s' 与 s 无法区分；
    双向都无法区分的区段对相连，取连通分量。返回每个区段的类别号。
    """
    n_sec = pen.shape[1]
    ind = np.eye(n_sec, dtype=bool)
    for s in range(n_sec):
        m = sec_true == s
        if m.sum() >= 5:
            ind[s] = ((pen[m] - pen[m, s][:, None]) < delta).mean(0) > frac
    sym = ind & ind.T
    return connected_components(csr_matrix(sym), directed=False)[1]


def fit_lambda(p, pen, cls, sec_true):
    """在验证集上最大化真实等价类的联合后验似然，得到 S = ln P - λ·J_pen 中的 λ。"""
    logp = np.log(p + 1e-12)
    same = cls[None, :] == cls[sec_true][:, None]

    def nll(log_lam):
        s = logp - np.exp(log_lam) * pen
        return -(logsumexp(np.where(same, s, -np.inf), axis=1) - logsumexp(s, axis=1)).mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-9, 4), method="bounded").x))


@dataclass
class Thresholds:
    """阶段三参数，均由验证集标定。"""
    lam: float           # 联合打分中残差的权重 λ
    j_threshold: float   # 假设正确时残差的 99% 分位数，候选全部超过则视为候选集不可信


def decide(p, pen, jval, alpha, ftype, cls, th, expand=True, k=2):
    """逐样本联合决策。

    p: (N, C) 校准概率；pen/jval/alpha/ftype: (N, C) 各区段的物理反演结果（实际只读取候选区段及扩展后的区段）。
    先在阶段一概率最高的 k 个候选内按 S = ln P - λ·J_pen 取最大；若候选的残差全部超过 j_threshold，
    说明候选集不可信，扩展到全部区段重新打分；扩展后仍全部超标则标记模型失配告警。
    置信度 conf 为候选池内联合后验 softmax(S) 中落在所选区段所属等价类的概率。
    返回字典，rule 取值 0=Top-k 内决策，1=扩展候选池后决策，2=模型失配告警。
    """
    n = len(p)
    order = np.argsort(-p, axis=1)
    score = np.log(p + 1e-12) - th.lam * pen
    sec = np.zeros(n, dtype=int)
    rule = np.zeros(n, dtype=int)
    conf = np.zeros(n)
    for i in range(n):
        cand = order[i, :k]
        if expand and jval[i, cand].min() > th.j_threshold:
            cand = order[i]
            rule[i] = 2 if jval[i].min() > th.j_threshold else 1
        sec[i] = cand[int(score[i, cand].argmax())]
        post = np.exp(score[i, cand] - score[i, cand].max())
        conf[i] = post[cls[cand] == cls[sec[i]]].sum() / post.sum()
    idx = np.arange(n)
    return dict(sec=sec, rule=rule, conf=conf, alpha=alpha[idx, sec], ftype=ftype[idx, sec].astype(int))
