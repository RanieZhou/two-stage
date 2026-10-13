"""阶段三：区段等价类、贝叶斯联合打分与防误改三区规则。"""
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
    """防误改规则的阈值，均由验证集标定。"""
    lam: float           # 联合打分中残差的权重 λ
    eps_normal: float    # 假设正确时残差的 95% 分位数
    j_threshold: float   # 假设正确时残差的 99% 分位数
    tau_b: float         # 规则 B 的残差差值阈值
    j_ref: float         # 置信度归一化用的典型残差


def decide(p, order, pen, jval, alpha, ftype, cls, th, p_a=0.85, rules="ABC", k_fixed=None):
    """逐样本联合决策。

    p: (N, C) 校准概率；order: (N, 6) 概率降序的区段序号；pen/jval/alpha/ftype: (N, C) 物理反演结果；
    cls: (C,) 区段等价类号；rules 选择启用的规则；k_fixed 不为空时候选数固定。返回字典，rule 取值 0=联合打分，1=规则A，2=规则B，3=规则C 触发后重新打分，4=模型失配告警。
    """
    n = len(p)
    k = np.where(p[np.arange(n), order[:, 0]] >= 0.5, 2, np.where(p[np.arange(n), order[:, 0]] >= 0.25, 3, 4))
    if k_fixed:
        k = np.full(n, k_fixed)
    sec = np.zeros(n, dtype=int)
    rule = np.zeros(n, dtype=int)
    conf = np.zeros(n)
    for i in range(n):
        cand = order[i]
        c1 = cand[0]
        score = np.log(p[i, cand] + 1e-12) - th.lam * pen[i, cand]
        kk = k[i]
        if "A" in rules and p[i, c1] > p_a and jval[i, c1] < th.eps_normal:
            pick, r = 0, 1
        elif "C" in rules and jval[i, cand[:kk]].min() > th.j_threshold:
            # 规则 C：扩展到 Top-6 重新打分；仍全部超标则告警
            pick = int(score.argmax())
            r = 4 if jval[i, cand].min() > th.j_threshold else 3
        else:
            pick, r = int(score[:kk].argmax()), 0
            alt = [j for j in range(1, kk) if cls[cand[j]] != cls[c1]]
            if "B" in rules and alt and jval[i, c1] > th.eps_normal:
                j2 = alt[int(np.argmin(jval[i, cand[alt]]))]
                if jval[i, cand[j2]] < th.eps_normal and jval[i, c1] - jval[i, cand[j2]] > th.tau_b:
                    pick, r = j2, 2
        s = cand[pick]
        sec[i], rule[i] = s, r
        conf[i] = p[i, s] * np.exp(-jval[i, s] / th.j_ref)
    idx = np.arange(n)
    return dict(sec=sec, rule=rule, conf=conf, alpha=alpha[idx, sec], ftype=ftype[idx, sec].astype(int), k=k)
