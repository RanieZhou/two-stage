"""阶段二：候选区段内的连续位置估计与加权物理残差。

对区段 s、故障类型 τ、位置 α，测点突变量 z 应落在 G_τ(s, α) 的列空间内（G 为故障相单位注入电流的响应）。
给定 α 后注入电流为复数线性最小二乘，对 α 网格逐点求解后取最小残差并做抛物线细化。
"""
import numpy as np

from simulate_dataset import FAULT_TYPES, valid_types

ALPHAS = np.linspace(0.02, 0.98, 49)


def effective_columns(lib_s, phases, ftype):
    """把区段响应库 (J, m, c) 按故障类型整理为 (J, m, p)：SLG 取单列，LL 取两列之差，LLG/LLLG 各故障相自由。"""
    kind, ph = FAULT_TYPES[ftype]
    cols = [phases.index(p) for p in ph]
    if kind == "LL":
        return (lib_s[:, :, cols[0]] - lib_s[:, :, cols[1]])[:, :, None]
    return lib_s[:, :, cols]


class PhysicsModel:
    """持有所选测点行上的响应库，对一批量测同时求各区段的最小加权残差。"""

    def __init__(self, net, libs, prior_pen, rows, sigma_noise, eps_rel, eps_abs=0.0, alphas=ALPHAS):
        """libs 为不同线路阻抗假设（如 0.9/1.0/1.1 倍）下的响应库列表，prior_pen 为各假设的先验惩罚（加在惩罚残差上），
        每个假设都作为备选，取惩罚残差最小者；报告的残差 J 不含先验惩罚。

        噪声方差之外，模型误差方差为 (eps_rel·|z_i|)² + (eps_abs·同类量测的最大幅值)²。
        """
        self.net = net
        self.alphas = alphas
        self.sigma2 = sigma_noise ** 2
        self.eps_rel2 = eps_rel ** 2
        self.eps_abs2 = eps_abs ** 2
        self.is_v = rows < len(net.nodes)
        self.hyp = []
        for k, s in enumerate(net.sections):
            per_type = []
            for lib, extra in zip(libs, prior_pen):
                ls = lib[k][:, rows, :]
                for t in valid_types(s["phases"]):
                    g = effective_columns(ls, s["phases"], t)
                    # 法方程所需的 conj(G_p) G_q，形状 (m, J*p*p)；以及 conj(G)，形状 (m, J*p)
                    j, m, p = g.shape
                    gg = np.einsum("jmp,jmq->mjpq", g.conj(), g).reshape(m, -1)
                    gc = np.einsum("jmp->mjp", g.conj()).reshape(m, -1)
                    per_type.append((t, p, gg, gc, extra))
            self.hyp.append(per_type)

    def fit_section(self, sec, z):
        """z: (N, m) 复数量测。返回 (J_pen, J, alpha, ftype)，均为长度 N 的数组。"""
        n = len(z)
        az = np.abs(z)
        zmax = np.zeros_like(az)
        for kind in (self.is_v, ~self.is_v):
            if kind.any():
                zmax[:, kind] = az[:, kind].max(1)[:, None]
        w2 = 1.0 / (self.sigma2[None, :] + self.eps_rel2 * az ** 2 + self.eps_abs2 * zmax ** 2)
        best_pen = np.full(n, np.inf)
        best = [np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)]
        z2 = (w2 * np.abs(z) ** 2).sum(1)
        wz = w2 * z
        for t, p, gg, gc, extra in self.hyp[sec]:
            j = len(self.alphas)
            mmat = (w2 @ gg).reshape(n, j, p, p)
            b = (wz @ gc).reshape(n, j, p)
            ridge = 1e-9 * np.einsum("njpp->nj", mmat).real + 1e-300
            x = np.linalg.solve(mmat + ridge[..., None, None] * np.eye(p), b[..., None])[..., 0]
            jres = np.maximum(z2[:, None] - np.einsum("njp,njp->nj", b.conj(), x).real, 0.0)
            ia = jres.argmin(1)
            jmin = jres[np.arange(n), ia]
            pen = jmin + 4.0 * p + extra
            upd = pen < best_pen
            best_pen[upd] = pen[upd]
            best[0][upd] = jmin[upd]
            best[1][upd] = _refine_alpha(jres[upd], ia[upd], self.alphas)
            best[2][upd] = t
        return best_pen, best[0], best[1], best[2]

    def fit_candidates(self, z, cand):
        """只对每个样本的候选区段 cand (N, K) 求解，返回 (J_pen, J, alpha, ftype)，形状 (N, K)。"""
        outs = [np.zeros(cand.shape) for _ in range(4)]
        for sec in np.unique(cand):
            rows, cols = np.where(cand == sec)
            r = self.fit_section(sec, z[rows])
            for o, v in zip(outs, r):
                o[rows, cols] = v
        return outs

    def fit_all(self, z, chunk=2000):
        """对全部区段求解，返回 (J_pen, J, alpha, ftype)，形状 (N, 区段数)。"""
        outs = [np.zeros((len(z), len(self.hyp))) for _ in range(4)]
        for lo in range(0, len(z), chunk):
            for s in range(len(self.hyp)):
                r = self.fit_section(s, z[lo:lo + chunk])
                for o, v in zip(outs, r):
                    o[lo:lo + chunk, s] = v
        return outs


def _refine_alpha(jres, ia, alphas=ALPHAS):
    """对残差曲线最小点做抛物线插值，得到连续 α。"""
    n, jn = jres.shape
    i0 = np.clip(ia, 1, jn - 2)
    y0, y1, y2 = jres[np.arange(n), i0 - 1], jres[np.arange(n), i0], jres[np.arange(n), i0 + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(den > 1e-12, 0.5 * (y0 - y2) / np.where(den > 1e-12, den, 1.0), 0.0)
    off = np.clip(off, -1, 1)
    step = alphas[1] - alphas[0]
    edge = (ia == 0) | (ia == jn - 1)
    return np.where(edge, alphas[ia], np.clip(alphas[i0] + off * step, alphas[0], alphas[-1]))
