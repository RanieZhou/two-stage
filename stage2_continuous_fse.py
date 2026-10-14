"""阶段二：候选区段内的连续位置估计与加权物理残差。

对区段 s、故障类型 τ、位置 α，测点突变量 z 应落在 G_τ(s, α) 的列空间内（G 为故障相单位注入电流的响应）。
PhysicsModel：给定 α 后注入电流为自由复数的线性最小二乘，对 α 网格逐点求解后取最小残差并做抛物线细化。
ConstrainedPhysicsModel：过渡电阻为实数，故障电流只剩 R 一个实参数（再乘带先验的实增益），在 (α, R) 网格上求最小残差。
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


# ---- 过渡电阻为实数的约束模型 ----
RF_GRID = np.logspace(-2, np.log10(3000.0), 48)   # Ω
RF_FINE = 13                                       # 粗网格最优点两侧各一格内的细搜索点数


def fault_current(zff, vf, ftype, phases, rf):
    """过渡电阻 R 为实数时的故障电流（注入网络的电流，形状 (..., p)，p 与 effective_columns 的列数一致）。

    响应库中的单位电流是注入网络的电流，故障电流从网络流出，故注入电流 I = −I_f。故障点电压 V_f = V_f,pre + Z_ff I = −R I，
    得 I(R) = −(R·E + Z_ff)^-1 V_f,pre，E 为单位阵；
    相间故障（LL）只有一个相间电阻，取 I = −(V_a − V_b) / (R + Z_aa + Z_bb − Z_ab − Z_ba)。
    zff: (..., c, c)，vf: (..., c)，rf: 与前者的批维广播的 (...)。
    """
    kind, ph = FAULT_TYPES[ftype]
    cols = [phases.index(p) for p in ph]
    if kind == "LL":
        a, b = cols
        z = zff[..., a, a] + zff[..., b, b] - zff[..., a, b] - zff[..., b, a]
        return (-(vf[..., a] - vf[..., b]) / (rf + z))[..., None]
    zs = zff[..., cols, :][..., :, cols]
    m = rf[..., None, None] * np.eye(len(cols)) + zs
    return -np.linalg.solve(m, vf[..., cols][..., None])[..., 0]


class ConstrainedPhysicsModel:
    """阶段二的物理模型，过渡电阻取实数：每个 (区段, 故障类型, α) 的故障电流只剩 R 一个实参数。

    与 PhysicsModel 的自由复电流最小二乘相比，J(R) = z^H W z − 2 Re(I(R)^H b) + I(R)^H M I(R)，b = G^H W z，M = G^H W G；
    先在 α 网格 × R 粗网格上求最小，再对最优 α 处的 R 在粗网格相邻两格内细搜索。惩罚残差为 J + 2（一个实参数）+ 阻抗先验。

    故障点故障前电压取名义负荷下的值，实际负荷水平与调压器档位使其幅值有偏差：故障电流再乘一个实增益 g，
    g 带先验 ((g − 1)/amp_sigma)²，对 g 的最小化有闭式解（amp_sigma 为 None 时 g 固定为 1）。
    """

    def __init__(self, net, libs, zffs, prior_pen, rows, sigma_noise, eps_rel, eps_abs=0.0, alphas=ALPHAS, rf=RF_GRID, amp_sigma=0.05):
        self.net, self.alphas, self.rf = net, alphas, rf
        self.fixed_gain = amp_sigma is None
        self.amp_prec = 0.0 if amp_sigma is None else 1.0 / amp_sigma ** 2
        self.sigma2 = sigma_noise ** 2
        self.eps_rel2, self.eps_abs2 = eps_rel ** 2, eps_abs ** 2
        self.is_v = rows < len(net.nodes)
        self.hyp = []
        for k, s in enumerate(net.sections):
            vf = net.prefault_fault_voltage(k, alphas)
            per_type = []
            for lib, zff, extra in zip(libs, zffs, prior_pen):
                ls = lib[k][:, rows, :]
                for t in valid_types(s["phases"]):
                    g = effective_columns(ls, s["phases"], t)
                    j, m, p = g.shape
                    gg = np.einsum("jmp,jmq->mjpq", g.conj(), g).reshape(m, -1)
                    gc = np.einsum("jmp->mjp", g.conj()).reshape(m, -1)
                    ik = fault_current(zff[k][:, None], vf[:, None], t, s["phases"], rf[None, :])   # (J, K, p)
                    ikc = ik.conj().transpose(0, 2, 1)                                              # (J, p, K)
                    outer = np.einsum("jkp,jkq->jpqk", ik.conj(), ik).reshape(j, p * p, -1)        # (J, p², K)
                    per_type.append((t, p, gg, gc, extra, ikc, outer, zff[k], vf))
            self.hyp.append(per_type)

    def _gain_residual(self, z2, cross, quad):
        """故障电流乘实增益 g 后的数据残差 z2 − 2g·cross + g²·quad 及最优 g（含先验的闭式解）。"""
        g = np.ones_like(cross) if self.fixed_gain else (cross + self.amp_prec) / (quad + self.amp_prec + 1e-300)
        return np.maximum(z2 - 2 * g * cross + g * g * quad, 0.0), g

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
        idx = np.arange(n)
        phases = self.net.sections[sec]["phases"]
        lr = np.log(self.rf)
        for t, p, gg, gc, extra, ikc, outer, zff, vf in self.hyp[sec]:
            j = len(self.alphas)
            mmat = (w2 @ gg).reshape(n, j, p * p)
            b = (wz @ gc).reshape(n, j, p)
            # J(R) = z^H W z − 2 Re(I(R)^H b) + I(R)^H M I(R)，I(R) 为约束下的故障电流轨迹
            cross = np.matmul(b.transpose(1, 0, 2), ikc).real
            quad = np.matmul(mmat.transpose(1, 0, 2), outer).real
            jr, gain = self._gain_residual(z2[None, :, None], cross, quad)                  # (J, n, K)
            tot = (jr + self.amp_prec * (gain - 1) ** 2).transpose(1, 0, 2)                  # (n, J, K) 含增益先验
            jr, gain = jr.transpose(1, 0, 2), gain.transpose(1, 0, 2)
            kb = tot.argmin(2)[..., None]
            tot_c = np.take_along_axis(tot, kb, 2)[..., 0]                                   # 各 α 处最优 R 的总残差 (n, J)
            ia = tot_c.argmin(1)
            # 最优 α 处沿 R 细搜索
            k0 = np.clip(kb[idx, ia, 0], 1, len(self.rf) - 2)
            rf_f = np.exp(lr[k0 - 1][:, None] + (lr[k0 + 1] - lr[k0 - 1])[:, None] * np.linspace(0, 1, RF_FINE)[None, :])
            ik = fault_current(zff[ia][:, None], vf[ia][:, None], t, phases, rf_f)           # (n, F, p)
            msel = mmat[idx, ia].reshape(n, p, p)
            cr = (ik.conj() * b[idx, ia][:, None, :]).sum(-1).real
            qd = (ik.conj() * np.matmul(ik, np.swapaxes(msel, -1, -2))).sum(-1).real
            jf, gf = self._gain_residual(z2[:, None], cr, qd)                                # (n, F)
            kf = (jf + self.amp_prec * (gf - 1) ** 2).argmin(1)
            jc_data = np.take_along_axis(jr, kb, 2)[..., 0][idx, ia]
            gc_best = np.take_along_axis(gain, kb, 2)[..., 0][idx, ia]
            use_fine = jf[idx, kf] + self.amp_prec * (gf[idx, kf] - 1) ** 2 < tot_c[idx, ia]
            jmin = np.where(use_fine, jf[idx, kf], jc_data)
            g_best = np.where(use_fine, gf[idx, kf], gc_best)
            pen = jmin + self.amp_prec * (g_best - 1) ** 2 + 2.0 + extra
            upd = pen < best_pen
            best_pen[upd] = pen[upd]
            best[0][upd] = jmin[upd]
            best[1][upd] = _refine_alpha(tot_c[upd], ia[upd], self.alphas)
            best[2][upd] = t
        return best_pen, best[0], best[1], best[2]

    fit_candidates = PhysicsModel.fit_candidates
    fit_all = PhysicsModel.fit_all
