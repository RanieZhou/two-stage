"""对比方法 L1：Liang 等，Efficient Two-Stage Fault Localization for Distribution Networks With Optimally Placed μPMUs，
IEEE Trans. Smart Grid, 2025, 16(6): 5185-5199。

阶段一：虚拟电流线性模型匹配。对每条线路用其两个端点母线的虚拟电流重构 μPMU 母线的电压突变量，
        匹配指数为最小二乘残差，升序取前 k 条线路为候选。
阶段二：对候选线路端点母线逐个做状态估计匹配。伪量测为故障前功率注入（按负荷模型迭代换算为等效电流），
        故障电流由根母线电流补偿，匹配指数为量测残差一范数加末端母线电压幅值偏差；选出最匹配的母线后，
        把相邻线路分段（虚拟零注入母线）重新匹配以细化故障位置。

本复现按论文的观测配置：μPMU 取本方案相同的 5 个测点母线（只用其电压相量）；另外按论文需要提供
根母线三相电压与电流相量、末端母线电压幅值、全网故障前功率注入伪量测。因此 L1 获得的量测多于本方案。
"""
from multiprocessing import Pool

import numpy as np
from threadpoolctl import threadpool_limits

from ieee34_bus_impedance import NOISE_I_ABS, NOISE_V_REL, Network

ALPHA_GRID = np.linspace(0.1, 0.9, 9)   # 相邻线路分段位置（虚拟母线）
TOP_K = 5                                # 阶段一候选线路数
E_INJ = 0.20                             # 伪量测相对误差（论文取 5%–50%），按 e/3 为标准差
E_VMAG = 0.001                           # 末端母线电压幅值相对误差，按 e/3 为标准差
VMIN = 0.85                              # 负荷电压低于此值（标幺）时按恒阻抗处理（OpenDSS vminpu）
ZERO_INJ_SIGMA = 0.01                    # 无负荷母线的零注入伪量测标准差（A）
EXCLUDED_LEAF = ("sourcebus", "800")


def load_factor(model, x):
    """负荷功率随电压标幺值 x 的变化系数：恒功率 1、恒阻抗 x²、恒电流 x，低于 VMIN 按恒阻抗。"""
    if model == 2:
        return x ** 2
    if model == 5:
        return np.where(x >= VMIN, x, VMIN * (x / VMIN) ** 2)
    return np.where(x >= VMIN, 1.0, (x / VMIN) ** 2)


class L1Locator:
    def __init__(self, net: Network, pmu_nodes):
        """pmu_nodes 为 μPMU 所在节点名列表（各母线的各相）。"""
        self.net = net
        self.nodes = net.nodes
        idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx = idx
        is_slack = np.array([n.startswith("sourcebus") or n.startswith("800.") for n in self.nodes])
        self.slack = np.where(is_slack)[0]
        self.free = np.where(~is_slack)[0]
        self.pos_free = {int(n): k for k, n in enumerate(self.free)}   # 节点序号 -> 自由节点位置
        # 阶段一：Z = Y^-1（含恒阻抗负荷，根母线为参考），观测行为 μPMU 母线电压
        self.pmu_idx = np.array([idx[n] for n in pmu_nodes])
        y, _, _ = net.assemble()
        z = np.linalg.inv(y[np.ix_(self.free, self.free)])
        self.stage1_pinv = []
        for s in net.sections:
            cols = [self.pos_free[idx[n]] for n in s["nodes"] if idx[n] in self.pos_free]
            a = z[np.ix_([self.pos_free[idx[n]] for n in pmu_nodes], cols)]
            self.stage1_pinv.append((a, np.linalg.pinv(a, rcond=1e-8)))
        # 负荷支路
        br = net.load_branches
        self.br_a = np.array([idx[b["a"]] for b in br])
        self.br_b = np.array([idx[b["b"]] if b["b"] else -1 for b in br])
        self.br_s = np.array([b["s"] for b in br])
        self.br_vn = np.sqrt(np.array([b["v2"] for b in br]))
        self.br_model = np.array([b["model"] for b in br])
        # 末端母线
        leaf_buses = {b for b, v in net.adj.items() if len({x for x, _ in v}) == 1 and b not in EXCLUDED_LEAF}
        self.leaf = np.array([i for i, n in enumerate(self.nodes) if n.split(".")[0] in leaf_buses])
        # 各母线节点
        self.bus_nodes = {}
        for i, n in enumerate(self.nodes):
            self.bus_nodes.setdefault(n.split(".")[0], []).append(i)
        self.phase_of = np.array([int(n.split(".")[1]) for n in self.nodes])
        # 状态估计变体：母线级（不拆线）与各线路各位置的虚拟母线
        v_nom = net.v_pre
        self.variants = {None: self._variant(None, v_nom)}
        for k in range(len(net.sections)):
            for a in ALPHA_GRID:
                self.variants[(k, round(float(a), 3))] = self._variant((k, a), v_nom)

    def _variant(self, split, v_nom):
        """构造一个网络变体的加权最小二乘伪逆。ECM 权重取名义工作点注入电流幅值的 e/3。"""
        y, nodes, fx = self.net.assemble(split=split, loads=False)
        n_base = len(self.nodes)
        free = list(self.free) + list(range(n_base, len(nodes)))
        slack = list(self.slack)
        ynn = y[np.ix_(free, free)]
        yns = y[np.ix_(free, slack)]
        i_nom = ynn[:, : len(self.free)] @ v_nom[self.free] + yns @ v_nom[slack]
        sig_i = np.maximum(E_INJ / 3 * np.abs(i_nom), ZERO_INJ_SIGMA)
        sig_v = np.maximum(NOISE_V_REL * np.abs(v_nom[self.pmu_idx]), 1e-3)
        sel = np.zeros((len(self.pmu_idx), len(free)))
        for r, node in enumerate(self.pmu_idx):
            sel[r, self.pos_free[int(node)]] = 1.0
        a_aug = np.vstack([ynn / sig_i[:, None], sel / sig_v[:, None]])
        return dict(ynn=ynn, yns=yns, pinv=np.linalg.pinv(a_aug, rcond=1e-10), sig_i=sig_i, sig_v=sig_v,
                    sel=sel, fx=[free.index(f) for f in fx], n_free=len(free))

    # ---------- 阶段一 ----------
    def stage1(self, dv_pmu):
        """dv_pmu: (N, m_pmu) 带噪电压突变量。返回 (N, 区段数) 匹配指数。"""
        mi = np.zeros((len(dv_pmu), len(self.net.sections)))
        for k, (a, ap) in enumerate(self.stage1_pinv):
            res = dv_pmu - (dv_pmu @ ap.T) @ a.T
            mi[:, k] = (np.abs(res) ** 2).sum(1)
        return mi

    # ---------- 阶段二 ----------
    def _load_inj(self, v_full, s_pre, x_pre):
        """给定全部节点电压，返回负荷引起的节点电流注入（负荷电流的反号）。"""
        va = v_full[self.br_a]
        vb = np.where(self.br_b >= 0, v_full[np.maximum(self.br_b, 0)], 0.0)
        vab = va - vb
        x = np.abs(vab) / self.br_vn
        s = np.empty_like(s_pre)
        for m in np.unique(self.br_model):
            msk = self.br_model == m
            s[msk] = s_pre[msk] * load_factor(m, x[msk]) / load_factor(m, x_pre[msk])
        cur = np.conj(s / vab)
        inj = np.zeros(len(self.nodes), dtype=complex)
        np.add.at(inj, self.br_a, -cur)
        mask = self.br_b >= 0
        np.add.at(inj, self.br_b[mask], cur[mask])
        return inj

    def _solve(self, var, cand_nodes, sample, n_iter=5):
        """在假定故障位于 cand_nodes（节点序号，变体自由节点内的位置）时做状态估计，返回匹配指数。"""
        v_full = sample["v_init"].copy()
        for _ in range(n_iter):
            inj = self._load_inj(v_full, sample["s_pre"], sample["x_pre"])
            # 故障电流补偿：各相故障电流 = 根母线电流 + 该相全部负荷节点注入之和
            i_f = {p: sample["i_root"][p - 1] + inj[self.phase_of == p].sum() for p in (1, 2, 3)}
            rhs_i = np.zeros(var["n_free"], dtype=complex)
            rhs_i[: len(self.free)] = inj[self.free]
            for pos, node_phase in cand_nodes:
                rhs_i[pos] -= i_f[node_phase]
            b_i = (rhs_i - var["yns"] @ sample["v_slack"]) / var["sig_i"]
            b_v = sample["v_pmu"] / var["sig_v"]
            v_free = var["pinv"] @ np.concatenate([b_i, b_v])
            v_full[self.free] = v_free[: len(self.free)]
            v_full[self.slack] = sample["v_slack"]
        r_i = rhs_i - (var["ynn"] @ v_free + var["yns"] @ sample["v_slack"])
        r_v = sample["v_pmu"] - var["sel"] @ v_free
        r_mag = sample["leaf_mag"] - np.abs(v_full[self.leaf])
        return (np.abs(r_i.real) + np.abs(r_i.imag)).sum() + (np.abs(r_v.real) + np.abs(r_v.imag)).sum() + np.abs(r_mag).sum()

    def _refine(self, lines, sample):
        """在给定线路上逐位置放虚拟零注入母线并状态估计匹配，返回 (最小匹配指数, 区段序号, α)。"""
        best = (np.inf, None, None)
        for k in lines:
            s = self.net.sections[k]
            js = []
            for a in ALPHA_GRID:
                var = self.variants[(k, round(float(a), 3))]
                js.append(self._solve(var, list(zip(var["fx"], s["phases"])), sample))
            js = np.array(js)
            i = int(js.argmin())
            if js[i] < best[0]:
                a_hat = ALPHA_GRID[i]
                if 0 < i < len(js) - 1:
                    den = js[i - 1] - 2 * js[i] + js[i + 1]
                    if den > 0:
                        a_hat = np.clip(a_hat + 0.5 * (js[i - 1] - js[i + 1]) / den * (ALPHA_GRID[1] - ALPHA_GRID[0]), 0.02, 0.98)
                best = (js[i], k, float(a_hat))
        return best

    def locate(self, sample, mi, literal=True):
        """阶段二：由阶段一匹配指数 mi (区段数,) 与样本量测给出 (区段序号, α)。

        literal=True 为论文流程：先在候选线路的端点母线上做状态估计选出最匹配母线，再细化其相邻线路；
        literal=False 把分段细化直接用于全部候选线路（线路较长时端点母线近似不成立，细化更可靠）。
        """
        top = np.argsort(mi)[:TOP_K]
        if not literal:
            _, k, a = self._refine(top, sample)
            return k, a
        buses = []
        for k in top:
            for b in (self.net.sections[k]["bus1"], self.net.sections[k]["bus2"]):
                if b not in ("800", "sourcebus") and b not in buses:
                    buses.append(b)
        var0 = self.variants[None]
        j_bus = {}
        for b in buses:
            cand = [(self.pos_free[i], int(self.phase_of[i])) for i in self.bus_nodes[b] if i in self.pos_free]
            j_bus[b] = self._solve(var0, cand, sample)
        best_bus = min(j_bus, key=j_bus.get)
        adjacent = [k for k, s in enumerate(self.net.sections) if best_bus in (s["bus1"], s["bus2"])]
        _, k, a = self._refine(adjacent, sample)
        return k, a

    def make_sample(self, rng, d, n, noise_scale=1.0):
        """由数据集第 n 条样本构造 L1 所需量测：μPMU 与根母线相量带噪声，末端电压幅值与伪量测带噪声。"""
        net = self.net
        v_post = d["v0"][n].astype(complex) + d["dv"][n].astype(complex)
        sig_v = noise_scale * NOISE_V_REL * np.abs(net.v_pre)

        def noisy_v(i):
            return v_post[i] + sig_v[i] * (rng.standard_normal(len(i)) + 1j * rng.standard_normal(len(i)))

        layout = net.layout()[1]
        i_root = (d["i0"][n].astype(complex) + d["di"][n].astype(complex))[layout[0][0]: layout[0][0] + 3]
        i_root = i_root + noise_scale * NOISE_I_ABS * (rng.standard_normal(3) + 1j * rng.standard_normal(3))
        mult = d["load_mult"][n]
        s_pre = self.br_s * mult * (1 + E_INJ / 3 * rng.standard_normal(len(self.br_s)))
        v0 = d["v0"][n].astype(complex)
        vb0 = np.where(self.br_b >= 0, v0[np.maximum(self.br_b, 0)], 0.0)
        x_pre = np.abs(v0[self.br_a] - vb0) / self.br_vn
        return dict(
            v_pmu=noisy_v(self.pmu_idx), v_slack=noisy_v(self.slack), i_root=i_root,
            leaf_mag=np.abs(v_post[self.leaf]) * (1 + E_VMAG / 3 * rng.standard_normal(len(self.leaf))),
            s_pre=s_pre, x_pre=x_pre, v_init=v0.copy(),
        )


_job = None


def _locate_one(n):
    loc, d, mi, noise_scale, seed = _job
    rng = np.random.default_rng(seed + n)
    sample = loc.make_sample(rng, d, n, noise_scale)
    k1, a1 = loc.locate(sample, mi[n], literal=True)
    k2, a2 = loc.locate(sample, mi[n], literal=False)
    return k1, a1, k2, a2


def run_l1(loc, d, mi, noise_scale, seed=0, workers=4):
    """对数据集 d（npz）的每个样本运行 L1 两种阶段二流程，返回 (区段, α) 数组：论文流程与全候选线路细化。"""
    global _job
    _job = (loc, d, mi, noise_scale, seed)
    with Pool(workers, initializer=threadpool_limits, initargs=(1,)) as pool:  # 每个进程单线程，避免 BLAS 线程互相抢占
        out = pool.map(_locate_one, range(len(mi)), chunksize=50)
    k1, a1, k2, a2 = (np.array(x) for x in zip(*out))
    return k1.astype(int), a1, k2.astype(int), a2


def greedy_pmu_placement(net, dv, sec_true, n_pmu=5, k=TOP_K, seed=0):
    """按 L1 的布点思路（线性模型匹配的前 k 命中率最大）贪心前向选取 n_pmu 个 μPMU 母线，返回节点名列表。

    论文用遗传算法求解同一目标；这里用贪心近似。dv 为全网节点电压突变量（无噪声），内部叠加量测噪声。
    """
    rng = np.random.default_rng(seed)
    nodes = net.nodes
    free = [i for i, n in enumerate(nodes) if not (n.startswith("sourcebus") or n.startswith("800."))]
    pos = {i: p for p, i in enumerate(free)}
    y, _, _ = net.assemble()
    z = np.linalg.inv(y[np.ix_(free, free)])
    idx = {n: i for i, n in enumerate(nodes)}
    cols = [[pos[idx[n]] for n in s["nodes"] if idx[n] in pos] for s in net.sections]
    sigma = NOISE_V_REL * np.abs(net.v_pre) * np.sqrt(2)
    noisy = dv + sigma * (rng.standard_normal(dv.shape) + 1j * rng.standard_normal(dv.shape))
    buses = sorted({n.split(".")[0] for i, n in enumerate(nodes) if i in pos})

    def hit_rate(bus_set):
        sel = [i for i, n in enumerate(nodes) if n.split(".")[0] in bus_set and i in pos]
        d = noisy[:, sel]
        mi = np.zeros((len(d), len(cols)))
        for l, c in enumerate(cols):
            a = z[np.ix_([pos[i] for i in sel], c)]
            res = d - (d @ np.linalg.pinv(a, rcond=1e-8).T) @ a.T
            mi[:, l] = (np.abs(res) ** 2).sum(1)
        return (np.argsort(mi, axis=1)[:, :k] == sec_true[:, None]).any(1).mean()

    chosen = []
    for _ in range(n_pmu):
        best = max((b for b in buses if b not in chosen), key=lambda b: hit_rate(set(chosen) | {b}))
        chosen.append(best)
    return [n for n in nodes if n.split(".")[0] in chosen and n in {nodes[i] for i in free}], chosen


def noisy_dv(net, d, pmu_idx, noise_scale, seed):
    """L1 自己的 μPMU 母线上的带噪电压突变量（突变量为两次量测之差，噪声标准差乘 √2）。"""
    rng = np.random.default_rng(seed)
    dv = d["dv"][:, pmu_idx].astype(complex)
    sigma = noise_scale * NOISE_V_REL * np.abs(net.v_pre[pmu_idx]) * np.sqrt(2)
    return dv + sigma * (rng.standard_normal(dv.shape) + 1j * rng.standard_normal(dv.shape))
