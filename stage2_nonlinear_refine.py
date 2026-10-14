"""阶段二的局部非线性精化：用 OpenDSS 潮流在候选区段上重新拟合故障位置 α 与过渡电阻 Rf。

线性叠加模型把负荷当作恒阻抗，非线性负荷造成电流量测的模型误差（见实验结果与分析第八节）。
精化用完整潮流（保留恒功率负荷、调压器档位冻结在故障前位置）计算测点突变量，
对 (α, log10 Rf) 做局部搜索，最小化与阶段二相同的加权残差；故障类型与区段沿用线性阶段的结果。

潮流需要全网负荷水平：用带噪的故障前测点相量（形成突变量本来就需要）在不同负荷倍率的潮流表上做最小二乘估计。
线路阻抗用若干倍率假设，各假设分别精化，由残差加先验惩罚选择。
"""
from multiprocessing import Pool

import numpy as np
import opendssdirect as dss
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from ieee34_bus_impedance import SENSORS, _complex
from simulate_dataset import Simulator, _fault_command, scale_line_impedance

LOAD_GRID = np.arange(0.5, 1.6001, 0.025)
RF_SCAN = (0.03, 0.3, 3.0, 30.0, 100.0, 300.0, 1000.0)
RF_RANGE = (0.01, 1000.0)
ALPHA_RANGE = (0.02, 0.98)
RF_GATE = 100.0   # 估计的 Rf 超过该值（Ω）时故障特征弱，精化的位置不可靠；由验证集上按估计 Rf 分段的测距误差确定


class NonlinearRefiner:
    """持有一个潮流电路。z 的列顺序与 Network.sensor_rows(SENSORS) 一致（每个测点：各相电压，再各相电流）。"""

    def __init__(self, sigma_noise, eps_rel, eps_abs, fixed_taps=False):
        self.sigma2 = sigma_noise ** 2
        self.eps_rel2, self.eps_abs2 = eps_rel ** 2, eps_abs ** 2
        self.fixed_taps = fixed_taps
        self.sim = Simulator(1.0, fixed_taps)
        net = self.sim.net
        self.names = [s["name"] for s in net.sections]
        self.volts, self.currents, is_v = [], [], []   # (母线, [相别]) 与 (元件, 切片)
        for name, term in SENSORS:
            s = net.sections[net.section_index[name]]
            c = len(s["phases"])
            nodes = s["nodes"][:c] if term == 0 else s["nodes"][c:]
            self.volts.append((nodes[0].split(".")[0], [int(n.split(".")[1]) for n in nodes]))
            self.currents.append((f"Line.{name}" if term == 0 else f"Line.{name}_b", slice(0, c) if term == 0 else slice(c, 2 * c)))
            is_v += [True] * c + [False] * c
        self.is_v = np.array(is_v)
        self.z_scale, self.tables = 1.0, {}
        self.set_z(1.0)

    def _measure(self):
        out = []
        for (bus, phases), (elem, sl) in zip(self.volts, self.currents):
            dss.Circuit.SetActiveBus(bus)
            at = dict(zip(dss.Bus.Nodes(), _complex(dss.Bus.Voltages())))
            out += [at[p] for p in phases]
            dss.Circuit.SetActiveElement(elem)
            out += list(_complex(dss.CktElement.Currents())[sl])
        return np.array(out)

    def set_load(self, mult):
        """把全网负荷设为名义值的 mult 倍，求故障前状态（调压器档位按该负荷调节，随后冻结）。"""
        dss.Command(f"Set LoadMult={mult}")
        dss.Command("Fault.F1.enabled=no")
        if not self.fixed_taps:
            dss.Command("Set ControlMode=STATIC")
        dss.Command("Solve")
        dss.Command("Set ControlMode=OFF")
        self.pre = self._measure()

    def set_z(self, scale):
        """把全部线路的串联阻抗调为建模参数的 scale 倍，并建立该阻抗下的故障前测点相量随负荷倍率的表。"""
        if scale != self.z_scale:
            scale_line_impedance(scale / self.z_scale)
            self.z_scale = scale
        if scale not in self.tables:
            table = []
            for m in LOAD_GRID:
                self.set_load(m)
                table.append(self.pre)
            self.tables[scale] = np.array(table)

    def estimate_load(self, pre_meas, sigma_pre):
        """由带噪的故障前测点相量估计负荷倍率：加权最小二乘并抛物线细化。"""
        c = (np.abs(pre_meas[None, :] - self.tables[self.z_scale]) ** 2 / sigma_pre ** 2).sum(1)
        k = int(np.clip(c.argmin(), 1, len(c) - 2))
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        off = 0.5 * (c[k - 1] - c[k + 1]) / den if den > 1e-12 else 0.0
        return float(np.clip(LOAD_GRID[k] + np.clip(off, -1, 1) * (LOAD_GRID[1] - LOAD_GRID[0]), LOAD_GRID[0], LOAD_GRID[-1]))

    def simulate(self, line, alpha, ftype, rf):
        """返回测点突变量（相对 set_load 设定的故障前状态）。"""
        self.sim._set_alpha(line, alpha)
        dss.Command(_fault_command("F1", line, ftype, rf))
        dss.Command("Solve")
        return self._measure() - self.pre

    def weights(self, z):
        az = np.abs(z)
        zmax = np.zeros_like(az)
        for kind in (self.is_v, ~self.is_v):
            zmax[kind] = az[kind].max()
        return 1.0 / (self.sigma2 + self.eps_rel2 * az ** 2 + self.eps_abs2 * zmax ** 2)

    def refine(self, z, sec, alpha0, ftype, max_evals=40):
        """从线性阶段的 (alpha0, ftype) 出发，返回 (残差 J, alpha, rf)。需先 set_load。"""
        line = self.names[sec]
        w = self.weights(z)

        def cost(x):
            alpha = float(np.clip(x[0], *ALPHA_RANGE))
            rf = float(np.clip(10 ** x[1], *RF_RANGE))
            return float((w * np.abs(z - self.simulate(line, alpha, ftype, rf)) ** 2).sum())

        x0 = np.log10(min((cost([alpha0, np.log10(rf)]), rf) for rf in RF_SCAN)[1])
        res = minimize(cost, [alpha0, x0], method="Nelder-Mead", options=dict(
            maxfev=max_evals, xatol=2e-3, fatol=1e-2, initial_simplex=[[alpha0, x0], [alpha0 + 0.05, x0], [alpha0, x0 + 0.4]]))
        dss.Command("Fault.F1.enabled=no")
        return float(res.fun), float(np.clip(res.x[0], *ALPHA_RANGE)), float(np.clip(10 ** res.x[1], *RF_RANGE))


_refiner = None


def _work(task):
    global _refiner
    (sigma, eps_rel, eps_abs, fixed_taps, hyps, sigma_pre), z, pre, sec, alpha0, ftype = task
    if _refiner is None or _refiner.fixed_taps != fixed_taps:
        _refiner = NonlinearRefiner(sigma, eps_rel, eps_abs, fixed_taps)
    out = np.zeros(sec.shape + (len(hyps), 3))
    for h, scale in enumerate(hyps):
        _refiner.set_z(scale)
        for i in range(len(z)):
            _refiner.set_load(_refiner.estimate_load(pre[i], sigma_pre))
            for k in range(sec.shape[1]):
                out[i, k, h] = _refiner.refine(z[i], sec[i, k], alpha0[i, k], int(ftype[i, k]))
    return out


def refine_batch(z, pre, sec, alpha0, ftype, sigma, eps_rel, eps_abs, sigma_pre, fixed_taps=False, hyps=(1.0,), workers=4, chunk=25):
    """z: 带噪突变量 (N, m)；pre: 带噪故障前测点相量 (N, m)；sec/alpha0/ftype: 各样本的候选 (N, K)。
    返回 (N, K, 假设数, 3)，最后一维为 (残差 J, alpha, rf)。"""
    cfg = (sigma, eps_rel, eps_abs, fixed_taps, tuple(hyps), sigma_pre)
    tasks = [(cfg, z[i:i + chunk], pre[i:i + chunk], sec[i:i + chunk], alpha0[i:i + chunk], ftype[i:i + chunk]) for i in range(0, len(z), chunk)]
    with threadpool_limits(1), Pool(workers) as pool:
        return np.concatenate(pool.map(_work, tasks))
