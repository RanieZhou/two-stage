"""基于 OpenDSS 的 IEEE 34 故障仿真：生成各区段、位置、故障类型、过渡电阻、负荷水平下的突变量样本。

每个区段线路在建模时预先拆成两段，仿真时只修改两段长度即可把故障点放到区段内任意位置。
样本保存无噪声的全网电压突变量与各线路两端电流突变量，测点选取与加噪在使用时完成。
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import opendssdirect as dss

from ieee34_bus_impedance import CASES, IEEE34, Network, _complex, load_circuit

# (类型名, 故障相)，相别为 1/2/3
FAULT_TYPES = [
    ("SLG", (1,)), ("SLG", (2,)), ("SLG", (3,)),
    ("LL", (1, 2)), ("LL", (2, 3)), ("LL", (3, 1)),
    ("LLG", (1, 2)), ("LLG", (2, 3)), ("LLG", (3, 1)),
    ("LLLG", (1, 2, 3)),
]


def valid_types(phases):
    return [k for k, (_, ph) in enumerate(FAULT_TYPES) if set(ph) <= set(phases)]


def _bus_spec(nodes):
    return nodes[0].split(".")[0] + "".join("." + n.split(".")[1] for n in nodes)


def split_lines(net):
    """把每个区段线路拆成 Lx 与 Lx_b 两段（各半长），中间母线为 mid_<线路名>，返回各线路全长（kft）。"""
    full = {}
    for s in net.sections:
        c = len(s["phases"])
        dss.Lines.Name(s["name"])
        code = dss.Lines.LineCode()
        full[s["name"]] = dss.Lines.Length()
        n2 = s["nodes"][c:]
        mid = f"mid_{s['name']}." + ".".join(map(str, s["phases"]))
        dss.Command(f"Edit Line.{s['name']} Bus2={mid} Length={full[s['name']] / 2}")
        dss.Command(
            f"New Line.{s['name']}_b Phases={c} Bus1={mid} Bus2={_bus_spec(n2)} "
            f"LineCode={code} Length={full[s['name']] / 2} units=kft"
        )
    return full


def _fault_command(name, line, ftype, rf, enabled=True):
    kind, ph = FAULT_TYPES[ftype]
    mid = f"mid_{line}"
    if kind == "LL":
        spec = f"Bus1={mid}.{ph[0]} Bus2={mid}.{ph[1]} Phases=1"
    else:
        spec = f"Bus1={mid}." + ".".join(map(str, ph)) + f" Phases={len(ph)}"
    return f"Edit Fault.{name} {spec} R={rf} enabled={'yes' if enabled else 'no'}"


def scale_line_impedance(factor):
    """把电路中全部线路的串联阻抗矩阵乘以 factor，用于构造与建模参数不一致的仿真。"""
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        dss.Lines.RMatrix([x * factor for x in dss.Lines.RMatrix()])
        dss.Lines.XMatrix([x * factor for x in dss.Lines.XMatrix()])


class Simulator:
    """持有一个已拆线的 OpenDSS 电路，按样本参数生成突变量。

    阻抗模型 Network 始终取算例原始参数，line_z_scale 只改变仿真所用线路的串联阻抗。
    fixed_taps 为真时调压器档位固定在名义负荷下的位置，不随各样本的负荷倍率重新调节。
    """

    def __init__(self, line_z_scale=1.0, fixed_taps=False, case=IEEE34):
        self.fixed_taps = fixed_taps
        self.net = Network(case=case)
        load_circuit(case=case)
        self.base_len = split_lines(self.net)
        if line_z_scale != 1.0:
            scale_line_impedance(line_z_scale)
        dss.Command("New Fault.F1 Bus1=800 Phases=1 R=1 enabled=no")

    def _set_alpha(self, line, alpha):
        full = self.base_len[line]
        dss.Command(f"Edit Line.{line} Length={alpha * full}")
        dss.Command(f"Edit Line.{line}_b Length={(1 - alpha) * full}")

    def _measure(self):
        # 故障元件会改变节点排序，每次测量都按节点名重新定位
        slot = {n.lower(): k for k, n in enumerate(dss.Circuit.AllNodeNames())}
        v = _complex(dss.Circuit.AllBusVolts())[[slot[n] for n in self.net.nodes]]
        i = []
        for s in self.net.sections:
            c = len(s["phases"])
            dss.Circuit.SetActiveElement(f"Line.{s['name']}")
            i.append(_complex(dss.CktElement.Currents())[:c])
            dss.Circuit.SetActiveElement(f"Line.{s['name']}_b")
            i.append(_complex(dss.CktElement.Currents())[c:])
        return v, np.concatenate(i)

    def run(self, sec, alpha, ftype, rf, load_mult):
        """返回 (ΔV, ΔI, 故障电流, 故障前电压, 故障前电流)；未收敛时抛出异常。"""
        line = self.net.sections[sec]["name"]
        dss.Command(f"Set LoadMult={load_mult}")
        dss.Command("Fault.F1.enabled=no")
        if not self.fixed_taps:
            dss.Command("Set ControlMode=STATIC")
        dss.Command("Solve")
        dss.Command("Set ControlMode=OFF")
        v0, i0 = self._measure()
        self._set_alpha(line, alpha)
        dss.Command(_fault_command("F1", line, ftype, rf))
        dss.Command("Solve")
        if not dss.Solution.Converged():
            raise RuntimeError(f"未收敛: sec={line} alpha={alpha} type={ftype} rf={rf} mult={load_mult}")
        v1, i1 = self._measure()
        dss.Circuit.SetActiveElement("Fault.F1")
        c = len(FAULT_TYPES[ftype][1])
        i_f = _complex(dss.CktElement.Currents())[:c]
        return v1 - v0, i1 - i0, i_f, v0, i0


def sample_records(rng, sections, n, rf_range, mult_range, by_length=True, no_fault=(), rf_uniform=False, alpha_range=(0.02, 0.98)):
    """by_length 为真时故障位置沿全网均匀（区段按长度加权），否则各区段等概率；区段内 α 在 alpha_range 内均匀；过渡电阻默认对数均匀，rf_uniform 为真时线性均匀。"""
    length = np.array([0.0 if s["name"] in no_fault else (s["length_km"] if by_length else 1.0) for s in sections])
    secs = rng.choice(len(sections), n, p=length / length.sum())
    recs = []
    for s in secs:
        types = valid_types(sections[s]["phases"])
        recs.append((
            int(s), rng.uniform(*alpha_range), int(rng.choice(types)),
            float(rng.uniform(*rf_range)) if rf_uniform else float(np.exp(rng.uniform(np.log(rf_range[0]), np.log(rf_range[1])))),
            rng.uniform(*mult_range),
        ))
    return recs


def read_sensor_data(path, rows):
    """读取数据集在测点量测行 rows 上的突变量 zc 与故障前相量 pre（缺 v0、i0 时为 None），并附标签。
    原始数据集（dv、di、v0、i0 为全网）与 --sensors 生成的精简数据集（只存 zc、pre）都可读。"""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    if "zc" not in d:
        d["zc"] = np.concatenate([d["dv"], d["di"]], 1)[:, rows].astype(complex)
        d["pre"] = np.concatenate([d["v0"], d["i0"]], 1)[:, rows].astype(complex) if "v0" in d else None
    return {k: d[k] for k in ("sec", "alpha", "ftype", "rf", "load_mult", "zc", "pre") if k in d}


_sim = None
_line_z_scale = 1.0
_fixed_taps = False
_case = IEEE34


def _work(recs):
    global _sim
    if _sim is None:
        _sim = Simulator(_line_z_scale, _fixed_taps, _case)
    out = []
    for r in recs:
        out.append(_sim.run(*r))
    return out


def generate(n, seed, rf_range, mult_range, workers, by_length=True, line_z_scale=1.0, fixed_taps=False, case=IEEE34,
             rf_uniform=False, alpha_range=(0.02, 0.98)):
    global _line_z_scale, _fixed_taps, _case
    _case = case
    _line_z_scale = line_z_scale
    _fixed_taps = fixed_taps
    net_sections = Network(case=case).sections
    rng = np.random.default_rng(seed)
    recs = sample_records(rng, net_sections, n, rf_range, mult_range, by_length, case["no_fault"], rf_uniform, alpha_range)
    chunks = [recs[i:i + 64] for i in range(0, n, 64)]
    with Pool(workers) as pool:
        flat = [x for chunk in pool.map(_work, chunks) for x in chunk]
    return dict(
        sec=np.array([r[0] for r in recs]), alpha=np.array([r[1] for r in recs]),
        ftype=np.array([r[2] for r in recs]), rf=np.array([r[3] for r in recs]),
        load_mult=np.array([r[4] for r in recs]),
        dv=np.array([f[0] for f in flat], dtype=np.complex64),
        di=np.array([f[1] for f in flat], dtype=np.complex64),
        v0=np.array([f[3] for f in flat], dtype=np.complex64),
        i0=np.array([f[4] for f in flat], dtype=np.complex64),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rf", type=float, nargs=2, default=(0.01, 1000.0))
    ap.add_argument("--mult", type=float, nargs=2, default=(0.6, 1.2))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rf-uniform", action="store_true", help="过渡电阻线性均匀（默认对数均匀）")
    ap.add_argument("--alpha", type=float, nargs=2, default=(0.02, 0.98), help="区段内故障位置比例范围")
    ap.add_argument("--case", choices=sorted(CASES), default="ieee34")
    ap.add_argument("--fixed-taps", action="store_true", help="调压器档位固定在名义负荷下的位置")
    ap.add_argument("--line-z-scale", type=float, default=1.0, help="仿真线路串联阻抗相对建模参数的倍数")
    ap.add_argument("--uniform-sections", action="store_true", help="各区段等概率而非按长度加权")
    ap.add_argument("--sensors", action="store_true", help="只保存该算例测点行上的突变量 zc 与故障前相量 pre（大系统用，省空间）")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()  # OpenDSS 编译会切换工作目录
    data = generate(a.n, a.seed, a.rf, a.mult, a.workers, not a.uniform_sections, a.line_z_scale, a.fixed_taps, CASES[a.case], a.rf_uniform, tuple(a.alpha))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.sensors:
        rows = Network(case=CASES[a.case]).sensor_rows(CASES[a.case]["sensors"])
        zc = np.concatenate([data["dv"], data["di"]], 1)[:, rows]
        pre = np.concatenate([data["v0"], data["i0"]], 1)[:, rows]
        data = {k: v for k, v in data.items() if k not in ("dv", "di", "v0", "i0")} | dict(zc=zc, pre=pre)
    np.savez_compressed(a.out, **data)
    print(f"saved {a.n} samples -> {a.out}")
