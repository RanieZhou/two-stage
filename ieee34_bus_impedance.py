"""IEEE 34 节点相域三相网络模型：线路区段、节点导纳矩阵、故障点到测点的转移阻抗。

故障后网络按线性叠加处理：故障点注入电流 i 在各节点产生的电压突变量为 ΔV = Z i，
Z 由节点导纳矩阵求逆得到。负荷取故障前工作点的恒阻抗等值，调压器档位固定。
"""
import os
from pathlib import Path

import numpy as np
import opendssdirect as dss

DSS_FILE = Path(__file__).parent / "data/ieee34/ieee34Mod1.dss"
# 调压器旁 3 m 长的连接线，不作为故障区段
NON_SECTION_LINES = ("l7", "l25")
FT_PER_KM = 3280.84
# 量测噪声：电压相量按名义电压的比例，电流相量为绝对值（A）；突变量为两次量测之差，标准差再乘 √2
NOISE_V_REL = 1e-3
NOISE_I_ABS = 0.2
# 5 个测点：(线路名, 端子)，端子 0 为 bus1 端、1 为 bus2 端；每个测点量测该端母线各相电压与该端线路电流。
# 选点依据：以物理穷举定位的平均测距误差为目标做贪心前向选择，5 个测点位于互不相同的母线。
SENSORS = [("l6", 1), ("l30", 1), ("l2", 0), ("l11", 1), ("l29", 1)]


def load_circuit(load_mult=1.0, fix_taps=True):
    """编译算例、设置负荷倍率并求解，随后冻结调压器档位。"""
    cwd = os.getcwd()
    dss.Command("Clear")
    dss.Command(f"Compile {DSS_FILE}")
    os.chdir(cwd)  # OpenDSS 编译会切换工作目录
    dss.Command(f"Set LoadMult={load_mult}")
    dss.Command("Solve")
    if fix_taps:
        dss.Command("Set ControlMode=OFF")


def _terminal_nodes(bus_names, node_order):
    """把 (各端子母线名, NodeOrder) 展开为全局节点名列表，接地节点为 None。"""
    n_term = len(bus_names)
    n_cond = len(node_order) // n_term
    nodes = []
    for t, bus in enumerate(bus_names):
        bus = bus.split(".")[0].lower()
        for k in range(n_cond):
            n = node_order[t * n_cond + k]
            nodes.append(None if n == 0 else f"{bus}.{n}")
    return nodes


def _yprim():
    y = np.array(dss.CktElement.YPrim(), dtype=float)
    n = int(round(np.sqrt(len(y) / 2)))
    return (y[0::2] + 1j * y[1::2]).reshape(n, n)


def read_sections():
    """读取全部故障区段（线路）的拓扑、相别、长度与全线 Yprim。"""
    sections = []
    for name in dss.Lines.AllNames():
        if name in NON_SECTION_LINES:
            continue
        dss.Lines.Name(name)
        nodes = _terminal_nodes(dss.CktElement.BusNames(), dss.CktElement.NodeOrder())
        c = len(nodes) // 2
        sections.append(dict(
            name=name,
            bus1=nodes[0].split(".")[0], bus2=nodes[c].split(".")[0],
            phases=[int(n.split(".")[1]) for n in nodes[:c]],
            length_km=_length_km(dss.Lines.Length(), dss.Lines.Units()),
            nodes=nodes, y=_yprim(),
        ))
    return sections


def _length_km(length, units):
    assert units == 2, f"仅支持 kft 单位，得到单位代码 {units}"
    return length * 1000.0 / FT_PER_KM


def read_fixed_elements():
    """读取固定元件（电源内阻抗、变压器、电容器、调压器旁连接线）的 Yprim，故障区段线路由 read_sections 负责。"""
    elems = []
    for cls in ("Vsource", "Transformer", "Capacitor", "Line"):
        dss.Circuit.SetActiveClass(cls)
        for name in dss.ActiveClass.AllNames():
            if cls == "Line" and name not in NON_SECTION_LINES:
                continue
            dss.Circuit.SetActiveElement(f"{cls}.{name}")
            nodes = _terminal_nodes(dss.CktElement.BusNames(), dss.CktElement.NodeOrder())
            elems.append((nodes, _yprim()))
    return elems


def read_load_branches():
    """按故障前工作点读取每个负荷的支路：节点对 (a, b)，b 为 None 表示接地；支路复功率 s（VA）、
    支路电压幅值平方的均值 v2、OpenDSS 负荷模型号。星形负荷为各相对地支路，三角形负荷为线间支路。"""
    out = []
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        terminals = _terminal_nodes(dss.CktElement.BusNames(), dss.CktElement.NodeOrder())
        nodes = [n for n in terminals if n]
        v = {n: x for n, x in zip(terminals, _complex(dss.CktElement.Voltages())) if n}
        s_total = complex(sum(dss.CktElement.Powers()[0::2]), sum(dss.CktElement.Powers()[1::2])) * 1e3
        if dss.Loads.IsDelta():
            # 单相三角形负荷接在两个端子之间（第二个端子可能是地），三相三角形负荷依次接相邻两相
            branches = [(terminals[0], terminals[1])] if dss.CktElement.NumPhases() == 1 else list(zip(nodes, nodes[1:] + nodes[:1]))
        else:
            branches = [(n, None) for n in nodes]
        # 各支路取同一电压平方（平均值），与 OpenDSS 恒阻抗负荷三相对称的处理一致
        v2 = np.mean([abs(v[a] - (v[b] if b else 0)) ** 2 for a, b in branches])
        for a, b in branches:
            out.append(dict(a=a, b=b, s=s_total / len(branches), v2=v2, model=dss.Loads.Model()))
    return out


def read_load_admittances():
    """按故障前工作点把每个负荷等值为恒阻抗，返回 [(节点列表, Y 矩阵)]，Y 单位西门子。"""
    out = []
    for br in read_load_branches():
        y = np.conj(br["s"]) / br["v2"]
        if br["b"]:
            out.append(([br["a"], br["b"]], y * np.array([[1, -1], [-1, 1]])))
        else:
            out.append(([br["a"]], np.array([[y]])))
    return out


def _complex(flat):
    a = np.array(flat, dtype=float)
    return a[0::2] + 1j * a[1::2]


def scale_line_yprim(y_full, factor):
    """把全线 Yprim 对应的串联阻抗乘以 factor（并联电容不变）。"""
    c = y_full.shape[0] // 2
    ys = -y_full[:c, c:]
    shunt = y_full[:c, :c] - ys
    ys = ys / factor
    return np.block([[ys + shunt, -ys], [-ys, ys + shunt]])


def line_segment_yprim(y_full, frac):
    """取全线 Yprim 中长度占比为 frac 的一段（集总 π 模型，串联导纳 ∝ 1/长度，并联导纳 ∝ 长度）。"""
    c = y_full.shape[0] // 2
    ys = -y_full[:c, c:]
    yc = 2 * (y_full[:c, :c] - ys)
    ya = ys / frac
    shunt = yc * frac / 2
    return np.block([[ya + shunt, -ya], [-ya, ya + shunt]])


class Network:
    """由故障前工作点构成的线性网络：区段、节点集合与除故障线外的固定导纳。"""

    def __init__(self, load_mult=1.0, line_z_scale=1.0):
        load_circuit(load_mult)
        self.sections = read_sections()
        for s in self.sections:
            s["y"] = scale_line_yprim(s["y"], line_z_scale)
        self.section_index = {s["name"]: i for i, s in enumerate(self.sections)}
        self.nodes = [n.lower() for n in dss.Circuit.AllNodeNames()]
        self._elements = read_fixed_elements()
        self._loads = read_load_admittances()
        self.distance_km = self._bus_distances()
        slot = {n.lower(): k for k, n in enumerate(dss.Circuit.AllNodeNames())}
        v = _complex(dss.Circuit.AllBusVolts())
        self.v_pre = np.array([v[slot[n]] for n in self.nodes])
        self.v_ref = np.abs(self.v_pre)
        self.load_branches = read_load_branches()

    def _bus_distances(self):
        """各母线到变电站的沿线距离（km），变压器与调压器视为零长度。"""
        adj = {}
        for s in self.sections:
            adj.setdefault(s["bus1"], []).append((s["bus2"], s["length_km"]))
            adj.setdefault(s["bus2"], []).append((s["bus1"], s["length_km"]))
        # 变压器、调压器与调压器旁连接线视为零长度
        links = [f"Transformer.{n}" for n in dss.Transformers.AllNames()]
        links += [f"Line.{n}" for n in NON_SECTION_LINES]
        for el in links:
            dss.Circuit.SetActiveElement(el)
            b = [x.split(".")[0].lower() for x in dss.CktElement.BusNames()]
            adj.setdefault(b[0], []).append((b[1], 0.0))
            adj.setdefault(b[1], []).append((b[0], 0.0))
        self.adj = adj
        dist = {"800": 0.0}
        stack = ["800"]
        while stack:
            u = stack.pop()
            for v, d in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + d
                    stack.append(v)
        return dist

    def fault_distance_km(self, sec, alpha):
        """区段 sec 上位置比例 alpha（自 bus1 起算）处的故障点到变电站的距离。"""
        s = self.sections[sec]
        d1, d2 = self.distance_km[s["bus1"]], self.distance_km[s["bus2"]]
        return d1 + alpha * (d2 - d1)

    def assemble(self, split=None, loads=True):
        """装配节点导纳矩阵（电源内阻抗作为接地并联导纳）。

        split=(区段序号, alpha) 时该线路在 alpha 处拆成两段并新增故障节点 fx.<相>；loads=False 时不含负荷导纳。
        返回 (Y, 节点名列表, 故障节点索引列表)。
        """
        nodes = list(self.nodes)
        fx = []
        if split is not None:
            sec, alpha = split
            fx = [f"fx.{p}" for p in self.sections[sec]["phases"]]
            nodes = nodes + fx
        idx = {n: i for i, n in enumerate(nodes)}
        y = np.zeros((len(nodes), len(nodes)), dtype=complex)

        def add(names, ym):
            ii = [idx.get(n) for n in names]
            for a, ia in enumerate(ii):
                if ia is None:
                    continue
                for b, ib in enumerate(ii):
                    if ib is not None:
                        y[ia, ib] += ym[a, b]

        for names, ym in self._elements + (self._loads if loads else []):
            add(names, ym)
        for i, s in enumerate(self.sections):
            if split is not None and i == split[0]:
                c = len(s["phases"])
                n1, n2 = s["nodes"][:c], s["nodes"][c:]
                add(n1 + fx, line_segment_yprim(s["y"], alpha))
                add(fx + n2, line_segment_yprim(s["y"], 1 - alpha))
            else:
                add(s["nodes"], s["y"])
        return y, nodes, [idx[n] for n in fx]

    # ---- 测量向量布局：[全部节点电压突变量] + [各区段线路首端、末端电流突变量] ----
    def layout(self):
        """返回 (节点索引字典, 各区段电流起始偏移 [首端, 末端])，电流偏移已含节点电压段。"""
        node_index = {n: i for i, n in enumerate(self.nodes)}
        off = len(self.nodes)
        cur_off = []
        for s in self.sections:
            c = len(s["phases"])
            cur_off.append((off, off + c))
            off += 2 * c
        return node_index, cur_off, off

    def response(self, sec, alpha):
        """区段 sec 的 alpha 处各故障相单位注入电流引起的全部测量量突变，形状 (测量数, 故障相数)。"""
        node_index, cur_off, n_meas = self.layout()
        y, nodes, fx = self.assemble(split=(sec, alpha))
        idx = {n: i for i, n in enumerate(nodes)}
        c = len(self.sections[sec]["phases"])
        rhs = np.zeros((len(nodes), c), dtype=complex)
        rhs[fx, np.arange(c)] = 1.0
        v = np.linalg.solve(y, rhs)
        out = np.zeros((n_meas, c), dtype=complex)
        out[: len(self.nodes)] = v[[idx[n] for n in self.nodes]]
        for k, s in enumerate(self.sections):
            ck = len(s["phases"])
            n1, n2 = s["nodes"][:ck], s["nodes"][ck:]
            if k == sec:
                ya, yb = line_segment_yprim(s["y"], alpha), line_segment_yprim(s["y"], 1 - alpha)
                out[cur_off[k][0]: cur_off[k][0] + ck] = ya[:ck] @ v[[idx[n] for n in n1] + fx]
                out[cur_off[k][1]: cur_off[k][1] + ck] = yb[ck:] @ v[fx + [idx[n] for n in n2]]
            else:
                vk = v[[idx[n] for n in n1 + n2]]
                out[cur_off[k][0]: cur_off[k][0] + ck] = s["y"][:ck] @ vk
                out[cur_off[k][1]: cur_off[k][1] + ck] = s["y"][ck:] @ vk
        return out

    def response_library(self, alphas):
        """各区段在 alphas 网格上的响应，返回列表，元素形状 (len(alphas), 测量数, 故障相数)。"""
        return [np.array([self.response(k, a) for a in alphas]) for k in range(len(self.sections))]

    def sensor_rows(self, sensors):
        """sensors 为 (线路名, 端子 0/1) 列表；返回所选测点的测量行号（该端母线各相电压 + 该端线路电流）。"""
        node_index, cur_off, _ = self.layout()
        rows = []
        for name, term in sensors:
            sec = self.section_index[name]
            s = self.sections[sec]
            c = len(s["phases"])
            names = s["nodes"][:c] if term == 0 else s["nodes"][c:]
            rows += [node_index[n] for n in names]
            rows += list(range(cur_off[sec][term], cur_off[sec][term] + c))
        return np.array(rows)

    def measurement_sigma(self, rows):
        """所选测量行的突变量噪声标准差（电压按名义电压比例，电流为绝对值）。"""
        n_nodes = len(self.nodes)
        sig = np.full(len(rows), NOISE_I_ABS * np.sqrt(2))
        is_v = rows < n_nodes
        sig[is_v] = NOISE_V_REL * self.v_ref[rows[is_v]] * np.sqrt(2)
        return sig


def add_measurement_noise(rng, z, sigma, scale=1.0):
    """给复数突变量 z（N, m）叠加圆对称复高斯噪声，sigma 为实部/虚部各自的标准差。"""
    return z + scale * sigma * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))
