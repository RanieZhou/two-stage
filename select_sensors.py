"""测点的贪心前向选择：每步加入使“全区段物理穷举定位的平均测距误差”最小的测点。

测点为 (线路名, 端子)，量测该端母线各相电压与该端线路电流；要求已选测点位于互不相同的母线。
评测样本为各区段等概率的仿真故障（由 simulate_dataset.py 生成），物理穷举用粗 α 网格、名义线路阻抗的单一假设。
IEEE 34 的测点按同一准则选出（当时的选择脚本未保存在仓库中）；IEEE 123 的选择结果见 results/sensor_selection_ieee123.json。
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from ieee34_bus_impedance import CASES, Network, add_measurement_noise
from stage2_continuous_fse import PhysicsModel

ALPHAS = np.linspace(0.1, 0.9, 5)
EPS_REL, EPS_ABS = 0.02, 0.01

_state = {}


def objective(sensors):
    """返回给定测点集上物理穷举的 (平均测距误差 km, 区段准确率)。"""
    net, lib, zc, d_true, sec = (_state[k] for k in ("net", "lib", "zc", "d_true", "sec"))
    rows = net.sensor_rows(sensors)
    sigma = net.measurement_sigma(rows)
    z = add_measurement_noise(np.random.default_rng(0), zc[:, rows], sigma)
    pm = PhysicsModel(net, [lib], [0.0], rows, sigma, EPS_REL, EPS_ABS, ALPHAS)
    pen, _, alpha, _ = pm.fit_all(z)
    hat = pen.argmin(1)
    d_hat = np.array([net.fault_distance_km(s, a) for s, a in zip(hat, alpha[np.arange(len(hat)), hat])])
    return float(np.abs(d_hat - d_true).mean()), float((hat == sec).mean())


def greedy(net, data, k, workers):
    zc = np.concatenate([data["dv"], data["di"]], 1).astype(complex)
    _state.update(net=net, lib=net.response_library(ALPHAS), zc=zc, sec=data["sec"],
                  d_true=np.array([net.fault_distance_km(s, a) for s, a in zip(data["sec"], data["alpha"])]))
    candidates = [(s["name"], t) for s in net.sections for t in (0, 1)]
    bus_of = lambda c: net.sections[net.section_index[c[0]]]["bus1" if c[1] == 0 else "bus2"]
    chosen, history = [], []
    for step in range(k):
        used = {bus_of(c) for c in chosen}
        pool = [c for c in candidates if bus_of(c) not in used]
        with threadpool_limits(1), Pool(workers) as p:
            scores = p.map(objective, [chosen + [c] for c in pool])
        best = min(range(len(pool)), key=lambda i: scores[i][0])
        chosen.append(pool[best])
        history.append(dict(sensors=[list(c) for c in chosen], buses=[bus_of(c) for c in chosen],
                            mean_dist_err_km=scores[best][0], section_acc=scores[best][1]))
        print(step + 1, history[-1], flush=True)
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default="ieee123")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    net = Network(case=CASES[a.case])
    with np.load(a.data) as f:
        data = {k: f[k] for k in f.files}
    a.out.write_text(json.dumps(greedy(net, data, a.k, a.workers), ensure_ascii=False, indent=1))
