"""验证 L1 复现的保真度：在论文自己的 IEEE 33 节点系统上，对照论文图 4（阶段一 Top-k 成功率）与图 9a（两阶段成功率）。

数据集：s33_r{100,300,500}.npz，由 simulate_dataset.py 生成：
  python simulate_dataset.py --case ieee33 --n 3000 --seed 3<R> --rf 0 <R> --rf-uniform --alpha 0.1 0.9 --mult 1 1 --out data/generated/s33_r<R>.npz
论文数值由图中读数得到，误差约 ±1 个百分点。用法：python validate_l1_ieee33.py
"""
import json
from pathlib import Path

import numpy as np

from baseline_l1 import L1Locator, noisy_phasor
from ieee34_bus_impedance import CASES, Network

# 论文表 I：μPMU 配置（母线号）
PMU_BUSES = {1: [26], 2: [16, 30], 3: [13, 23, 29], 4: [13, 19, 23, 29], 5: [8, 13, 19, 23, 29],
             6: [9, 13, 19, 23, 29, 31], 7: [4, 9, 13, 19, 23, 29, 31], 8: [4, 9, 11, 16, 19, 23, 29, 31]}
# 论文图 4 实线（恒阻抗负荷，故障电阻 0~100 Ω）：{k: [N=1..8 的成功率 %]}
PAPER_FIG4 = {2: [4, 12, 26, 38, 47.5, 56.5, 67, 65], 3: [6, 18.5, 50, 69, 79, 86, 87.5, 88.5],
              4: [9, 24, 79.5, 89, 94, 94, 95.5, 97], 5: [13, 26.5, 86, 95.5, 98.5, 99, 99, 98.5]}
# 论文图 9a 两阶段成功率（恒阻抗负荷，5 个 μPMU）：{(故障电阻上限 Ω, 伪量测误差 %): 成功率 %}
PAPER_FIG9 = {(100, 20): 97.8, (300, 20): 93.8, (500, 20): 89.3, (100, 50): 95.5, (500, 50): 64.0}
# 论文噪声：幅值相对误差已含 1/3 因子；pmu、root_i 为 (幅值相对误差, 相角误差/度)
NOISE = dict(pmu=(0.0016e-2 / 3, 0.0033), root_i=(0.01e-2 / 3, 0.057), vmag=0.1e-2 / 3)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def pmu_nodes(net, buses):
    return [n for n in net.nodes if n.split(".")[0] in {str(b) for b in buses}]


def paper_dv(rng, net, d, idx):
    """μPMU 母线上的电压突变量：故障前后各自带论文的相量误差后相减。"""
    pre = d["v0"][:, idx].astype(complex)
    post = pre + d["dv"][:, idx]
    return noisy_phasor(rng, post, *NOISE["pmu"]) - noisy_phasor(rng, pre, *NOISE["pmu"])


def stage1_curve(net, d):
    """图 4：不同 μPMU 数量下阶段一 Top-k 成功率（%）。"""
    rng = np.random.default_rng(0)
    out = {k: [] for k in (2, 3, 4, 5)}
    for n_pmu, buses in PMU_BUSES.items():
        loc = L1Locator(net, pmu_nodes(net, buses), build_se=False)
        order = np.argsort(loc.stage1(paper_dv(rng, net, d, loc.pmu_idx)), axis=1)
        for k in out:
            out[k].append(100 * float((order[:, :k] == d["sec"][:, None]).any(1).mean()))
    return out


def two_stage(net, loc, d, e_inj, seed=1):
    """图 9a：阶段二找到故障线路任一端点母线的成功率（%）与阶段一 Top-5 成功率（%）。"""
    rng = np.random.default_rng(seed)
    paper = dict(NOISE, e_inj=e_inj / 100)
    mi = loc.stage1(paper_dv(rng, net, d, loc.pmu_idx))
    ok1 = (np.argsort(mi, axis=1)[:, :5] == d["sec"][:, None]).any(1)
    hits = []
    for n in range(len(mi)):
        sample = loc.make_sample(np.random.default_rng(seed * 100003 + n), d, n, paper=paper)
        s = net.sections[d["sec"][n]]
        hits.append(loc.best_bus(sample, mi[n]) in (s["bus1"], s["bus2"]))
    return 100 * float(np.mean(hits)), 100 * float(ok1.mean())


def main():
    net = Network(case=CASES["ieee33"])
    data = Path("data/generated")
    res = {"图4": {}, "图9a": {}}
    ours = stage1_curve(net, load(data / "s33_r100.npz"))
    print("阶段一 Top-k 成功率（%），N=1..8；每行为 本复现 / 论文")
    for k, vals in ours.items():
        print(f"top-{k} 复现", " ".join(f"{v:5.1f}" for v in vals))
        print(f"top-{k} 论文", " ".join(f"{v:5.1f}" for v in PAPER_FIG4[k]))
        res["图4"][k] = dict(复现=vals, 论文=PAPER_FIG4[k])
    err = np.abs(np.array([ours[k] for k in ours]) - np.array([PAPER_FIG4[k] for k in ours]))
    print(f"平均绝对偏差 {err.mean():.1f} 个百分点；N≥3 时 {err[:, 2:].mean():.1f}")
    res["图4"]["平均绝对偏差"] = float(err.mean())
    loc = L1Locator(net, pmu_nodes(net, PMU_BUSES[5]))
    print("两阶段成功率（%），5 个 μPMU：故障电阻上限 / 伪量测误差 -> 复现(阶段一Top-5) / 论文")
    for (r, e), paper in PAPER_FIG9.items():
        succ, top5 = two_stage(net, loc, load(data / f"s33_r{r}.npz"), e)
        print(f"r<={r}Ω e_inj={e}%: {succ:.1f} ({top5:.1f}) / {paper:.1f}")
        res["图9a"][f"r{r}_e{e}"] = dict(复现=succ, 阶段一Top5=top5, 论文=paper)
    Path("results").mkdir(exist_ok=True)
    Path("results/l1_validation_ieee33.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
