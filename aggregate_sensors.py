"""汇总 IEEE 34 测点数量扫描（results/improved/ieee34_sensors/n<K>/）：每个测点数、每个场景给出
本方法、阶段一（MLP）、物理穷举与 GCN（突变量输入）的区段准确率（精确/等价类）、测距误差，均值与标准差（ddof=1）。

评测命令见 实验结果与分析.md；种子只改变神经网络的初始化与小批量抽样，测试样本相同。
GCN 只输出区段：“区段中点”取区段中点为故障点，“+物理定位”把它的 Top-1 区段交给阶段二估计区段内位置，“+本方法的决策”把它的概率代入阶段三。
"""
import json
from pathlib import Path

import numpy as np

from ieee34_bus_impedance import SENSOR_ORDER_34

ROOT = Path("results/improved/ieee34_sensors")
KS = (3, 4, 5, 6, 8)
SCENARIOS = ("域内", "高阻外推")
OURS = "本方案(联合打分+候选扩展)"


def stat(v):
    v = np.array(v, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=len(v))


def exact(samples, sc, key):
    return float((samples[f"{sc}/{key}/sec"] == samples[f"{sc}/sec_true"]).mean())


out = {}
for K in KS:
    d = ROOT / f"n{K}"
    seeds = sorted(int(p.name[1:]) for p in d.glob("s[0-9]*") if p.is_dir() and (p / "evaluation.json").exists())
    cls = np.array(json.load(open(d / "s0/evaluation.json"))["equivalence_classes"])
    rec = {"测点": [list(x) for x in SENSOR_ORDER_34[:K]], "等价类数": int(cls.max() + 1)}
    for sc in SCENARIOS:
        acc = {}
        for s in seeds:
            ev = json.load(open(d / f"s{s}/evaluation.json"))[sc]
            sm = np.load(d / f"s{s}_samples.npz")
            ok = lambda key: float((cls[sm[f"{sc}/{key}/sec"]] == cls[sm[f"{sc}/sec_true"]]).mean())
            row = {
                "本方法 类准确率": ev[OURS]["class_acc"], "本方法 精确区段": exact(sm, sc, "本方案"), "本方法 测距误差": ev[OURS]["d_mean"],
                "阶段一 类准确率": ev["基准1 硬串联"]["class_acc"], "阶段一 精确区段": exact(sm, sc, "硬串联"),
                "物理穷举 类准确率": ev["基准3 物理穷举"]["class_acc"], "物理穷举 精确区段": exact(sm, sc, "物理穷举"), "物理穷举 测距误差": ev["基准3 物理穷举"]["d_mean"],
                "真实区段已知 测距误差": ev["基准4 Oracle"]["d_mean"],
            }
            g = d / f"gcn_s{s}.json"
            if g.exists():
                gr = json.load(open(g))["gcn_delta"][sc]
                gs = np.load(d / f"gcn_s{s}_swap_samples.npz")
                sw = json.load(open(d / f"gcn_s{s}/evaluation.json"))[sc]
                row.update({
                    "GCN 类准确率": gr["class_acc"], "GCN 精确区段": gr["section_acc"], "GCN 测距误差(区段中点)": gr["d_mean_midpoint"],
                    "GCN+物理定位 测距误差": sw["基准1 硬串联"]["d_mean"],
                    "GCN+本方法的决策 类准确率": sw[OURS]["class_acc"], "GCN+本方法的决策 精确区段": exact(gs, sc, "本方案"), "GCN+本方法的决策 测距误差": sw[OURS]["d_mean"],
                })
            for k, v in row.items():
                acc.setdefault(k, []).append(v)
        rec[sc] = {k: stat(v) for k, v in acc.items()}
    out[f"n{K}"] = rec
(ROOT / "sensor_sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for sc in SCENARIOS:
    print(sc)
    for K in KS:
        r = out[f"n{K}"]
        f = lambda k: f"{r[sc][k]['mean'] * 100:5.1f}" if "测距" not in k else f"{r[sc][k]['mean']:5.2f}"
        print(f"K={K} 类数{r['等价类数']:2d} | 本方法 类{f('本方法 类准确率')} 精确{f('本方法 精确区段')} km{f('本方法 测距误差')} | 阶段一 类{f('阶段一 类准确率')} 精确{f('阶段一 精确区段')} | 物理 类{f('物理穷举 类准确率')} 精确{f('物理穷举 精确区段')} km{f('物理穷举 测距误差')} | 已知区段 km{f('真实区段已知 测距误差')}"
              + (f" | GCN 类{f('GCN 类准确率')} 精确{f('GCN 精确区段')} km中点{f('GCN 测距误差(区段中点)')} +物理{f('GCN+物理定位 测距误差')}" if "GCN 类准确率" in r[sc] else ""))
