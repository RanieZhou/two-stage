"""对比方法：稀疏贝叶斯学习的稀疏量测故障定位（Jiang 等，*Block-Sparse Bayesian Learning Method for Fault Location in Active Distribution Networks
With Limited Synchronized Measurements*，IEEE TPWRS 2021；Jia 等，IEEE TSG 2020 的贝叶斯压缩感知同属这一类）。

无官方代码，按论文思路复现并适配到本实验的设定：
  * 未知量为全网各候选位置的三相“虚拟注入电流”（复数，按 a、b、c 相占位），只有故障位置对应的块非零（块稀疏）；
  * 字典为 OpenDSS 名义参数下各区段在 α 网格（0.05–0.95，共 10 点）上的故障相注入电流响应，块大小为该区段的相数；
  * 块稀疏贝叶斯学习的 EM 迭代：块 b 的先验协方差为 γ_b B_b，γ_b 由数据估计，所有块共用一个相间相关矩阵 B（3×3 复厄米，按块的相别取子阵）；
    B 每次迭代由各块的后验均值外积 Σ_b μ_b μ_b^H/γ_b 加对角正则更新并归一化（对应原文式 (28)–(29)，只用均值项）；
    量测噪声方差取与本方案阶段二相同的加权（量测噪声加模型误差）；
  * 取后验能量最大的块作为故障位置，得到区段与区段内位置（位置受 α 网格限制）。
与原文的差别：原文的未知量是节点注入电流（实部、虚部分开，每节点 6 维）并由相邻两个最大节点按式 (39) 的比值求位置，这里是线路区段内网格位置的注入电流，B 取复数相间相关矩阵。
不使用神经网络，也不假设过渡电阻；线路参数取名义值，因此对线路阻抗偏差没有校正。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_pipeline import EPS_ABS, EPS_REL, NOISE_LEVELS, SCENARIOS, load_set
from ieee34_bus_impedance import CASES, Network

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["ieee34", "ieee123"], default="ieee34")
ap.add_argument("--data", type=Path, default=Path("data/generated"))
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--limit", type=int, default=5000)
ap.add_argument("--chunk", type=int, default=1000)
ap.add_argument("--noise-levels", action="store_true")
ap.add_argument("--scenarios", nargs="+", help="只评测所列场景")
ap.add_argument("--shared-b", action="store_true", help="学习共享的相间相关矩阵 B（默认 B 为单位阵，即各块相互独立）")
ap.add_argument("--eta", type=float, default=0.1, help="--shared-b 时 B 的对角正则（相对 Σ_b μ_b μ_b^H/γ_b 的迹的比例）")
ap.add_argument("--out", type=Path, required=True, help="结果 json；逐样本预测另存为同名 _samples.npz")
a = ap.parse_args()

case = CASES[a.case]
fp = (lambda n: a.data / n) if a.case == "ieee34" else (lambda n: a.data / f"{a.case}_{n}")
net = Network(case=case)
rows = net.sensor_rows(case["sensors"])
sigma = net.measurement_sigma(rows)
is_v = rows < len(net.nodes)
cls = np.array(json.loads((Path("results/evaluation.json") if a.case == "ieee34" else Path(f"results/{a.case}/evaluation.json")).read_text())["equivalence_classes"])

alphas = np.linspace(0.05, 0.95, 10)
lib = net.response_library(alphas)
m = len(rows)
G, b_sec, b_alpha, b_c, Pm = [], [], [], [], []
for k, s in enumerate(net.sections):
    for j, al in enumerate(alphas):
        g = np.zeros((m, 3), dtype=complex)
        ph = [p - 1 for p in s["phases"]]
        g[:, ph] = lib[k][j][rows, :]
        G.append(g); b_sec.append(k); b_alpha.append(al); b_c.append(len(ph))
        Pm.append(np.isin(np.arange(3), ph))
G, b_sec, b_alpha, b_c, Pm = np.array(G), np.array(b_sec), np.array(b_alpha), np.array(b_c), np.array(Pm, dtype=float)
B = len(G)
Gh = G.conj()
Qcd = np.einsum("bic,bjd->bijcd", G, Gh).reshape(B, -1)         # G_b[i,c] conj(G_b[j,d])：Σ_b γ_b G_b B G_b^H 用
R = np.einsum("bic,bjd->ijbcd", Gh, G).reshape(m * m, -1)       # conj(G_b[i,c]) G_b[j,d]：G_b^H M G_b 用
gbar = (np.abs(G) ** 2).sum(2).mean(0)
print("字典块数", B, "量测数", m, flush=True)


def bsbl(z):
    """块稀疏贝叶斯学习（EM）：z (N, m) 复数量测 → 后验能量最大块的序号；--shared-b 时同时学习共享的相间相关矩阵 B。"""
    n = len(z)
    az = np.abs(z)
    zmax = np.zeros_like(az)
    for kind in (is_v, ~is_v):
        zmax[:, kind] = az[:, kind].max(1)[:, None]
    w = 1.0 / (sigma[None, :] ** 2 + EPS_REL ** 2 * az ** 2 + EPS_ABS ** 2 * zmax ** 2)
    d = np.sqrt(w)
    gam = np.repeat(((w * az ** 2).sum(1) / (w * gbar[None]).sum(1))[:, None], B, 1)   # 先验方差的初值：加权信号能量与字典块平均能量之比
    bm = np.tile(np.eye(3, dtype=complex), (n, 1, 1))                                  # 共享的相间相关矩阵，初值为单位阵
    eye = np.eye(m)
    for _ in range(a.iters):
        h0 = np.einsum("nijcd,ncd->nij", (gam @ Qcd).reshape(n, m, m, 3, 3), bm)       # Σ_b γ_b G_b B G_b^H
        sy = eye + d[:, :, None] * h0 * d[:, None, :]
        mm = d[:, :, None] * np.linalg.inv(sy) * d[:, None, :]
        u = np.einsum("nij,nj->ni", mm, z)
        gu = np.einsum("bmc,nm->nbc", Gh, u)                                           # G_b^H u
        v = Pm[None] * np.einsum("ncd,nbd->nbc", bm, gu)                               # B_b G_b^H u（按块的相别取子阵）
        qb = (mm.reshape(n, -1) @ R).reshape(n, B, 3, 3)                               # G_b^H M G_b
        bb = bm[:, None] * Pm[None, :, :, None] * Pm[None, :, None, :]                 # 各块的 B_b
        quad = np.einsum("nbc,nbcd,nbd->nb", gu.conj(), bb, gu).real - np.einsum("nbcd,nbdc->nb", qb, bb).real
        mu2 = gam ** 2 * (np.abs(v) ** 2).sum(-1)
        if a.shared_b:
            bt = np.einsum("nb,nbc,nbd->ncd", gam, v, v.conj())                        # Σ_b μ_b μ_b^H / γ_b
            tr = np.trace(bt, axis1=1, axis2=2).real[:, None, None]
            bt = bt + a.eta * tr / 3 * np.eye(3)
            bm = bt * 3 / np.trace(bt, axis1=1, axis2=2).real[:, None, None]
        gam = np.maximum(gam + gam ** 2 * quad / b_c[None], 1e-12)
    return mu2.argmax(1)


scen = {**SCENARIOS, **(NOISE_LEVELS if a.noise_levels else {})}
results, samples = {}, {}
samples_path = a.out.with_name(a.out.stem + "_samples.npz")
if a.out.exists() and samples_path.exists():   # 断点续跑：读取已完成场景
    results = json.loads(a.out.read_text())
    samples = {k: v for k, v in np.load(samples_path).items()}
for name, (fn, ns) in scen.items():
    if not fp(fn).exists() or name in results or (a.scenarios and name not in a.scenarios):
        continue
    ds = load_set(fp(fn), net, rows, sigma, seed=202, noise_scale=ns, limit=a.limit)
    blk = np.concatenate([bsbl(ds["z"][i:i + a.chunk]) for i in range(0, len(ds["z"]), a.chunk)])
    sec, al = b_sec[blk], b_alpha[blk]
    d_err = np.abs(np.array([net.fault_distance_km(s, x) for s, x in zip(sec, al)]) - ds["d_true"])
    results[name] = dict(section_acc=float((sec == ds["sec"]).mean()), class_acc=float((cls[sec] == cls[ds["sec"]]).mean()), d_mean=float(d_err.mean()),
                         d_median=float(np.median(d_err)))
    samples[f"bsbl/{name}/sec"], samples[f"bsbl/{name}/d_err"] = sec, d_err
    print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    a.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    np.savez_compressed(samples_path, **samples)
