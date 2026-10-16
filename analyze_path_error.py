"""IEEE 34 主结果的测距误差改按沿馈线最短路径距离统计（原先是预测点与真实点到变电站距离之差，预测到另一支线时被低估）。

不重新训练：各方法选出的区段取自逐样本文件，区段内位置取自物理反演缓存（data/generated/phys_*.npz，各区段的 α 估计）；
块稀疏贝叶斯的位置在 10 点网格上，由已存的根距离误差反推；Liang 的区段与位置取自其缓存。
先用同一组区段与位置重算原来的根距离误差，与已存值对照，确认复原无误，再输出路径误差。
输出 results/improved/ieee34/path_error.json（均值、中位数、P90、区段判对时的均值）与 path_error_samples.npz（逐样本，供图 4）。
"""
import json
from pathlib import Path

import numpy as np

from evaluate_pipeline import EPS_ABS, EPS_REL, NOISE_LEVELS, SCENARIOS, Z_SCALES, phys_tag
from ieee34_bus_impedance import CASES, Network

R = Path("results/improved/ieee34")
DATA = Path("data/generated")
TAGS = {"gcn_p": "gcn_paper", "gcn_d": "gcn_delta", "gat_p": "gat_paper", "gat_d": "gat_delta", "svm": "pca_svm", "rf": "pca_rf"}
net = Network(case=CASES["ieee34"])
tag = phys_tag(CASES["ieee34"]["sensors"], Z_SCALES, True, 0.05)
cls = np.array(json.load(open(R / "evaluation.json"))["equivalence_classes"])
ours = np.load(R / "samples.npz")
swaps = {k: np.load(R / "stage1_swap" / v / "samples.npz") for k, v in TAGS.items()}
bsbl = np.load(R / "baselines_bsbl_samples.npz")
grid = np.linspace(0.05, 0.95, 10)
scen = {**SCENARIOS, **NOISE_LEVELS}
root = lambda sec, al: np.array([net.fault_distance_km(s, a) for s, a in zip(sec, al)])


def stats(err, ok=None):
    out = dict(mean=float(err.mean()), median=float(np.median(err)), p90=float(np.percentile(err, 90)))
    if ok is not None and ok.any():
        out["区段判对时均值"] = float(err[ok].mean())
    return out


def check(sec, al, d_true, stored, what):
    diff = np.abs(np.abs(root(sec, al) - d_true) - stored).max()
    assert diff < 1e-6, f"{what}: 复原的根距离误差与已存值相差 {diff}"


out, curves = {}, {}
for sc in [s for s in scen if f"{s}/sec_true" in ours.files]:
    fn, noise = scen[sc]
    lt = "_L5000"
    cache = np.load(DATA / f"phys_{fn[:-4]}_n{noise}_{tag}{lt}.npz")
    al_all = cache["alpha"]
    n = len(ours[f"{sc}/sec_true"])
    idx = np.arange(n)
    st, at, dt = ours[f"{sc}/sec_true"], ours[f"{sc}/alpha"], ours[f"{sc}/d_true"]
    ok_ = lambda sec: cls[sec] == cls[st]
    ex_ = lambda sec: sec == st
    res = {}
    paths = {}

    def add(name, sec, al, ok=None):
        e = net.path_distance_km(sec, al, st, at)
        res[name] = stats(e, ex_(sec))
        paths[name] = e

    for name, key in (("ours", "本方案"), ("nn_phys", "硬串联"), ("phys_all", "物理穷举")):
        sec = ours[f"{sc}/{key}/sec"]
        al = al_all[idx, sec]
        check(sec, al, dt, ours[f"{sc}/{key}/d_err"], f"{sc}/{name}")
        add(name, sec, al)
    add("oracle", st, al_all[idx, st])
    for k, s in swaps.items():
        if f"{sc}/硬串联/sec" not in s.files:
            continue
        sec = s[f"{sc}/硬串联/sec"]
        al = al_all[idx, sec]
        check(sec, al, dt, s[f"{sc}/硬串联/d_err"], f"{sc}/{k}+phys")
        add(f"{k}_phys", sec, al)
        add(f"{k}_mid", sec, np.full(n, 0.5))
        sec = s[f"{sc}/本方案/sec"]
        add(f"{k}_flow", sec, al_all[idx, sec])
    if f"bsbl/{sc}/sec" in bsbl.files:
        sec, de = bsbl[f"bsbl/{sc}/sec"][:n], bsbl[f"bsbl/{sc}/d_err"][:n]
        dg = np.stack([root(sec, np.full(n, a)) for a in grid], 1)
        cand = np.stack([dt + de, dt - de], 1)
        resid = np.abs(dg[:, :, None] - cand[:, None, :]).min((1, 2))
        assert resid.max() < 1e-5, f"{sc}/bsbl: 位置不在 10 点网格上，残差 {resid.max()}"
        al = grid[np.abs(dg[:, :, None] - cand[:, None, :]).min(2).argmin(1)]
        check(sec, al, dt, de, f"{sc}/bsbl")
        add("bsbl", sec, al)
    for lab, pre in (("L1(同测点)", "l1"), ("L1(自身布点)", "l1own")):
        f = DATA / f"{pre}_{fn[:-4]}_n{noise}_k5.npz"
        if f.exists():
            c = np.load(f)
            for var, (ks, as_) in (("论文流程", ("k1", "a1")), ("全候选线路细化", ("k2", "a2"))):
                add(f"{lab} {var}", c[ks], c[as_])
    out[sc] = res
    if sc in ("域内", "高阻外推"):
        curves.update({f"{sc}/{k}": v for k, v in paths.items() if k in ("ours", "gcn_d_phys", "gcn_d_mid", "bsbl")})
    print(sc, {k: round(v["mean"], 2) for k, v in res.items() if k in ("ours", "nn_phys", "gcn_d_phys", "gcn_d_mid", "bsbl")}, flush=True)
(R / "path_error.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
np.savez_compressed(R / "path_error_samples.npz", **curves)
