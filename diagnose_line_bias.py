"""诊断：把每个样本真实的逐线阻抗倍率交给物理模型，量化“个别线路阻抗±20%”场景下的损失有多少来自模型失配。

对比三种判决（阶段一均为最终配置的 MLP）：阶段一 Top-1、名义阻抗物理模型的联合判决、逐线真值物理模型的联合判决；
并给出真实区段已知时的测距误差。仅 IEEE 34。
"""
import argparse
import json
import time
import types
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from evaluate_pipeline import (ALPHAS, CASES, EPS_ABS, EPS_REL, Z_SCALES, Network, add_measurement_noise, features,
                               load_set, logits_of, phys_tag, probabilities, read_sensor_data, train)
from ieee34_bus_impedance import scale_line_yprim
from stage2_continuous_fse import ConstrainedPhysicsModel

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2500)
ap.add_argument("--out", default="results/improved/ieee34/oracle_line_params.json")
args = ap.parse_args()
N = args.n

D = Path("data/generated")
case = CASES["ieee34"]
net = Network(case=case)
rows = net.sensor_rows(case["sensors"])
sigma = net.measurement_sigma(rows)
cls = np.array(json.load(open("results/evaluation.json"))["equivalence_classes"])
lam = json.load(open("results/improved/ieee34/evaluation.json"))["thresholds"]["lam"]

ds = load_set(D / "test_zline.npz", net, rows, sigma, seed=202, limit=5000)
tag = phys_tag(case["sensors"], Z_SCALES, True, 0.05)
cache = np.load(D / f"phys_test_zline_n1.0_{tag}_L5000.npz")
pen_nom, j_nom, alpha_nom = (cache[k][:N] for k in ("pen", "j", "alpha"))

parts = [read_sensor_data(D / f, rows) for f in ["train.npz", "train_z0.88.npz", "train_z0.94.npz", "train_z1.06.npz", "train_z1.12.npz"]]
zc = np.concatenate([p["zc"] for p in parts])
sec_tr = np.concatenate([p["sec"] for p in parts])
rng = np.random.default_rng(0)


def minibatch(b):
    i = rng.integers(0, len(zc), b)
    return features(add_measurement_noise(rng, zc[i], sigma), sigma), sec_tr[i].astype(np.int64)


model = train(minibatch, 3 * len(rows) + 1, len(net.sections), steps=6000)
p = probabilities(logits_of(model, features(ds["z"][:N], sigma)))
cand = np.argsort(-p, 1)[:, :2]

# 每个样本的逐线阻抗倍率：与 simulate_dataset 的偏差抽样一致（样本种子取自 default_rng(seed+10**6)）
seeds = np.random.default_rng(301 + 10 ** 6).integers(0, 2 ** 31, 5000)[:N]
y0 = [s["y"] for s in net.sections]


def work(i):
    r = np.random.default_rng(int(seeds[i]))
    f = [1.0 + 0.2 * r.uniform(-1, 1) for _ in net.sections]
    idxs = sorted(set(cand[i].tolist()) | {int(ds["sec"][i])})
    for k, s in enumerate(net.sections):
        s["y"] = scale_line_yprim(y0[k], f[k])
    lib, zff = [], []
    for k in idxs:
        out = [net.response(k, a, True) for a in ALPHAS]
        lib.append(np.array([o[0] for o in out]))
        zff.append(np.array([o[1] for o in out]))
    sub = types.SimpleNamespace(sections=[net.sections[k] for k in idxs], nodes=net.nodes,
                                prefault_fault_voltage=lambda kk, al: net.prefault_fault_voltage(idxs[kk], al))
    pm = ConstrainedPhysicsModel(sub, [lib], [zff], [0.0], rows, sigma, EPS_REL, EPS_ABS, amp_sigma=0.05)
    z = ds["z"][i:i + 1]
    res = {}
    for kk, k in enumerate(idxs):
        r = pm.fit_section(kk, z)
        res[k] = (float(r[0][0]), float(r[2][0]))
    for k, s in enumerate(net.sections):
        s["y"] = y0[k]
    return res


t0 = time.time()
with Pool(4) as pool:
    outs = pool.map(work, range(N), chunksize=5)
print(f"time {time.time() - t0:.1f}s")

sec_true = ds["sec"][:N]
acc = lambda s: float((cls[s] == cls[sec_true]).mean() * 100)


def joint(pen_of):
    return np.array([cand[i][int(np.argmax([np.log(p[i, k] + 1e-12) - lam * pen_of(i, k) for k in cand[i]]))] for i in range(N)])


d_true = ds["d_true"][:N]
err_nom = np.abs(np.array([net.fault_distance_km(sec_true[i], alpha_nom[i, sec_true[i]]) for i in range(N)]) - d_true)
err_orc = np.abs(np.array([net.fault_distance_km(sec_true[i], outs[i][int(sec_true[i])][1]) for i in range(N)]) - d_true)
result = dict(
    n=N,
    class_acc=dict(stage1_top1=acc(p.argmax(1)), joint_nominal=acc(joint(lambda i, k: pen_nom[i, k])),
                   joint_oracle=acc(joint(lambda i, k: outs[i][k][0]))),
    distance_err_km_true_section=dict(mean_nominal=float(err_nom.mean()), mean_oracle=float(err_orc.mean()),
                                      median_nominal=float(np.median(err_nom)), median_oracle=float(np.median(err_orc))),
)
print(json.dumps(result, ensure_ascii=False, indent=1))
Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
