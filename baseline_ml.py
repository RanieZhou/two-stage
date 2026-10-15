"""机器学习基线：PCA+SVM、PCA+随机森林（Chen 等 JSAC 论文中的两个对照协议：先 PCA 降维，再用 SVM 或随机森林分类）。

输入与本方案阶段一相同的特征（features：各量测除以幅值最大的量测的实部与虚部、对数信噪比），标准化后做 PCA（保留 99% 的方差），
训练集相同（含阻抗扩增样本）并逐样本叠加量测噪声。只输出故障区段，故障点取预测区段的中点。
区段概率可导出，供 evaluate_pipeline.py --stage1-probs 作阶段一。
SVM（RBF 核）的训练样本数受核方法的计算量限制，取训练集的随机子样本（--svm-train）；决策函数经单一温度标定为概率。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_pipeline import NOISE_LEVELS, SCENARIOS, load_set
from ieee34_bus_impedance import CASES, Network, add_measurement_noise
from simulate_dataset import read_sensor_data
from stage1_classifier import features

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["ieee34", "ieee123"], default="ieee34")
ap.add_argument("--data", type=Path, default=Path("data/generated"))
ap.add_argument("--models", nargs="+", default=["pca_svm", "pca_rf"])
ap.add_argument("--train-extra", nargs="+", default=[])
ap.add_argument("--draws", type=int, default=2, help="每个训练样本叠加噪声的次数")
ap.add_argument("--svm-train", type=int, default=30000)
ap.add_argument("--trees", type=int, default=150)
ap.add_argument("--noise-levels", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True, help="结果 json；逐样本区段预测另存为同名 _samples.npz，区段概率存为 data/generated/<前缀>ml_<模型>_probs.npz")
a = ap.parse_args()

case = CASES[a.case]
fp = (lambda n: a.data / n) if a.case == "ieee34" else (lambda n: a.data / f"{a.case}_{n}")
pre = "" if a.case == "ieee34" else f"{a.case}_"
net = Network(case=case)
rows = net.sensor_rows(case["sensors"])
sigma = net.measurement_sigma(rows)
cls = np.array(json.loads((Path("results/evaluation.json") if a.case == "ieee34" else Path(f"results/{a.case}/evaluation.json")).read_text())["equivalence_classes"])

tr = read_sensor_data(fp("train.npz"), rows)
for name in a.train_extra:
    e = read_sensor_data(a.data / name, rows)
    tr = {k: np.concatenate([tr[k], e[k]]) for k in ("sec", "zc")}
rng = np.random.default_rng(a.seed)
x_tr = np.concatenate([features(add_measurement_noise(rng, tr["zc"], sigma), sigma) for _ in range(a.draws)])
y_tr = np.tile(tr["sec"].astype(np.int64), a.draws)
scaler = StandardScaler().fit(x_tr)
pca = PCA(n_components=0.99, random_state=a.seed).fit(scaler.transform(x_tr))
print("训练样本", x_tr.shape, "PCA 维数", pca.n_components_, flush=True)
reduce = lambda x: pca.transform(scaler.transform(x))
r_tr = reduce(x_tr)

val = load_set(fp("val.npz"), net, rows, sigma, seed=101)
x_val = features(val["z"], sigma)
scen = {**SCENARIOS, **(NOISE_LEVELS if a.noise_levels else {})}
sets = {n: load_set(fp(fn), net, rows, sigma, seed=202, noise_scale=ns) for n, (fn, ns) in scen.items() if fp(fn).exists()}
x_sets = {n: features(d["z"], sigma) for n, d in sets.items()}
n_sec = len(net.sections)

results, samples = {}, {}
for model in a.models:
    t0 = time.time()
    if model == "pca_svm":
        idx = np.random.default_rng(a.seed).choice(len(r_tr), min(a.svm_train, len(r_tr)), replace=False)
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", decision_function_shape="ovr").fit(r_tr[idx], y_tr[idx])

        def score(x):
            d = np.full((len(x), n_sec), -1e3)
            d[:, clf.classes_] = clf.decision_function(reduce(x))
            return d

        sv = score(x_val[:5000])
        temp = minimize_scalar(lambda t: -np.log(softmax(sv * np.exp(t), 1)[np.arange(5000), val["sec"][:5000]] + 1e-12).mean(), bounds=(-3, 6), method="bounded").x
        proba = lambda x: softmax(score(x) * np.exp(temp), 1)
    else:
        clf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=3, n_jobs=4, random_state=a.seed).fit(r_tr, y_tr)

        def proba(x):
            out = np.zeros((len(x), n_sec))
            out[:, clf.classes_] = clf.predict_proba(reduce(x))
            return out
    print(model, f"训练 {time.time() - t0:.0f}s", flush=True)
    res, probs = {}, {"val": proba(x_val)}
    for n, d in sets.items():
        p = proba(x_sets[n])
        probs[n] = p
        sec = p.argmax(1)
        d_pred = np.array([net.fault_distance_km(s, 0.5) for s in sec])
        res[n] = dict(section_acc=float((sec == d["sec"]).mean()), class_acc=float((cls[sec] == cls[d["sec"]]).mean()),
                      d_mean_midpoint=float(np.abs(d_pred - d["d_true"]).mean()))
        samples[f"{model}/{n}/sec"] = sec
        print(model, n, json.dumps(res[n], ensure_ascii=False), flush=True)
    results[model] = res
    np.savez_compressed(a.data / f"{pre}ml_{model}_probs.npz", **probs)
a.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
np.savez_compressed(a.out.with_name(a.out.stem + "_samples.npz"), **samples)
