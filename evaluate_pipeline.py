"""全流程评测：训练阶段一、标定阶段三阈值，在各场景下对比方法并输出 CR、FCR、测距误差与消融结果。

用法：python evaluate_pipeline.py --data data/generated --out results
数据集由 simulate_dataset.py 生成，文件名约定见 SCENARIOS。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from cross_stage_verifier import Thresholds, decide, equivalence_classes, fit_lambda
from ieee34_bus_impedance import SENSORS, Network, add_measurement_noise
from stage1_classifier import (adaptive_candidates, features, fit_temperature, logits_of,
                               probabilities, train)
from stage2_continuous_fse import ALPHAS, PhysicsModel

EPS_REL = 0.02       # 线性化模型误差：相对本量测幅值的部分
EPS_ABS = 0.01       # 线性化模型误差：相对同类量测最大幅值的部分
# 线路阻抗假设（相对算例参数的倍数）：物理反演对每个倍数各算一遍并取残差最小者
Z_SCALES = (0.9, 0.95, 1.0, 1.05, 1.1)
TAG = f"r{EPS_REL}_a{EPS_ABS}_z{'-'.join(map(str, Z_SCALES))}_" + "".join(f"{n}{t}" for n, t in SENSORS)  # 物理残差缓存文件名后缀，参数变化即失效
# 场景名 -> 数据文件
SCENARIOS = {
    "域内": ("test_in.npz", 1.0),
    "负荷偏移": ("test_load.npz", 1.0),
    "高阻外推": ("test_rf.npz", 1.0),
    "噪声×3": ("test_in.npz", 3.0),
    "线路阻抗+5%": ("test_z105.npz", 1.0),
    "线路阻抗+10%": ("test_zscale.npz", 1.0),
    "线路阻抗-10%": ("test_z090.npz", 1.0),
}


def load_set(path, net, rows, sigma, seed, noise_scale=1.0):
    d = np.load(path)
    zc = np.concatenate([d["dv"], d["di"]], 1)[:, rows].astype(complex)
    rng = np.random.default_rng(seed)
    return dict(
        zc=zc, z=add_measurement_noise(rng, zc, sigma, noise_scale),
        sec=d["sec"], alpha=d["alpha"], rf=d["rf"],
        d_true=np.array([net.fault_distance_km(s, a) for s, a in zip(d["sec"], d["alpha"])]),
    )


def physics_cached(pm, z, cache):
    if cache.exists():
        c = np.load(cache)
        return [c[k] for k in ("pen", "j", "alpha", "ftype")]
    out = pm.fit_all(z)
    np.savez_compressed(cache, pen=out[0], j=out[1], alpha=out[2], ftype=out[3])
    return out


def distance(net, sec, alpha):
    return np.array([net.fault_distance_km(s, a) for s, a in zip(sec, alpha)])


def summarize(net, sec, alpha, ds, cls, stage1_top1):
    """区段按等价类判对；CR/FCR 相对阶段一 Top-1 的对错定义。"""
    ok = cls[sec] == cls[ds["sec"]]
    ok1 = cls[stage1_top1] == cls[ds["sec"]]
    err = np.abs(distance(net, sec, alpha) - ds["d_true"])
    return dict(
        class_acc=float(ok.mean()),
        CR=float(ok[~ok1].mean()) if (~ok1).any() else float("nan"),
        FCR=float((~ok[ok1]).mean()) if ok1.any() else float("nan"),
        d_mean=float(err.mean()), d_median=float(np.median(err)), d_p90=float(np.percentile(err, 90)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    net = Network()
    rows = net.sensor_rows(SENSORS)
    sigma = net.measurement_sigma(rows)
    libs = [Network(line_z_scale=z).response_library(ALPHAS) for z in Z_SCALES]
    pm = PhysicsModel(net, libs, rows, sigma, EPS_REL, EPS_ABS)
    n_sec = len(net.sections)

    # ---- 阶段一训练：每个批次重新叠加噪声 ----
    tr = np.load(a.data / "train.npz")
    zc_tr = np.concatenate([tr["dv"], tr["di"]], 1)[:, rows].astype(complex)
    rng = np.random.default_rng(0)

    def make_batch(b):
        i = rng.integers(0, len(zc_tr), b)
        return features(add_measurement_noise(rng, zc_tr[i], sigma), sigma), tr["sec"][i].astype(np.int64)

    t0 = time.time()
    model = train(make_batch, 3 * len(rows) + 1, n_sec, steps=a.steps)
    print(f"stage1 trained in {time.time() - t0:.0f}s", flush=True)

    # ---- 温度校准（名义验证集）与等价类 ----
    val = load_set(a.data / "val.npz", net, rows, sigma, seed=101)
    lg_val = logits_of(model, features(val["z"], sigma))
    temp = fit_temperature(lg_val, val["sec"].astype(np.int64))
    ana = load_set(a.data / "analysis.npz", net, rows, sigma, seed=303)
    pen_a = physics_cached(pm, ana["z"], a.data / f"phys_analysis_{TAG}.npz")[0]
    cls = equivalence_classes(pen_a, ana["sec"])
    print("等价类数", cls.max() + 1, [list(np.where(cls == c)[0]) for c in range(cls.max() + 1) if (cls == c).sum() > 1], flush=True)

    # ---- 阶段三标定：仅名义参数的验证集，或再加入线路阻抗 ±5% 的验证集 ----
    calib_sets = {"名义标定": ["val.npz"], "含阻抗偏差标定": ["val.npz", "val_zp.npz", "val_zm.npz"]}
    cals = {}
    for cal_name, files in calib_sets.items():
        if not all((a.data / f).exists() for f in files):
            continue
        parts = []
        for k, f in enumerate(files):
            d = load_set(a.data / f, net, rows, sigma, seed=101 + k)
            p = probabilities(logits_of(model, features(d["z"], sigma)), temp)
            parts.append((d["sec"], p) + tuple(physics_cached(pm, d["z"], a.data / f"phys_{f[:-4]}_n1.0_{TAG}.npz")))
        sec_c = np.concatenate([x[0] for x in parts])
        p_c, pen_c, j_c, al_c, ft_c = (np.concatenate([x[i] for x in parts]) for i in range(1, 6))
        j_true = j_c[np.arange(len(sec_c)), sec_c]
        th = Thresholds(lam=fit_lambda(p_c, pen_c, cls, sec_c), eps_normal=float(np.quantile(j_true, 0.95)),
                        j_threshold=float(np.quantile(j_true, 0.99)), tau_b=0.0, j_ref=float(np.median(j_true)))
        order_c = np.argsort(-p_c, axis=1)[:, :6]
        ok1_c = cls[order_c[:, 0]] == cls[sec_c]
        best = None
        for tau in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            t = Thresholds(th.lam, th.eps_normal, th.j_threshold, tau, th.j_ref)
            ok = cls[decide(p_c, order_c, pen_c, j_c, al_c, ft_c, cls, t)["sec"]] == cls[sec_c]
            if (~ok[ok1_c]).mean() <= 0.015 and (best is None or ok.mean() > best[0]):
                best = (ok.mean(), t)
        cals[cal_name] = best[1] if best else th
        print(cal_name, cals[cal_name], flush=True)

    results = {"temperature": temp, "thresholds": {k: v.__dict__ for k, v in cals.items()}, "equivalence_classes": cls.tolist()}
    for name, (fn, noise_scale) in SCENARIOS.items():
        if not (a.data / fn).exists():
            continue
        ds = load_set(a.data / fn, net, rows, sigma, seed=202, noise_scale=noise_scale)
        p = probabilities(logits_of(model, features(ds["z"], sigma)), temp)
        pen, j, al, ft = physics_cached(pm, ds["z"], a.data / f"phys_{fn[:-4]}_n{noise_scale}_{TAG}.npz")
        order = np.argsort(-p, axis=1)[:, :6]
        top1 = order[:, 0]
        idx = np.arange(len(p))
        res = {}
        res["基准1 硬串联"] = summarize(net, top1, al[idx, top1], ds, cls, top1)
        res["基准3 物理穷举"] = summarize(net, pen.argmin(1), al[idx, pen.argmin(1)], ds, cls, top1)
        res["基准4 Oracle"] = summarize(net, ds["sec"], al[idx, ds["sec"]], ds, cls, top1)
        res["阶段一Top-K类别召回"] = {k: float((cls[order[:, :k]] == cls[ds["sec"]][:, None]).any(1).mean()) for k in (1, 2, 3, 4, 6)}
        for cal_name, th in cals.items():
            full = (np.log(p + 1e-12) - th.lam * pen).argmax(1)
            res[f"联合打分(全区段) {cal_name}"] = summarize(net, full, al[idx, full], ds, cls, top1)
            for name_v, kw in (("联合打分(无规则)", dict(rules="")), ("联合打分+规则A", dict(rules="A")),
                               ("联合打分+规则A+B", dict(rules="AB")), ("本方案(A+B+C)", dict(rules="ABC"))):
                r = decide(p, order, pen, j, al, ft, cls, th, **kw)
                res[f"{name_v} {cal_name}"] = summarize(net, r["sec"], r["alpha"], ds, cls, top1)
                res[f"{name_v} {cal_name}"]["alarm_rate"] = float((r["rule"] == 4).mean())
                res[f"{name_v} {cal_name}"]["rules"] = {int(k): float((r["rule"] == k).mean()) for k in range(5)}
        th0 = cals["名义标定"]
        res["λ扫描(名义标定,无规则)"] = {}
        for mult in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            t = Thresholds(th0.lam * mult, th0.eps_normal, th0.j_threshold, th0.tau_b, th0.j_ref)
            r = decide(p, order, pen, j, al, ft, cls, t, rules="")
            res["λ扫描(名义标定,无规则)"][str(mult)] = summarize(net, r["sec"], r["alpha"], ds, cls, top1)
        results[name] = res
        print(name, json.dumps(res, ensure_ascii=False, indent=1), flush=True)

    # ---- 推断耗时：1000 个域内样本 ----
    ds = load_set(a.data / "test_in.npz", net, rows, sigma, seed=202)
    z = ds["z"][:1000]
    t0 = time.perf_counter()
    p = probabilities(logits_of(model, features(z, sigma)), temp)
    t1 = time.perf_counter()
    order = np.argsort(-p, axis=1)[:, :4]
    pm.fit_candidates(z, order)
    t2 = time.perf_counter()
    pm.fit_all(z)
    t3 = time.perf_counter()
    results["耗时_ms每样本"] = {"阶段一": (t1 - t0) * 1.0, "阶段二(Top-4候选)": (t2 - t1) * 1.0, "物理穷举(全部区段)": (t3 - t2) * 1.0}
    print(results["耗时_ms每样本"], flush=True)
    (a.out / "evaluation.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
