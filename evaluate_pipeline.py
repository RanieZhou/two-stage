"""全流程评测：训练阶段一、标定阶段三，在各场景下对比方法并输出 CR、FCR、测距误差、消融、置信度与耗时。

数据集由 simulate_dataset.py 生成（目录 data/generated，经 Git LFS 存入仓库），命令如下：
  train.npz      --n 80000 --seed 1  --rf 0.01 300  --mult 0.6 1.1
  val.npz        --n 20000 --seed 2  --rf 0.01 300  --mult 0.6 1.1
  test_in.npz    --n 10000 --seed 3  --rf 0.01 300  --mult 0.6 1.1
  test_load.npz  --n 10000 --seed 4  --rf 0.01 300  --mult 1.25 1.5
  test_rf.npz    --n 10000 --seed 5  --rf 300 1000  --mult 0.6 1.1
  analysis.npz   --n 9000  --seed 6  --rf 0.01 300  --mult 0.6 1.1 --uniform-sections
  test_zscale.npz --n 10000 --seed 7  --rf 0.01 300 --mult 0.6 1.1 --line-z-scale 1.1
  test_z105.npz  --n 10000 --seed 13 --rf 0.01 300 --mult 0.6 1.1 --line-z-scale 1.05
  test_z090.npz  --n 10000 --seed 14 --rf 0.01 300 --mult 0.6 1.1 --line-z-scale 0.9
  与 L1 对比的调压器档位固定场景（均加 --fixed-taps，各 3000 样本）：
  fx_in.npz --seed 21 --rf 0.01 300 --mult 0.6 1.1；fx_load.npz --seed 22 --rf 0.01 300 --mult 1.25 1.5；
  fx_rf.npz --seed 23 --rf 300 1000 --mult 0.6 1.1；fx_zp.npz --seed 24 --line-z-scale 1.1；fx_zm.npz --seed 25 --line-z-scale 0.9
  （fx_zp、fx_zm 的 --rf 0.01 300 --mult 0.6 1.1）
用法：python evaluate_pipeline.py --data data/generated --out results
消融（测点只用电压）：python evaluate_pipeline.py --voltage-only --out results/voltage_only
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from baseline_l1 import TOP_K, L1Locator, greedy_pmu_placement, noisy_dv, run_l1
from cross_stage_verifier import Thresholds, decide, equivalence_classes, fit_lambda
from ieee34_bus_impedance import SENSORS, Network, add_measurement_noise
from stage1_classifier import features, logits_of, probabilities, train
from stage2_continuous_fse import ALPHAS, PhysicsModel

EPS_REL = 0.02       # 线性化模型误差：相对本量测幅值的部分
EPS_ABS = 0.01       # 线性化模型误差：相对同类量测最大幅值的部分
# 线路阻抗假设（相对算例参数的倍数）：物理反演对每个倍数各算一遍，惩罚残差加高斯先验 w·((s-1)/σ)² 后取最小者
Z_SCALES = (0.9, 0.95, 1.0, 1.05, 1.1)
Z_PRIOR_SIGMA = 0.05
Z_PRIOR_WEIGHT = 1.0  # 在 ±5% 阻抗偏差验证集上网格搜索得到
TAG = f"r{EPS_REL}_a{EPS_ABS}_z{'-'.join(map(str, Z_SCALES))}_w{Z_PRIOR_WEIGHT}_" + "".join(f"{n}{t}" for n, t in SENSORS)  # 物理残差缓存后缀
# 场景名 -> (数据文件, 噪声倍数)
SCENARIOS = {
    "域内": ("test_in.npz", 1.0),
    "负荷偏移": ("test_load.npz", 1.0),
    "高阻外推": ("test_rf.npz", 1.0),
    "噪声×3": ("test_in.npz", 3.0),
    "线路阻抗+5%": ("test_z105.npz", 1.0),
    "线路阻抗+10%": ("test_zscale.npz", 1.0),
    "线路阻抗-10%": ("test_z090.npz", 1.0),
    # 调压器档位固定、网络参数精确已知的场景，满足 L1 的前提；同时运行 L1
    "固定档位 域内": ("fx_in.npz", 1.0),
    "固定档位 负荷偏移": ("fx_load.npz", 1.0),
    "固定档位 高阻外推": ("fx_rf.npz", 1.0),
    "固定档位 噪声×3": ("fx_in.npz", 3.0),
    "固定档位 阻抗+10%": ("fx_zp.npz", 1.0),
    "固定档位 阻抗-10%": ("fx_zm.npz", 1.0),
}
L1_PREFIX = "固定档位"


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


def ece(conf, ok):
    """10 等分置信度区间的期望校准误差。"""
    bins = np.minimum((conf * 10).astype(int), 9)
    return float(sum(abs(ok[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))


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


def l1_results(l1, label, net, ds, path, noise_scale, dv_pmu, cls, top1, cache):
    """运行（或读取缓存的）L1，返回阶段一召回与两种阶段二流程的指标；dv_pmu 为 L1 自己 μPMU 处的带噪电压突变量。"""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}  # 先读入内存，避免多进程共享压缩文件句柄
    mi = l1.stage1(dv_pmu)
    if cache.exists():
        c = np.load(cache)
        k1, a1, k2, a2 = (c[k] for k in ("k1", "a1", "k2", "a2"))
    else:
        k1, a1, k2, a2 = run_l1(l1, d, mi, noise_scale)
        np.savez_compressed(cache, k1=k1, a1=a1, k2=k2, a2=a2)
    order = np.argsort(mi, axis=1)
    return {
        f"{label} 阶段一Top-K类别召回": {k: float((cls[order[:, :k]] == cls[ds["sec"]][:, None]).any(1).mean()) for k in (1, 2, 5)},
        f"{label} 论文流程": summarize(net, k1, a1, ds, cls, top1),
        f"{label} 全候选线路细化": summarize(net, k2, a2, ds, cls, top1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--voltage-only", action="store_true", help="消融：测点只用电压相量（不用线路电流），不运行 L1")
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    net = Network()
    rows = net.sensor_rows(SENSORS)
    tag = TAG
    if a.voltage_only:
        rows = rows[rows < len(net.nodes)]
        tag += "_V"
    else:
        # L1 两种测点：与本方案相同的 5 个测点母线；L1 按自身布点目标（前 k 命中率）选取的 5 个母线
        l1_same = L1Locator(net, [net.nodes[r] for r in rows if r < len(net.nodes)])
        with np.load(a.data / "analysis.npz") as f:
            own_nodes, own_buses = greedy_pmu_placement(net, f["dv"][:3000].astype(complex), f["sec"][:3000])
        l1_own = L1Locator(net, own_nodes)
        print("L1 自身布点母线", own_buses, flush=True)
    sigma = net.measurement_sigma(rows)
    isv = rows < len(net.nodes)
    libs = [Network(line_z_scale=z).response_library(ALPHAS) for z in Z_SCALES]
    prior_pen = [Z_PRIOR_WEIGHT * ((z - 1) / Z_PRIOR_SIGMA) ** 2 for z in Z_SCALES]
    pm = PhysicsModel(net, libs, prior_pen, rows, sigma, EPS_REL, EPS_ABS)
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

    # ---- 等价类 ----
    val = load_set(a.data / "val.npz", net, rows, sigma, seed=101)
    lg_val = logits_of(model, features(val["z"], sigma))
    ana = load_set(a.data / "analysis.npz", net, rows, sigma, seed=303)
    pen_a = physics_cached(pm, ana["z"], a.data / f"phys_analysis_{tag}.npz")[0]
    cls = equivalence_classes(pen_a, ana["sec"])
    print("等价类数", cls.max() + 1, [list(np.where(cls == c)[0]) for c in range(cls.max() + 1) if (cls == c).sum() > 1], flush=True)

    # ---- 阶段三标定：名义验证集 ----
    p_val = probabilities(lg_val)
    pen_v, j_v, al_v, ft_v = physics_cached(pm, val["z"], a.data / f"phys_val_n1.0_{tag}.npz")
    j_true = j_v[np.arange(len(val["sec"])), val["sec"]]
    th = Thresholds(lam=fit_lambda(p_val, pen_v, cls, val["sec"]), j_threshold=float(np.quantile(j_true, 0.99)))
    print("阶段三参数", th, flush=True)

    results = {"thresholds": th.__dict__, "equivalence_classes": cls.tolist()}
    if a.voltage_only:  # 仅电压的等价类更粗，另按含电流配置的 22 类划分统计区段准确率，便于与其余方法同口径比较
        cls_ref = np.array(json.loads((Path("results") / "evaluation.json").read_text())["equivalence_classes"])
    for name, (fn, noise_scale) in SCENARIOS.items():
        if not (a.data / fn).exists():
            continue
        ds = load_set(a.data / fn, net, rows, sigma, seed=202, noise_scale=noise_scale)
        lg = logits_of(model, features(ds["z"], sigma))
        p = probabilities(lg)
        pen, j, al, ft = physics_cached(pm, ds["z"], a.data / f"phys_{fn[:-4]}_n{noise_scale}_{tag}.npz")
        order = np.argsort(-p, axis=1)
        top1 = order[:, 0]
        idx = np.arange(len(p))
        res = {}
        res["基准1 硬串联"] = summarize(net, top1, al[idx, top1], ds, cls, top1)
        res["基准3 物理穷举"] = summarize(net, pen.argmin(1), al[idx, pen.argmin(1)], ds, cls, top1)
        res["基准4 Oracle"] = summarize(net, ds["sec"], al[idx, ds["sec"]], ds, cls, top1)
        res["阶段一Top-K类别召回"] = {k: float((cls[order[:, :k]] == cls[ds["sec"]][:, None]).any(1).mean()) for k in (1, 2, 3, 4, 6)}
        full = (np.log(p + 1e-12) - th.lam * pen).argmax(1)
        res["联合打分(全区段,参考)"] = summarize(net, full, al[idx, full], ds, cls, top1)
        for name_v, kw in (("联合打分(Top-2,无候选扩展)", dict(expand=False)), ("本方案(联合打分+候选扩展)", {})):
            r = decide(p, pen, j, al, ft, cls, th, **kw)
            res[name_v] = summarize(net, r["sec"], r["alpha"], ds, cls, top1)
            res[name_v]["rules"] = {int(k): float((r["rule"] == k).mean()) for k in range(3)}
            res[name_v]["ECE"] = ece(r["conf"], cls[r["sec"]] == cls[ds["sec"]])
        r = decide(p, pen, j, al, ft, cls, th)
        ok, hi = cls[r["sec"]] == cls[ds["sec"]], r["conf"] >= 0.6
        if a.voltage_only:
            res["基准1 硬串联"]["class_acc_22类"] = float((cls_ref[top1] == cls_ref[ds["sec"]]).mean())
            res["本方案(联合打分+候选扩展)"]["class_acc_22类"] = float((cls_ref[r["sec"]] == cls_ref[ds["sec"]]).mean())
        res["置信度C"] = dict(高置信占比=float(hi.mean()), 高置信准确率=float(ok[hi].mean()) if hi.any() else float("nan"),
                            低置信准确率=float(ok[~hi].mean()) if (~hi).any() else float("nan"), ECE=ece(r["conf"], ok))
        if name.startswith(L1_PREFIX) and not a.voltage_only:
            with np.load(a.data / fn) as f:
                dv_own = noisy_dv(net, {"dv": f["dv"]}, l1_own.pmu_idx, noise_scale, seed=303)
            res.update(l1_results(l1_same, "L1(同测点)", net, ds, a.data / fn, noise_scale, ds["z"][:, isv], cls, top1,
                                  a.data / f"l1_{fn[:-4]}_n{noise_scale}_k{TOP_K}.npz"))
            res.update(l1_results(l1_own, "L1(自身布点)", net, ds, a.data / fn, noise_scale, dv_own, cls, top1,
                                  a.data / f"l1own_{fn[:-4]}_n{noise_scale}_k{TOP_K}.npz"))
        res["λ扫描(Top-2,无候选扩展)"] = {}
        for mult in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            r = decide(p, pen, j, al, ft, cls, Thresholds(th.lam * mult, th.j_threshold), expand=False)
            res["λ扫描(Top-2,无候选扩展)"][str(mult)] = summarize(net, r["sec"], r["alpha"], ds, cls, top1)
        results[name] = res
        print(name, json.dumps(res, ensure_ascii=False, indent=1), flush=True)

    # ---- 推断耗时：1000 个域内样本 ----
    ds = load_set(a.data / "test_in.npz", net, rows, sigma, seed=202)
    z = ds["z"][:1000]
    t0 = time.perf_counter()
    p = probabilities(logits_of(model, features(z, sigma)))
    t1 = time.perf_counter()
    order = np.argsort(-p, axis=1)[:, :2]
    pm.fit_candidates(z, order)
    t2 = time.perf_counter()
    pm.fit_all(z)
    t3 = time.perf_counter()
    results["耗时_ms每样本"] = {"阶段一": (t1 - t0) * 1.0, "阶段二(Top-2候选)": (t2 - t1) * 1.0, "物理穷举(全部区段)": (t3 - t2) * 1.0}
    print(results["耗时_ms每样本"], flush=True)
    (a.out / "evaluation.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
