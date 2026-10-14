"""基准 5：图卷积网络（Chen 等，*Fault Location in Power Distribution Systems via Deep Graph Convolutional Networks*，
IEEE JSAC 2020，arXiv:1812.09464）。无官方代码，按论文描述复现，并适配到本实验的区段分类任务。

与论文一致的部分：
  * 节点为母线，边权由母线间沿线最短距离的高斯核给出：每行保留最近的 Kn 个距离，σ_S 为第 Kn 近距离的均值；
  * 输入 X ∈ R^{母线数×12}：测点母线三相电压、电流相量的（幅值, 相角），未测量的相别与母线置零；各量测按训练集标准化；
  * 切比雪夫谱图卷积（阶数 K 逐层递增）若干层 + 两层全连接（dropout 0.5）+ softmax；训练时每个批次重新叠加噪声。
适配：类别为 30 个故障区段（论文为故障母线）；论文没有位置回归，故障点取预测区段的中点（α=0.5）。
论文输入需要故障前相量，而 train.npz、val.npz 没有保存 v0、i0：用相同参数重新生成（含 v0、i0）后，
由 --prepare 只保留测点量测行，得到 train_gcn.npz、val_gcn.npz（区段、位置、突变量 zc 与故障前相量 pre）。
两种输入：
  * 论文输入（"gcn"）：故障期间的电压、电流相量（故障前相量 + 突变量，叠加量测噪声）；
  * 突变量输入（"gcn_delta"）：与本方案阶段一完全相同的特征（features()），按测点所在母线和相别放到图节点上。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from torch import nn

from evaluate_pipeline import SCENARIOS, load_set
from ieee34_bus_impedance import SENSORS, Network, add_measurement_noise
from stage1_classifier import features

CONV_K = (3, 4, 5)
F_PAPER, F_DELTA = 12, 19


def bus_graph(net, kn):
    """返回 (母线名列表, 缩放后的归一化拉普拉斯 2L/λmax − I)。"""
    buses = sorted(net.adj)
    idx = {b: i for i, b in enumerate(buses)}
    n = len(buses)
    w = np.zeros((n, n))
    for u, nbrs in net.adj.items():
        for v, d in nbrs:
            w[idx[u], idx[v]] = w[idx[v], idx[u]] = max(d, 1e-6)
    s = dijkstra(w, directed=False)
    s_near = np.sort(s, axis=1)[:, :kn]
    sigma = s_near[:, -1].mean()
    adj = np.zeros((n, n))
    for i in range(n):
        near = np.argsort(s[i])[:kn]
        adj[i, near] = np.exp(-(s[i, near] ** 2) / sigma ** 2)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    d = adj.sum(1)
    lap = np.eye(n) - adj / np.sqrt(np.outer(d, d))
    lam_max = np.linalg.eigvalsh(lap).max()
    return buses, (2 * lap / lam_max - np.eye(n)).astype(np.float32)


def sensor_slots(net, buses):
    """量测列（与 sensor_rows 的顺序一致）→ (母线序号, 种类 0 电压/1 电流, 相别 0..2)。"""
    idx = {b: i for i, b in enumerate(buses)}
    slots = []
    for name, term in SENSORS:
        s = net.sections[net.section_index[name]]
        bus = idx[s["bus1"] if term == 0 else s["bus2"]]
        slots += [(bus, kind, ph - 1) for kind in (0, 1) for ph in s["phases"]]
    return slots


class InputBuilder:
    """把一批量测变成 (样本数, 母线数, 特征数) 的图信号，并按训练集标准化。"""

    def __init__(self, net, buses, mode, rows, sigma, train, rng):
        self.mode, self.sigma, self.nb = mode, sigma, len(buses)
        self.slots = sensor_slots(net, buses)
        self.nf = F_PAPER if mode == "gcn" else F_DELTA
        self.mask = np.zeros((self.nb, self.nf), dtype=bool)
        for bus, kind, ph in self.slots:
            self.mask[bus, self._cols(kind, ph)] = True
        if mode == "gcn_delta":
            self.mask[[b for b, _, _ in self.slots], 18] = True
        x = self.build(*self.noisy(train, rng, np.arange(20000), 1.0))
        self.mean = x.mean(0) * self.mask
        self.std = np.where(self.mask, x.std(0) + 1e-8, 1.0)

    def _cols(self, kind, ph):
        if self.mode == "gcn":
            return [6 * kind + 2 * ph, 6 * kind + 2 * ph + 1]
        return [3 * (3 * kind + ph) + c for c in range(3)]

    def noisy(self, d, rng, i, scale):
        """返回 (原始 zc 加噪后, 故障前相量) —— gcn 用后者构造故障期间相量，gcn_delta 用前者。"""
        if self.mode == "gcn":
            post = d["pre"][i] + d["zc"][i]
            post = post + scale * self.sigma / np.sqrt(2) * (rng.standard_normal(post.shape) + 1j * rng.standard_normal(post.shape))
            return post, None
        return add_measurement_noise(rng, d["zc"][i], self.sigma, scale), None

    def build(self, z, _):
        x = np.zeros((len(z), self.nb, self.nf), dtype=np.float32)
        if self.mode == "gcn":
            for k, (bus, kind, ph) in enumerate(self.slots):
                c = self._cols(kind, ph)
                x[:, bus, c[0]] = np.abs(z[:, k])
                x[:, bus, c[1]] = np.angle(z[:, k])
        else:
            f = features(z, self.sigma)
            m = z.shape[1]
            for k, (bus, kind, ph) in enumerate(self.slots):
                c = self._cols(kind, ph)
                x[:, bus, c[0]], x[:, bus, c[1]], x[:, bus, c[2]] = f[:, k], f[:, m + k], f[:, 2 * m + k]
            x[:, [b for b, _, _ in self.slots], 18] = f[:, 3 * m][:, None]
        return x

    def standardize(self, x):
        return ((x - self.mean) / self.std * self.mask).astype(np.float32)


class ChebLayer(nn.Module):
    def __init__(self, n_in, n_out, k):
        super().__init__()
        self.k, self.lin = k, nn.Linear(k * n_in, n_out)

    def forward(self, x, lap):
        t = [x, torch.matmul(lap, x)]
        for _ in range(2, self.k):
            t.append(2 * torch.matmul(lap, t[-1]) - t[-2])
        return self.lin(torch.cat(t[: self.k], -1))


class GCN(nn.Module):
    def __init__(self, lap, n_in, n_classes, filters, hidden=(512, 256)):
        super().__init__()
        self.register_buffer("lap", torch.from_numpy(lap))
        dims = (n_in,) + (filters,) * len(CONV_K)
        self.conv = nn.ModuleList(ChebLayer(dims[i], dims[i + 1], k) for i, k in enumerate(CONV_K))
        n = lap.shape[0] * filters
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(n, hidden[0]), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], n_classes))

    def forward(self, x):
        for layer in self.conv:
            x = torch.relu(layer(x, self.lap))
        return self.fc(x.flatten(1))


def train_gcn(builder, lap, train, n_classes, filters, steps, batch, lr, seed=0):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = GCN(lap, builder.nf, n_classes, filters)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for step in range(steps):
        i = rng.integers(0, len(train["sec"]), batch)
        x = builder.standardize(builder.build(*builder.noisy(train, rng, i, 1.0)))
        loss = loss_fn(model(torch.from_numpy(x)), torch.from_numpy(train["sec"][i].astype(np.int64)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    return model.eval()


@torch.no_grad()
def predict(model, builder, d, noise_scale, seed):
    """对整个数据集（d["zc"] 为无噪量测）加噪后预测区段，噪声种子与评测脚本一致。"""
    rng = np.random.default_rng(seed)
    out = []
    for lo in range(0, len(d["sec"]), 2000):
        i = np.arange(lo, min(lo + 2000, len(d["sec"])))
        x = builder.standardize(builder.build(*builder.noisy(d, rng, i, noise_scale)))
        out.append(model(torch.from_numpy(x)).argmax(1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_probs(model, builder, z):
    """突变量输入：对已带噪的复数量测 z (N, m) 预测区段概率，供其余阶段直接使用同一组观测。"""
    out = []
    for lo in range(0, len(z), 2000):
        x = builder.standardize(builder.build(z[lo:lo + 2000], None))
        out.append(torch.softmax(model(torch.from_numpy(x)), 1).numpy())
    return np.concatenate(out)


def load_raw(path, rows):
    """返回 sec、测点行的突变量 zc 与故障前相量 pre；原始数据集与 --prepare 的输出都可读。"""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    if "zc" not in d:
        d["zc"] = np.concatenate([d["dv"], d["di"]], 1)[:, rows].astype(complex)
        d["pre"] = np.concatenate([d["v0"], d["i0"]], 1)[:, rows].astype(complex)
    return {k: d[k] for k in ("sec", "alpha", "zc", "pre")}


def prepare(src, dst, rows):
    d = load_raw(src, rows)
    np.savez_compressed(dst, **{k: v.astype(np.complex64) if v.dtype.kind == "c" else v for k, v in d.items()})
    print(f"{src} -> {dst}: {len(d['sec'])} 条", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--out", type=Path, default=Path("results/gcn_baseline.json"))
    ap.add_argument("--kn", type=int, default=6, help="图中每个母线保留的最近母线数")
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", nargs="+", default=["gcn", "gcn_delta"])
    ap.add_argument("--val-only", action="store_true", help="只在验证集上评估，用于选择超参数")
    ap.add_argument("--export-probs", type=Path, help="gcn_delta 模式：导出验证集（val、val_gcn）与各场景（与评测脚本相同的带噪观测）上的区段概率")
    ap.add_argument("--prepare", nargs=2, type=Path, metavar=("SRC", "DST"), help="从含 v0、i0 的数据集提取测点量测行")
    a = ap.parse_args()

    net = Network()
    rows = net.sensor_rows(SENSORS)
    if a.prepare:
        prepare(*a.prepare, rows)
        return
    sigma = net.measurement_sigma(rows)
    buses, lap = bus_graph(net, a.kn)
    print(f"图：{len(buses)} 个母线，Kn={a.kn}", flush=True)
    train = load_raw(a.data / "train_gcn.npz", rows)
    cls = np.array(json.loads(Path("results/evaluation.json").read_text())["equivalence_classes"])
    n_sec = len(net.sections)

    results = {"kn": a.kn, "filters": a.filters, "steps": a.steps}
    for mode in a.modes:
        builder = InputBuilder(net, buses, mode, rows, sigma, train, np.random.default_rng(1))
        model = train_gcn(builder, lap, train, n_sec, a.filters, a.steps, a.batch, a.lr)
        res = {}
        if a.val_only:
            d = load_raw(a.data / "val_gcn.npz", rows)
            sec = predict(model, builder, d, 1.0, seed=202)
            res["验证集"] = dict(section_acc=float((sec == d["sec"]).mean()), class_acc=float((cls[sec] == cls[d["sec"]]).mean()))
            print(mode, res["验证集"], flush=True)
        for name in [] if a.val_only else SCENARIOS:
            fn, noise_scale = SCENARIOS[name]
            if not (a.data / fn).exists():
                continue
            d = load_raw(a.data / fn, rows)
            ds = load_set(a.data / fn, net, rows, sigma, seed=202)
            sec = predict(model, builder, d, noise_scale, seed=202)
            d_pred = np.array([net.fault_distance_km(s, 0.5) for s in sec])
            d_orc = np.array([net.fault_distance_km(s, 0.5) for s in ds["sec"]])
            res[name] = dict(
                section_acc=float((sec == ds["sec"]).mean()),
                class_acc=float((cls[sec] == cls[ds["sec"]]).mean()),
                d_mean_midpoint=float(np.abs(d_pred - ds["d_true"]).mean()),
                d_mean_oracle_section_midpoint=float(np.abs(d_orc - ds["d_true"]).mean()),
            )
            print(mode, name, json.dumps(res[name], ensure_ascii=False), flush=True)
        results[mode] = res
        if a.export_probs and mode == "gcn_delta":
            probs = {"val": predict_probs(model, builder, load_set(a.data / "val.npz", net, rows, sigma, seed=101)["z"])}
            for name, (fn, noise_scale) in SCENARIOS.items():
                if (a.data / fn).exists():
                    probs[name] = predict_probs(model, builder, load_set(a.data / fn, net, rows, sigma, seed=202, noise_scale=noise_scale)["z"])
            d = load_raw(a.data / "val_gcn.npz", rows)   # 带故障前相量的验证集，供非线性精化标定
            probs["val_gcn"] = predict_probs(model, builder, add_measurement_noise(np.random.default_rng(101), d["zc"], sigma))
            np.savez_compressed(a.export_probs, **probs)
    if not a.val_only:
        a.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
