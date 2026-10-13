"""阶段一：稀疏测点突变量 → 区段后验概率（轻量 MLP + 温度缩放校准）与自适应 Top-K 截断。

特征对故障电流的未知复幅值不敏感：各分量除以幅值最大的分量，另附各分量相对噪声的信噪比。
"""
import numpy as np
import torch
from torch import nn

HIDDEN = (160, 160, 96)


def features(z, sigma):
    """z: (N, m) 复数量测，sigma: (m,) 噪声标准差。返回 float32 特征 (N, 3m+1)。"""
    ref = np.abs(z).argmax(1)
    z_ref = z[np.arange(len(z)), ref]
    u = z / z_ref[:, None]
    snr = np.log10(np.abs(z) / sigma + 1e-3)
    ref_snr = np.log10(np.abs(z_ref) / sigma[ref] + 1e-3)[:, None]
    return np.concatenate([u.real, u.imag, np.clip(snr, -3, 6), ref_snr], 1).astype(np.float32)


def build_model(n_in, n_classes):
    layers, d = [], n_in
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_classes))
    return nn.Sequential(*layers)


def train(make_batch, n_in, n_classes, steps=6000, batch=512, lr=2e-3, seed=0):
    """make_batch(batch) 返回 (特征 float32, 标签 int64)，每次重新叠加噪声。"""
    torch.manual_seed(seed)
    model = build_model(n_in, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(steps):
        x, y = make_batch(batch)
        loss = loss_fn(model(torch.from_numpy(x)), torch.from_numpy(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def logits_of(model, x):
    return model(torch.from_numpy(x)).numpy()


def fit_temperature(logits, labels):
    """在验证集上拟合温度 T，使 softmax(logits / T) 的负对数似然最小。"""
    lg, y = torch.from_numpy(logits), torch.from_numpy(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


def probabilities(logits, temperature):
    e = np.exp((logits - logits.max(1, keepdims=True)) / temperature)
    return e / e.sum(1, keepdims=True)


def adaptive_candidates(p):
    """按 Top-1 概率确定 K（≥0.5 取 2，0.25~0.5 取 3，否则取 4），返回 (候选区段 (N, 4), K (N,))。"""
    order = np.argsort(-p, axis=1)[:, :4]
    p1 = p[np.arange(len(p)), order[:, 0]]
    k = np.where(p1 >= 0.5, 2, np.where(p1 >= 0.25, 3, 4))
    return order, k
