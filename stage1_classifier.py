"""阶段一：稀疏测点突变量 → 区段后验概率（轻量 MLP）。

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
    torch.set_num_threads(1)  # 单线程使训练结果可复现（多线程归约顺序不固定）
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


def probabilities(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
