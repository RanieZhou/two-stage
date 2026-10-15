"""论文插图：python make_figures.py [--lang zh|en]，输出 figures/<lang>/figN.{pdf,png}。

数据来源：results/improved/ 下的 breakdown.json、tier1.json、multi_seed.json、samples.npz、stage1_swap/、baselines_*_samples.npz；
图 1、图 2 为示意图，图 2 的拓扑来自算例（以变电站为根的树形布局，横轴为沿馈线到变电站的距离）。
配色为 dataviz 参考调色板的前几个类别色（按固定顺序分配给方法，不随图变化），同一方法在所有图中颜色与标记一致。
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ap = argparse.ArgumentParser()
ap.add_argument("--lang", choices=["zh", "en"], default="zh")
ap.add_argument("--only", nargs="*", type=int, help="只生成指定编号的图")
a = ap.parse_args()
zh = a.lang == "zh"
T = lambda z, e: z if zh else e
R = Path("results/improved/ieee34")
OUT = Path("figures") / a.lang
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdad5", "#ffffff"
COL = {"本方案": "#2a78d6", "GCN": "#eb6834", "GAT": "#1baf7a", "块稀疏贝叶斯": "#eda100", "PCA+SVM": "#e87ba4", "PCA+随机森林": "#6250d6",
       "神经网络 Top-1": "#8a8984", "物理穷举": "#008300"}
MK = {"本方案": "o", "GCN": "s", "GAT": "^", "块稀疏贝叶斯": "D", "PCA+SVM": "v", "PCA+随机森林": "P", "神经网络 Top-1": "x", "物理穷举": "+"}
EN = {"本方案": "Ours", "GCN": "GCN", "GAT": "GAT", "块稀疏贝叶斯": "Block-sparse Bayesian", "PCA+SVM": "PCA+SVM", "PCA+随机森林": "PCA+RF",
      "神经网络 Top-1": "Neural network Top-1", "物理穷举": "Exhaustive physics"}
name = lambda k: k if zh else EN[k]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"] if zh else ["DejaVu Sans"],
    "axes.unicode_minus": False, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.linewidth": 0.8, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, n):
    fig.savefig(OUT / f"fig{n}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"fig{n}.png")


def want(n):
    return not a.only or n in a.only


# ---------------------------------------------------------------- 图 1：方法框图
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=INK, linespacing=1.4)

    def arrow(x1, y1, x2, y2, head=True):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>" if head else "-", mutation_scale=9, lw=1.0, color=INK2))
    box(1, 17, 16, 12, T("测点量测\n故障前后突变量 z", "Sensor data\ndisturbance z"), fc="#f1f0ec")
    box(22, 17, 20, 12, T("阶段一：神经网络\n区段概率 P(s|z)\n取 Top-2 候选", "Stage 1: neural network\nP(s|z)\nTop-2 candidates"), fc="#e4eefb")
    box(47, 17, 24, 12, T("阶段二：物理反演\n过渡电阻为实数\n残差 J、位置 α", "Stage 2: physics inversion\nreal fault resistance\nresidual J, position α"), fc="#e6f4ee")
    box(76, 17, 22, 12, T("阶段三：联合打分\nS = ln P - λJ\n判决区段", "Stage 3: joint score\nS = ln P - λJ\nsection decision"), fc="#fdeee6")
    arrow(17.5, 23, 21.5, 23); arrow(42.5, 23, 46.5, 23); arrow(71.5, 23, 75.5, 23)
    arrow(9, 16.5, 9, 7, head=False); arrow(9, 7, 59, 7, head=False); arrow(59, 7, 59, 16.4)
    ax.text(34, 8.4, T("z 同时送入阶段二", "z also feeds stage 2"), fontsize=7, color=INK2, ha="center")
    arrow(87, 29.6, 87, 33.6)
    box(48, 34, 50, 8, T("候选残差均偏大 → 扩展到全部区段；仍偏大 → 失配告警", "All candidate residuals above threshold → expand to all sections;\nstill large → mismatch alarm"), fc="#f7f6f3")
    arrow(87, 16.4, 87, 8.6)
    ax.text(87, 4.5, T("输出：区段、位置、置信度", "Output: section, position, confidence"), fontsize=7, color=INK, ha="center")
    save(fig, 1)


# ---------------------------------------------------------------- 图 2：拓扑与测点
def fig2():
    from ieee34_bus_impedance import CASES, Network
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.3]})
    for ax, key, title in zip(axes, ("ieee34", "ieee123"), (T("IEEE 34 节点（5 个测点，30 个区段）", "IEEE 34-bus (5 sensors, 30 sections)"), T("IEEE 123 节点（8 个测点，118 个区段）", "IEEE 123-bus (8 sensors, 118 sections)"))):
        case = CASES[key]
        net = Network(case=case)
        root = case["root"]
        parent, order, dist = {root: None}, [root], {root: 0.0}
        children = {}
        for u in order:
            for v, d in sorted(net.adj.get(u, []), key=lambda x: x[0]):
                if v not in parent:
                    parent[v] = u; dist[v] = dist[u] + max(d, 0.05); order.append(v); children.setdefault(u, []).append(v)
        y, nxt = {}, [0]

        def place(u):
            kids = children.get(u, [])
            if not kids:
                y[u] = nxt[0]; nxt[0] += 1
            else:
                for k in kids:
                    place(k)
                y[u] = float(np.mean([y[k] for k in kids]))
        import sys
        sys.setrecursionlimit(10000)
        place(root)
        for u, p in parent.items():
            if p is not None:
                ax.plot([dist[p], dist[u], dist[u]], [y[p], y[p], y[u]], color="#9a9893", lw=0.7, zorder=1)
        sens = []
        for ln, term in case["sensors"]:
            s = net.sections[net.section_index[ln]]
            sens.append(s["bus1"] if term == 0 else s["bus2"])
        ax.scatter([dist[b] for b in sens], [y[b] for b in sens], s=46, color=COL["本方案"], edgecolor="white", linewidth=1.0, zorder=3, label=T("测点母线", "sensor bus"))
        ax.scatter([dist[root]], [y[root]], marker="s", s=60, color=INK, zorder=3, label=T("变电站", "substation"))
        for b in sens:
            ax.annotate(b, (dist[b], y[b]), xytext=(4, 5), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel(T("沿馈线到变电站的距离 (km)", "Distance from substation along the feeder (km)"))
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        ax.legend(loc="upper right", fontsize=7)
    save(fig, 2)


# ---------------------------------------------------------------- 图 3：过渡电阻与噪声
def best_input(t1, base, scs):
    """GCN / GAT 有两种输入：取域内类准确率较高者，使每个方法在图中只有一条线。"""
    cand = [f"{base} 论文输入", f"{base} 突变量输入"]
    return max(cand, key=lambda k: t1["故障类型 全部"][k]["class_acc"])


def fig3():
    bd = json.load(open(R / "breakdown.json"))["过渡电阻曲线"]
    t1 = json.load(open(R / "tier1.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    bins = list(bd)
    xs = np.arange(len(bins))
    for key, mk in (("硬串联", "神经网络 Top-1"), ("物理穷举", "物理穷举"), ("本方案", "本方案")):
        ys = [bd[b][key]["class_acc"] * 100 for b in bins]
        ax.plot(xs, ys, color=COL[mk], marker=MK[mk], ms=5, lw=1.8 if mk == "本方案" else 1.4, label=name(mk), mfc=COL[mk] if mk in ("本方案",) else "none")
    ax.axvline(3.5, color=INK2, lw=0.8, ls=":")
    ax.text(3.6, 52, T("训练范围外", "beyond training"), fontsize=7, color=INK2)
    ax.text(3.4, 52, T("训练范围内", "within training"), fontsize=7, color=INK2, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([b.replace("Rf ", "").replace(" Ω", "") for b in bins], fontsize=7)
    ax.set_xlabel(T("过渡电阻 (Ω)", "Fault resistance (Ω)")); ax.set_ylabel(T("类准确率 (%)", "Class accuracy (%)"))
    ax.set_ylim(50, 100); ax.set_title("(a)", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    levels = [0.5, 1, 2, 3, 5]
    series = {"本方案": "本方案", "GCN": best_input(t1, "GCN", None), "GAT": best_input(t1, "GAT", None), "块稀疏贝叶斯": "块稀疏贝叶斯学习", "PCA+SVM": "PCA+SVM", "PCA+随机森林": "PCA+随机森林"}
    for mk, key in series.items():
        ys = [t1[f"噪声 ×{l:g}"][key]["class_acc"] * 100 for l in levels]
        lab = name(mk) + ("" if mk in ("本方案", "块稀疏贝叶斯", "PCA+SVM", "PCA+随机森林") else f" ({key.split(' ')[1][:2] if zh else ('paper' if '论文' in key else 'disturbance')})")
        ax.plot(range(len(levels)), ys, color=COL[mk], marker=MK[mk], ms=5, lw=1.8 if mk == "本方案" else 1.2, label=lab)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"×{l:g}" for l in levels])
    ax.set_xlabel(T("量测噪声倍数", "Measurement noise multiplier")); ax.set_title("(b)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left", ncol=1)
    save(fig, 3)


# ---------------------------------------------------------------- 图 4：测距误差累积分布
def fig4():
    ours = np.load(R / "samples.npz")
    gcn = np.load(R / "stage1_swap/gcn_delta/samples.npz")
    gp = np.load(R / "stage1_swap/gcn_paper/samples.npz")
    bs = np.load(R / "baselines_bsbl_samples.npz")
    gcn_s = np.load(R / "baselines_gcn_samples.npz")
    from ieee34_bus_impedance import CASES, Network
    net = Network(case=CASES["ieee34"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, sc, ttl in zip(axes, ("域内", "高阻外推"), (T("域内", "In-domain"), T("高阻外推（300–1000 Ω）", "High resistance (300–1000 Ω)"))):
        n = len(ours[f"{sc}/sec_true"])
        d_true = ours[f"{sc}/d_true"]
        mid = np.abs(np.array([net.fault_distance_km(s, 0.5) for s in gcn_s[f"gcn_delta/{sc}/sec"][:n]]) - d_true)
        curves = [("本方案", ours[f"{sc}/本方案/d_err"], "-"), ("GCN+" + T("物理定位", "physics location"), gcn[f"{sc}/硬串联/d_err"], "--"),
                  ("GCN (" + T("区段中点", "section midpoint") + ")", mid, ":"), ("块稀疏贝叶斯", bs[f"bsbl/{sc}/d_err"][:n], "-.")]
        cols = [COL["本方案"], COL["GCN"], COL["GCN"], COL["块稀疏贝叶斯"]]
        for (lab, e, ls), c in zip(curves, cols):
            e = np.sort(e)
            ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, color=c, ls=ls, lw=1.8 if lab == "本方案" else 1.3, label=name(lab) if lab in EN else lab)
        ax.set_xlim(0, 8); ax.set_xlabel(T("测距误差 (km)", "Location error (km)")); ax.set_title(ttl, loc="left", fontsize=9)
    axes[0].set_ylabel(T("累积比例 (%)", "Cumulative share (%)"))
    axes[0].legend(fontsize=7, loc="lower right")
    save(fig, 4)


# ---------------------------------------------------------------- 图 5：外推场景分组柱状图
def fig5():
    ms = json.load(open(R / "multi_seed.json"))
    scs = ["负荷偏移", "高阻外推", "噪声×3", "线路阻抗+10%", "线路阻抗-10%", "个别线路阻抗±20%"]
    en_sc = {"负荷偏移": "Load shift", "高阻外推": "High R", "噪声×3": "Noise ×3", "线路阻抗+10%": "Line Z +10%", "线路阻抗-10%": "Line Z −10%", "个别线路阻抗±20%": "Line Z ±20% (indiv.)"}
    meths = [("本方案", "本方案"), ("GCN", ("GCN(论文输入)", "GCN(突变量输入)")), ("GAT", ("GAT(论文输入)", "GAT(突变量输入)")), ("PCA+SVM", "PCA+SVM"), ("PCA+随机森林", "PCA+随机森林")]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    w = 0.16
    for i, (mk, key) in enumerate(meths):
        vals, errs = [], []
        for s in scs:
            keys = [key] if isinstance(key, str) else list(key)
            best = max((ms[s][k] for k in keys if k in ms[s]), key=lambda x: x["mean"])
            vals.append(best["mean"] * 100); errs.append(best["std"] * 100)
        ax.bar(np.arange(len(scs)) + (i - 2) * w, vals, w * 0.92, yerr=errs, color=COL[mk], label=name(mk), error_kw=dict(lw=0.8, ecolor=INK2, capsize=1.5))
    ax.set_xticks(np.arange(len(scs))); ax.set_xticklabels([s if zh else en_sc[s] for s in scs], fontsize=7.5)
    ax.set_ylabel(T("类准确率 (%)", "Class accuracy (%)")); ax.set_ylim(30, 100)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, 5)


for n, f in ((1, fig1), (2, fig2), (3, fig3), (4, fig4), (5, fig5)):
    if want(n):
        f()
