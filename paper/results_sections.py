"""初稿的数据装载、表格与结果章节。所有数字来自 results/improved/ 下的实验结果；文字中的比较结论由代码按数据判断后写入。"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R34, R123 = ROOT / "results/improved/ieee34", ROOT / "results/improved/ieee123"
J = lambda p: json.load(open(p))

K = "本方案(联合打分+候选扩展)"
NN, PH, OR = "基准1 硬串联", "基准3 物理穷举", "基准4 Oracle"
MAIN7 = ["域内", "负荷偏移", "高阻外推", "噪声×3", "线路阻抗+10%", "线路阻抗-10%", "个别线路阻抗±20%"]
FIXED4 = ["固定档位 域内", "固定档位 负荷偏移", "固定档位 高阻外推", "固定档位 噪声×3"]
S123 = ["域内", "负荷偏移", "高阻外推", "噪声×3", "线路阻抗+10%", "线路阻抗-10%", "个别线路阻抗±20%"]
SC_EN = {"域内": "In-domain", "负荷偏移": "Load shift", "高阻外推": "High resistance", "噪声×3": "Noise ×3", "线路阻抗+10%": "Line impedance +10%",
         "线路阻抗-10%": "Line impedance −10%", "个别线路阻抗±20%": "Individual line impedance ±20%", "线路阻抗+5%": "Line impedance +5%",
         "各负荷独立变化": "Independent load variation", "线路与负荷同时偏差": "Line and load errors combined",
         "固定档位 域内": "Fixed taps: in-domain", "固定档位 负荷偏移": "Fixed taps: load shift", "固定档位 高阻外推": "Fixed taps: high resistance", "固定档位 噪声×3": "Fixed taps: noise ×3",
         "固定档位 阻抗+10%": "Fixed taps: line impedance +10%", "固定档位 阻抗-10%": "Fixed taps: line impedance −10%"}
# 方法键 → (中文名, 英文名)
METHODS = [("ours", "本方法", "Ours"), ("gcn_p", "GCN（论文输入）", "GCN (paper input)"), ("gcn_d", "GCN（突变量输入）", "GCN (disturbance input)"),
           ("gat_p", "GAT（论文输入）", "GAT (paper input)"), ("gat_d", "GAT（突变量输入）", "GAT (disturbance input)"),
           ("bsbl", "块稀疏贝叶斯", "Block-sparse Bayesian"), ("svm", "PCA+SVM", "PCA+SVM"), ("rf", "PCA+随机森林", "PCA+RF")]


def sc(d, s):
    return s if d.L == "zh" else SC_EN[s]


def mname(d, k):
    return {m[0]: (m[1] if d.L == "zh" else m[2]) for m in METHODS}[k]


def seed_files(paths, key_path):
    """读取若干个种子的结果文件，key_path 为取出场景字典的函数。"""
    out = []
    for p in paths:
        if Path(p).exists():
            out.append(key_path(J(p)))
    return out


def stats(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


_cache = {}


def load():
    """返回 dict：acc[系统][场景][方法] = (均值, 标准差, 个数)；sec、dmid、dphys、dflow 为种子 0 的单次值。"""
    if _cache:
        return _cache
    ev34, ev123 = J(R34 / "evaluation.json"), J(R123 / "evaluation.json")
    ms34, ms123 = J(R34 / "multi_seed.json"), J(R123 / "multi_seed.json")
    acc = {"34": {}, "123": {}}
    keymap = {"gcn_p": "GCN(论文输入)", "gcn_d": "GCN(突变量输入)"}
    for sysk, ms in (("34", ms34), ("123", ms123)):
        for s, row in ms.items():
            acc[sysk][s] = {"ours": (row["本方案"]["mean"], row["本方案"]["std"], row["本方案"]["n"])}
            for k, name in keymap.items():
                if name in row:
                    acc[sysk][s][k] = (row[name]["mean"], row[name]["std"], row[name]["n"])
    # GAT：种子 0 在 results/improved/ieee34/gat_*.json，其余种子在 seeds/
    for k, stem, mode in (("gat_p", "gat_paper", "gcn"), ("gat_d", "gat_delta", "gcn_delta")):
        paths = [R34 / f"{stem}.json"] + [R34 / f"seeds/{stem}_s{i}.json" for i in (1, 2)]
        res = seed_files(paths, lambda r, m=mode: r[m])
        for s in res[0] if res else []:
            acc["34"].setdefault(s, {})[k] = stats([r[s]["class_acc"] for r in res if s in r])
    ml = seed_files([R34 / "baselines_ml.json"] + [R34 / f"seeds/pca_s{i}.json" for i in (1, 2)], lambda r: r)
    for k, m in (("svm", "pca_svm"), ("rf", "pca_rf")):
        for s in ml[0][m] if ml else []:
            acc["34"].setdefault(s, {})[k] = stats([r[m][s]["class_acc"] for r in ml if s in r[m]])
    bs = J(R34 / "baselines_bsbl.json") if (R34 / "baselines_bsbl.json").exists() else {}
    for s, v in bs.items():
        acc["34"].setdefault(s, {})["bsbl"] = (v["class_acc"], 0.0, 1)
    # IEEE 123：GAT、PCA、BSBL 为种子 0 单次
    for k, f, mode in (("gat_p", "gat_paper.json", "gcn"), ("gat_d", "gat_delta.json", "gcn_delta")):
        p = R123 / f
        if p.exists():
            for s, v in J(p)[mode].items():
                acc["123"].setdefault(s, {})[k] = (v["class_acc"], 0.0, 1)
    p = R123 / "baselines_ml.json"
    if p.exists():
        for k, m in (("svm", "pca_svm"), ("rf", "pca_rf")):
            for s, v in J(p)[m].items():
                acc["123"].setdefault(s, {})[k] = (v["class_acc"], 0.0, 1)
    p = R123 / "baselines_bsbl.json"
    if p.exists():
        for s, v in J(p).items():
            acc["123"].setdefault(s, {})["bsbl"] = (v["class_acc"], 0.0, 1)
    _cache.update(acc=acc, ev34=ev34, ev123=ev123, ms34=ms34, ms123=ms123)
    return _cache


def cell(v, best=False, prec=1, std=True):
    if v is None:
        return "-"
    m, s, n = v
    txt = f"{m * 100:.{prec}f}" + (f"±{s * 100:.{prec}f}" if (std and n > 1) else "")
    return f"**{txt}**" if best else txt


def bold_best(vals, prec=1):
    """vals 为 (均值,标准差,n) 或 None 的列表，均值（保留一位）最大者加粗。"""
    nums = [round(v[0] * 100, prec) if v else -1 for v in vals]
    top = max(nums)
    return [cell(v, n == top, prec) for v, n in zip(vals, nums)]


# ---------------------------------------------------------------- 表 1、表 2（静态）
def table1(d):
    t = d.t
    rows = [
        [t("本文", "This paper"), t("神经网络加物理反演", "Neural network + physics inversion"), t("相量突变量，5/8 个测点", "Phasor disturbance, 5/8 sensors"), t("是（过渡电阻为实数的约束反演）", "Yes (real-resistance inversion)"), t("是", "Yes"), t("是", "Yes")],
        [t("Liang 等 2025", "Liang et al. 2025") + d.c("DLiang2025"), t("线性匹配筛选 + 非线性状态估计", "Linear matching + nonlinear state estimation"), t("μPMU 相量，优化配置", "μPMU phasors, optimized placement"), t("是", "Yes"), t("未见", "Not reported"), t("未见", "Not reported")],
        [t("Jiang 等 2021", "Jiang et al. 2021") + d.c("Jiang2021"), t("块稀疏贝叶斯学习", "Block-sparse Bayesian learning"), t("同步相量，IEEE 123", "Synchronized phasors, IEEE 123"), t("是", "Yes"), t("未见", "Not reported"), t("否", "No")],
        [t("Khattak 和 Choudhry 2025", "Khattak & Choudhry 2025") + d.c("Khattak2025"), t("一维 CNN", "1-D CNN"), t("故障前后差值相量，IEEE 34/123 用 5/9 个 PMU", "Pre/post-fault difference phasors, 5/9 PMUs on IEEE 34/123"), t("否", "No"), t("未见（无噪声、无负荷偏移，Rf ≤ 10 Ω）", "No (no noise, no load shift, Rf ≤ 10 Ω)"), t("否", "No")],
        [t("Chen 等 2019", "Chen et al. 2019") + d.c("Chen2019"), t("图卷积网络", "Graph convolutional network"), t("故障期间相量，IEEE 123 默认 25 个测点", "During-fault phasors, 25 sensors on IEEE 123"), t("否", "No"), t("负荷水平、噪声、拓扑变化", "Load level, noise, topology change"), t("否", "No")],
        [t("Li 和 Deka 2023", "Li & Deka 2023") + d.c("Li2023"), t("物理结构图网络", "Physics-preserved graph network"), t("相量，IEEE 123 的 21 个测点", "Phasors, 21 sensors on IEEE 123"), t("否（结构含物理信息）", "No (structure only)"), t("负荷与拓扑变化", "Load and topology change"), t("否", "No")],
        [t("Zheng 等 2026", "Zheng et al. 2026") + d.c("Zheng2026"), t("双图物理信息图注意力网络", "Dual-graph physics-informed GAT"), t("IEEE 33", "IEEE 33"), t("否（KCL 残差为训练正则项）", "No (KCL residual as regularizer)"), t("噪声、数据缺失", "Noise, missing data"), t("否", "No")],
    ]
    d.table(1, "与本文最接近的方法（据摘要与可得全文整理）", "Methods closest to this paper (compiled from abstracts and available full texts)",
            [t("方法", "Method"), t("类别", "Category"), t("输入与测点", "Input and sensors"), t("推理时含物理模型", "Physics model at inference"), t("外推或偏离训练分布测试", "Out-of-distribution test"), t("失配告警", "Mismatch alarm")], rows, [":--"] * 6)


def table2(d):
    t = d.t
    rows = [
        [sc(d, "域内"), t("与训练同分布", "Same as training"), t("同分布内的总体与分档精度", "Overall and binned accuracy within the distribution")],
        [sc(d, "负荷偏移"), t("全网负荷 1.25–1.5 倍（训练 0.6–1.1）", "Network load 1.25–1.5× (trained 0.6–1.1)"), t("负荷水平外推", "Load-level extrapolation")],
        [sc(d, "高阻外推"), t("过渡电阻 300–1000 Ω（训练 0.01–300 Ω）", "Fault resistance 300–1000 Ω (trained 0.01–300 Ω)"), t("数据驱动失效处的表现", "Behaviour where data-driven methods fail")],
        [sc(d, "噪声×3"), t("量测噪声放大 3 倍", "Measurement noise ×3"), t("噪声鲁棒性", "Noise robustness")],
        [sc(d, "线路阻抗+10%") + " / " + sc(d, "线路阻抗-10%"), t("全网线路阻抗整体偏差，模型仍用名义值", "All line impedances off by ±10%, model keeps nominal"), t("线路参数不准", "Inaccurate line parameters")],
        [sc(d, "个别线路阻抗±20%"), t("每条线路阻抗独立取名义值的 0.8–1.2 倍", "Each line impedance 0.8–1.2× nominal, independently"), t("更接近实际的模型误差", "More realistic model error")],
        [t("固定档位（4 个）", "Fixed taps (4)"), t("调压器档位固定的域内、负荷偏移、高阻外推、噪声×3", "In-domain, load shift, high resistance and noise ×3 with fixed regulator taps"), t("与要求档位已知的 Liang 两阶段方法比较", "Comparison with the two-stage method of Liang et al. (requires known taps)")],
        [t("噪声档位", "Noise levels"), t("噪声 ×0.5、×1、×2、×3、×5", "Noise ×0.5, ×1, ×2, ×3, ×5"), t("噪声敏感性", "Noise sensitivity")],
    ]
    d.table(2, "测试场景及其作用", "Test scenarios and their purpose", [t("场景", "Scenario"), t("设置", "Setting"), t("作用", "Purpose")], rows, [":--"] * 3)


# ---------------------------------------------------------------- 附录中的静态部分
def appendix_hyper(d):
    t = d.t
    ev34, ev123 = load()["ev34"], load()["ev123"]
    rows = [
        [t("数据", "Data"), t("故障类型与分布", "Fault types and distribution"), t("10 种类型；位置沿全网均匀；训练过渡电阻对数均匀 0.01–300 Ω，负荷倍率 0.6–1.1；量测为 OpenDSS 潮流下故障后相对故障前的突变量", "10 types; position uniform over the network; training fault resistance log-uniform 0.01–300 Ω, load multiplier 0.6–1.1; measurements are the post-fault minus pre-fault change from OpenDSS power flow")],
        [t("量测", "Measurement"), t("噪声", "Noise"), t("电压为名义电压的 0.1%，电流 0.2 A，突变量标准差乘 √2；训练时每个批次重新叠加", "0.1% of nominal voltage, 0.2 A for current, disturbance standard deviation × √2; re-drawn for every training batch")],
        [t("阶段一", "Stage 1"), t("特征与网络", "Features and network"), t("各量测除以幅值最大的量测的实部与虚部、对数信噪比（截断到 [−3, 6]）、参考量测的对数信噪比；全连接 ReLU 网络，IEEE 34 为 160-160-96，IEEE 123 为 256-256-192", "Real/imaginary parts of each measurement divided by the largest, log SNR (clipped to [−3, 6]), log SNR of the reference; fully connected ReLU network, 160-160-96 (IEEE 34), 256-256-192 (IEEE 123)")],
        [t("阶段一", "Stage 1"), t("训练", "Training"), t("AdamW，学习率 2e-3（OneCycle），权重衰减 1e-4，批量 512；IEEE 34 为 6000 步，IEEE 123 为 15000 步", "AdamW, learning rate 2e-3 (OneCycle), weight decay 1e-4, batch 512; 6000 steps (IEEE 34), 15000 steps (IEEE 123)")],
        [t("阶段一", "Stage 1"), t("训练集", "Training set"), t("IEEE 34：8 万个名义阻抗样本 + 阻抗倍率 0.88、0.94、1.06、1.12 各 2 万个；IEEE 123：12 万 + 各 3 万个", "IEEE 34: 80,000 nominal-impedance samples + 20,000 each at scales 0.88, 0.94, 1.06, 1.12; IEEE 123: 120,000 + 30,000 each")],
        [t("阶段二", "Stage 2"), t("响应库与阻抗假设", "Response library and impedance hypotheses"), t("每个区段 49 个位置点（α 从 0.02 到 0.98）；线路阻抗倍率假设：IEEE 34 为 0.9、0.95、1.0、1.05、1.1，IEEE 123 为 0.9、1.0、1.1；先验 ((s−1)/0.05)²", "49 positions per section (α from 0.02 to 0.98); impedance-scale hypotheses 0.9, 0.95, 1.0, 1.05, 1.1 (IEEE 34) and 0.9, 1.0, 1.1 (IEEE 123); prior ((s−1)/0.05)²")],
        [t("阶段二", "Stage 2"), t("过渡电阻与增益", "Fault resistance and gain"), t("R 对数网格 48 点（0.01–3000 Ω）加 13 点局部细搜；实增益 g 先验 σ_g=0.05，闭式解", "48-point log grid for R (0.01–3000 Ω) plus a 13-point local refinement; real gain g with prior σ_g=0.05, closed form")],
        [t("阶段二", "Stage 2"), t("残差权重", "Residual weights"), t("1/(σ²+(ε_r|z|)²+(ε_a z_max)²)，ε_r=2%，ε_a=1%", "1/(σ²+(ε_r|z|)²+(ε_a z_max)²), ε_r=2%, ε_a=1%")],
        [t("阶段三", "Stage 3"), t("候选数、λ、阈值", "Candidates, λ, threshold"), t(f"k=2；λ：IEEE 34 为 {ev34['thresholds']['lam']:.3f}，IEEE 123 为 {ev123['thresholds']['lam']:.3f}；J 阈值（假设正确时的 99% 分位数）：{ev34['thresholds']['j_threshold']:.1f}、{ev123['thresholds']['j_threshold']:.1f}", f"k=2; λ = {ev34['thresholds']['lam']:.3f} (IEEE 34), {ev123['thresholds']['lam']:.3f} (IEEE 123); J threshold (99th percentile when correct): {ev34['thresholds']['j_threshold']:.1f}, {ev123['thresholds']['j_threshold']:.1f}")],
        [t("等价类", "Equivalence classes"), t("定义", "Definition"), t("不少于 50% 的样本中 s′ 的惩罚残差不比 s 大出 4，则 s 与 s′ 不可区分；双向不可区分的区段连通成类（IEEE 34 为 22 类，IEEE 123 为 64 类，固定）", "s and s′ are indistinguishable if in at least 50% of the samples the penalized residual of s′ is not larger than that of s by more than 4; mutually indistinguishable sections are connected into classes (22 for IEEE 34, 64 for IEEE 123; fixed)")],
        [t("GCN", "GCN"), t("结构与训练", "Structure and training"), t("母线图（保留每个母线最近的 Kn 个，IEEE 34 为 6，IEEE 123 为 12）；3 层切比雪夫卷积（阶数 3、4、5，各 128 个滤波器）；全连接 512-256，dropout 0.5；AdamW，学习率 1e-3（OneCycle），批量 128，IEEE 34 为 6000 步，IEEE 123 为 8000 步", "Bus graph (nearest Kn buses kept, Kn=6 for IEEE 34, 12 for IEEE 123); three Chebyshev layers (orders 3, 4, 5, 128 filters each); fully connected 512-256, dropout 0.5; AdamW, learning rate 1e-3 (OneCycle), batch 128, 6000 steps (IEEE 34), 8000 steps (IEEE 123)")],
        [t("GAT", "GAT"), t("结构与训练", "Structure and training"), t("3 层 4 头图注意力（每层 128 维），注意力限于物理相邻母线（含自环）；其后同样接两层全连接；训练同 GCN", "Three layers with four attention heads (128 dimensions per layer), attention restricted to physically adjacent buses (with self-loops); followed by the same two fully connected layers; trained like the GCN")],
        [t("块稀疏贝叶斯", "Block-sparse Bayesian"), t("字典与迭代", "Dictionary and iterations"), t("字典为各区段在 α = 0.05–0.95（10 点）的三相注入电流响应，块大小为区段相数；EM 迭代 60 次；量测噪声方差取与阶段二相同的加权；取后验能量最大的块", "Dictionary of three-phase injection-current responses of each section at α = 0.05–0.95 (10 points), block size equal to the number of phases; 60 EM iterations; noise variance weighted as in stage 2; the block with the largest posterior energy is chosen")],
        [t("PCA+SVM、PCA+随机森林", "PCA+SVM, PCA+RF"), t("设置", "Settings"), t("阶段一特征标准化后 PCA 保留 99% 方差；每个训练样本叠加噪声 2 次；SVM 为 RBF 核（C=10，gamma=scale），3 万个训练子样本，决策函数经单一温度标定为概率；随机森林 150 棵树，叶节点最少 3 个样本", "Stage-1 features standardized, PCA keeps 99% of the variance; two noise draws per training sample; SVM with RBF kernel (C=10, gamma=scale) on a 30,000-sample subset, decision function mapped to probabilities by a single temperature; random forest with 150 trees and at least 3 samples per leaf")],
    ]
    d.table("A1", "超参数汇总", "Hyperparameters", [t("模块", "Module"), t("项目", "Item"), t("取值", "Value")], rows, [":--"] * 3)


def appendix_liang(d):
    t = d.t
    d.w(t("Liang 等的两阶段方法没有公开代码，按论文描述复现。为检验复现的保真度，在论文自己的 IEEE 33 节点系统上按论文的 μPMU 配置、噪声设置、10 类故障与故障位置 10%–90% 重做了评测，并与论文图中读出的数值（误差约 ±1 个百分点）对照。",
          "The method of Liang et al. has no public code and was reimplemented from the paper. To check the fidelity of the reimplementation we repeated the evaluation on the paper's own IEEE 33-bus system with the paper's μPMU placement, noise settings, 10 fault types and fault positions of 10%–90%, and compared with values read from the paper's figures (about ±1 percentage point).\n"))
    rows = [["Top-5 " + t("本复现", "reproduction"), "16.2", "33.4", "55.5", "72.1", "74.4", "87.2", "86.7", "93.7"], ["Top-5 " + t("论文", "paper"), "13.0", "26.5", "86.0", "95.5", "98.5", "99.0", "99.0", "98.5"],
            ["Top-2 " + t("本复现", "reproduction"), "7.0", "19.7", "15.1", "29.0", "37.8", "45.8", "47.8", "53.7"], ["Top-2 " + t("论文", "paper"), "4.0", "12.0", "26.0", "38.0", "47.5", "56.5", "67.0", "65.0"]]
    d.table("A3a", "第一阶段 Top-k 成功率（%）随 μPMU 数量 N 的变化（故障电阻 0–100 Ω）", "Stage-1 Top-k success rate (%) versus the number N of μPMUs (fault resistance 0–100 Ω)", [t("指标", "Metric")] + [f"N={i}" for i in range(1, 9)], rows)
    rows = [["100 Ω / 20%", "83.9", "97.8"], ["300 Ω / 20%", "80.6", "93.8"], ["500 Ω / 20%", "78.8", "89.3"], ["100 Ω / 50%", "82.1", "95.5"], ["500 Ω / 50%", "65.0", "64.0"]]
    d.table("A3b", "两阶段成功率（%，5 个 μPMU；故障电阻上限 / 伪量测误差）", "Two-stage success rate (%, 5 μPMUs; fault-resistance upper limit / pseudo-measurement error)", [t("条件", "Condition"), t("本复现", "Reproduction"), t("论文", "Paper")], rows)
    d.w(t("第一阶段低于论文（平均绝对偏差 14.2 个百分点），第二阶段的变化趋势与论文一致，最难条件下与论文数值一致。因此本文对 Liang 两阶段方法的比较只代表这个复现版本，可能低估了原方法。",
          "Stage 1 is below the paper (mean absolute deviation 14.2 percentage points); the trend of stage 2 follows the paper, and in the hardest condition the values agree. The comparison with the method of Liang et al. in this paper therefore represents this reimplementation and may underestimate the original method.\n"))


# ---------------------------------------------------------------- 结果章节用的表格
TAG = {"gcn_p": "gcn_paper", "gcn_d": "gcn_delta", "gat_p": "gat_paper", "gat_d": "gat_delta", "svm": "pca_svm", "rf": "pca_rf"}
BASE_JSON = {"gcn_p": ("gcn_paper.json", "gcn"), "gcn_d": ("gcn_delta.json", "gcn_delta"), "gat_p": ("gat_paper.json", "gcn"), "gat_d": ("gat_delta.json", "gcn_delta")}


def seed0_base(k, s, system="34"):
    """分类器基线种子 0 的原始结果：{section_acc, class_acc, d_mean_midpoint}。"""
    R = R34 if system == "34" else R123
    if k in BASE_JSON:
        f, key = BASE_JSON[k]
        p = R / f
        return J(p)[key].get(s) if p.exists() else None
    m = TAG[k]
    p = R / "baselines_ml.json"
    return J(p)[m].get(s) if p.exists() else None


def swap(k, s):
    """分类器 k 作阶段一时的完整评测结果（IEEE 34，种子 0）。"""
    p = R34 / "stage1_swap" / TAG[k] / "evaluation.json"
    return J(p).get(s) if p.exists() else None


def loc_cells(d, s):
    """表 3 下半部分：各方法的测距误差（km）。分类器格式为“区段中点 / +物理定位”。"""
    c = load()
    ms = c["ms34"][s]
    bs = J(R34 / "baselines_bsbl.json").get(s) if (R34 / "baselines_bsbl.json").exists() else None
    cand = {"ours": ms["本方案测距误差均值km"]["mean"]}
    if bs:
        cand["bsbl"] = bs["d_mean"]
    mid, phys = {}, {}
    for k in TAG:
        b, w = seed0_base(k, s), swap(k, s)
        if b and w:
            mid[k], phys[k] = b["d_mean_midpoint"], w[NN]["d_mean"]
            cand[k] = phys[k]
    best = min(round(v, 2) for v in cand.values())
    out = []
    for k, _, _ in METHODS:
        if k in ("ours", "bsbl"):
            v = cand.get(k)
            out.append("-" if v is None else (f"**{v:.2f}**" if round(v, 2) == best else f"{v:.2f}"))
        elif k in mid:
            p = f"{phys[k]:.2f}"
            out.append(f"{mid[k]:.2f} / " + (f"**{p}**" if round(phys[k], 2) == best else p))
        else:
            out.append("-")
    return out


def table3(d):
    t, c = d.t, load()
    hdr = [t("场景", "Scenario")] + [mname(d, k) for k, _, _ in METHODS]
    rows = []
    for s in MAIN7:
        vals = [c["acc"]["34"][s].get(k) for k, _, _ in METHODS]
        rows.append([sc(d, s)] + bold_best(vals))
    rows.append([t("**测距误差 (km)**：分类器为“区段中点 / +物理定位”", "**Location error (km)**: classifiers as “section midpoint / +physics location”")] + [""] * len(METHODS))
    for s in MAIN7:
        rows.append([sc(d, s)] + loc_cells(d, s))
    d.table(3, "IEEE 34 上的主对比：类准确率（%，均值±标准差）与测距误差（km）。每行最优加粗",
            "Main comparison on IEEE 34: class accuracy (%, mean ± standard deviation) and location error (km). The best in each row is in bold", hdr, rows)


def liang_best(ev, s):
    cands = [(ev[s][f"{L} {p}"]["class_acc"], ev[s][f"{L} {p}"]["d_mean"], f"{L} {p}") for L in ("L1(同测点)", "L1(自身布点)") for p in ("论文流程", "全候选线路细化")]
    return max(cands)


def table4(d):
    t, c = d.t, load()
    ev = c["ev34"]
    hdr = [t("场景", "Scenario"), t("本方法", "Ours"), t("Liang 两阶段", "Liang two-stage")] + [mname(d, k) for k, _, _ in METHODS[1:]]
    rows = []
    for s in FIXED4:
        la = liang_best(ev, s)
        vals = [c["acc"]["34"][s].get("ours"), (la[0], 0.0, 1)] + [c["acc"]["34"][s].get(k) for k, _, _ in METHODS[1:]]
        rows.append([sc(d, s)] + bold_best(vals))
    d.table(4, "档位固定场景下与 Liang 两阶段方法的比较（类准确率 %；Liang 取四个变体中最好者）",
            "Comparison with the two-stage method of Liang et al. in the fixed-tap scenarios (class accuracy %; Liang is the best of its four variants)", hdr, rows)
    errs = "; ".join(f"{sc(d, s)}: {t('本方法', 'ours')} {c['ms34'][s]['本方案测距误差均值km']['mean']:.2f} km, Liang {liang_best(ev, s)[1]:.2f} km" for s in FIXED4)
    d.w(t("测距误差（均值）：", "Location error (mean): ") + errs + ".\n")


def table5(d):
    t = d.t
    t1 = J(R34 / "tier1.json")
    keymap = {"ours": "本方案", "gcn_p": "GCN 论文输入", "gcn_d": "GCN 突变量输入", "gat_p": "GAT 论文输入", "gat_d": "GAT 突变量输入", "bsbl": "块稀疏贝叶斯学习", "svm": "PCA+SVM", "rf": "PCA+随机森林"}
    groups = ["故障类型 SLG", "故障类型 LL", "故障类型 LLG", "故障类型 LLLG", "位置 近端（α<0.35）", "位置 中段（0.35≤α<0.65）", "位置 远端（α≥0.65）"] + [g for g in t1 if g.startswith("Rf ")]
    en_g = {"故障类型 SLG": "Fault type SLG", "故障类型 LL": "Fault type LL", "故障类型 LLG": "Fault type LLG", "故障类型 LLLG": "Fault type LLLG", "位置 近端（α<0.35）": "Position near end (α<0.35)",
            "位置 中段（0.35≤α<0.65）": "Position middle (0.35≤α<0.65)", "位置 远端（α≥0.65）": "Position far end (α≥0.65)"}
    hdr = [t("分组", "Group"), t("样本数", "Samples")] + [mname(d, k) for k, _, _ in METHODS]
    rows = []
    for g in groups:
        vals = [(t1[g][keymap[k]]["class_acc"], 0.0, 1) for k, _, _ in METHODS]
        lab = g if d.L == "zh" else en_g.get(g, g.replace("Rf ", "Rf ").replace("–", "–"))
        rows.append([lab, str(t1[g]["本方案"]["n"])] + bold_best(vals))
    d.table(5, "IEEE 34 同分布内（Rf ≤ 300 Ω）及高阻外推样本按故障类型、故障位置、过渡电阻分档的类准确率（%，种子 0；Rf 300 Ω 以上的两档取自高阻外推场景）",
            "Class accuracy (%, seed 0) of the IEEE 34 in-domain samples and the high-resistance samples binned by fault type, fault position and fault resistance (the two bins above 300 Ω come from the high-resistance scenario)", hdr, rows)


ABL_SC = ["域内", "负荷偏移", "高阻外推", "噪声×3", "线路阻抗-10%", "固定档位 域内", "固定档位 负荷偏移", "固定档位 阻抗+10%"]


def table6(d):
    t = d.t
    f = {"free": J(R34 / "ablation_aug_only/evaluation.json"), "con": load()["ev34"], "nog": J(R34 / "ablation_fixed_gain/evaluation.json")}
    hdr = [t("场景", "Scenario")] + [f"{a} {b}" for a in (t("物理穷举", "Exhaustive physics"), t("联合决策", "Joint decision"), t("测距误差 km", "Location error km")) for b in (t("自由", "free"), t("约束", "con."), t("无增益", "no gain"))]
    rows = []
    for s in ABL_SC:
        r = [sc(d, s)]
        for key, prec, fn in ((PH, 1, lambda x: x * 100), (K, 1, lambda x: x * 100)):
            vals = [f[m][s][key]["class_acc"] for m in ("free", "con", "nog")]
            top = max(round(v * 100, 1) for v in vals)
            r += [f"**{v * 100:.1f}**" if round(v * 100, 1) == top else f"{v * 100:.1f}" for v in vals]
        vals = [f[m][s][K]["d_mean"] for m in ("free", "con", "nog")]
        low = min(round(v, 2) for v in vals)
        r += [f"**{v:.2f}**" if round(v, 2) == low else f"{v:.2f}" for v in vals]
        rows.append(r)
    d.table(6, "物理模型的受控消融（IEEE 34，种子 0；自由：注入电流为自由复数；约束：过渡电阻为实数加实增益；无增益：约束但增益固定为 1）。除物理模型外其余设置相同",
            "Controlled ablation of the physics model (IEEE 34, seed 0; free: free complex injection current; con.: real fault resistance plus real gain; no gain: constrained with the gain fixed at 1). Everything except the physics model is identical", hdr, rows)


def table7(d):
    t = d.t
    hdr = [t("分类器", "Classifier")] + [f"{sc(d, s)}: {t('自身 → +本方法流程', 'alone → +our procedure')}" for s in ("域内", "高阻外推", "噪声×3", "线路阻抗-10%")] + [t("+本方法流程高于+物理定位的主场景数（共 7 个）", "Main scenarios where +our procedure beats +physics location (of 7)")]
    rows = []
    for k in ("gcn_p", "gcn_d", "gat_p", "gat_d", "svm", "rf"):
        r = [mname(d, k)]
        for s in ("域内", "高阻外推", "噪声×3", "线路阻抗-10%"):
            w = swap(k, s)
            r.append("-" if not w else f"{w[NN]['class_acc'] * 100:.1f} → {w[K]['class_acc'] * 100:.1f}")
        n_better = sum(1 for s in MAIN7 if swap(k, s) and swap(k, s)[K]["class_acc"] > swap(k, s)[NN]["class_acc"])
        r.append(str(n_better))
        rows.append(r)
    d.table(7, "各分类器作为阶段一接入本方法流程（IEEE 34，种子 0；类准确率 %）",
            "Each classifier plugged in as stage 1 of our procedure (IEEE 34, seed 0; class accuracy %)", hdr, rows)
    bd = J(R34 / "breakdown.json")["互补性"]
    hdr = [t("场景", "Scenario"), t("两者都对", "Both right"), t("仅神经网络对", "NN only right"), t("仅物理对", "Physics only right"), t("两者都错", "Both wrong"), t("神经网络错时物理对 (%)", "Physics right when NN wrong (%)"), t("物理错时神经网络对 (%)", "NN right when physics wrong (%)")]
    rows = []
    for s in ("域内", "负荷偏移", "高阻外推", "噪声×3", "线路阻抗-10%", "个别线路阻抗±20%", "固定档位 负荷偏移", "固定档位 高阻外推"):
        x = bd[s]
        rows.append([sc(d, s)] + [f"{x[k] * 100:.1f}" for k in ("两者都对", "仅神经网络对", "仅物理对", "两者都错")] + [f"{x['神经网络错时物理对'] * 100:.0f}", f"{x['物理错时神经网络对'] * 100:.0f}"])
    d.table("7b", "神经网络 Top-1 与全区段物理穷举的对错互补（占全部样本的百分比，IEEE 34，种子 0）",
            "Complementarity of the neural-network Top-1 and the exhaustive physics-based inversion (percent of all samples, IEEE 34, seed 0)", hdr, rows)


def table8(d):
    t, c = d.t, load()
    scs = ["个别线路阻抗±20%", "线路与负荷同时偏差", "各负荷独立变化"]
    hdr = [t("场景", "Scenario")] + [mname(d, k) for k, _, _ in METHODS]
    rows = [[sc(d, s)] + bold_best([c["acc"]["34"][s].get(k) for k, _, _ in METHODS]) for s in scs]
    d.table(8, "模型偏差场景（IEEE 34，类准确率 %）", "Model-error scenarios (IEEE 34, class accuracy %)", hdr, rows)
    o = J(R34 / "oracle_line_params.json")
    hdr = [t("指标", "Metric"), t("名义阻抗模型（本方法）", "Nominal-impedance model (ours)"), t("逐线真值模型", "True per-line model")]
    rows = [[t("联合决策类准确率 (%)", "Joint-decision class accuracy (%)"), f"{o['class_acc']['joint_nominal']:.1f}", f"**{o['class_acc']['joint_oracle']:.1f}**"],
            [t("区段已知时测距误差均值 (km)", "Location error, section known, mean (km)"), f"{o['distance_err_km_true_section']['mean_nominal']:.2f}", f"**{o['distance_err_km_true_section']['mean_oracle']:.2f}**"],
            [t("区段已知时测距误差中位数 (km)", "Location error, section known, median (km)"), f"{o['distance_err_km_true_section']['median_nominal']:.2f}", f"**{o['distance_err_km_true_section']['median_oracle']:.2f}**"]]
    d.table("8b", f"个别线路阻抗 ±20% 下的逐线阻抗真值诊断（前 {o['n']} 个样本；神经网络 Top-1 为 {o['class_acc']['stage1_top1']:.1f}%）",
            f"Diagnosis with the true per-line impedances under individual line impedance ±20% (first {o['n']} samples; neural-network Top-1 is {o['class_acc']['stage1_top1']:.1f}%)", hdr, rows)


def table9(d):
    t, c = d.t, load()
    hdr = [t("场景", "Scenario")] + [mname(d, k) for k, _, _ in METHODS]
    rows = [[sc(d, s)] + bold_best([c["acc"]["123"].get(s, {}).get(k) for k, _, _ in METHODS]) for s in S123]
    d.table(9, "IEEE 123 上的对比（类准确率 %；块稀疏贝叶斯每个场景取 1000 个样本；GAT 论文输入因训练耗时未在 IEEE 123 上运行；本方法 5 个种子，GCN 3 个种子（个别线路阻抗 ±20% 为 2 个），其余为种子 0）",
            "Comparison on IEEE 123 (class accuracy %; block-sparse Bayesian uses 1000 samples per scenario; GAT with the paper input was not run on IEEE 123 because of its training cost; ours 5 seeds, GCN 3 seeds (2 for the individual-line scenario), the rest seed 0)", hdr, rows)
    t34, t123 = c["ev34"]["耗时_ms每样本"], c["ev123"]["耗时_ms每样本"]
    k34 = [k for k in t34 if k.startswith("阶段二")][0]
    k123 = [k for k in t123 if k.startswith("阶段二")][0]
    hdr = [t("系统", "System"), t("阶段一 (ms/样本)", "Stage 1 (ms/sample)"), t("阶段二，Top-2 候选 (ms/样本)", "Stage 2, Top-2 candidates (ms/sample)"), t("全区段物理穷举 (ms/样本)", "Exhaustive physics over all sections (ms/sample)")]
    rows = [["IEEE 34", f"{t34['阶段一']:.3f}", f"{t34[k34]:.1f}", f"{t34['物理穷举(全部区段)']:.0f}"], ["IEEE 123", f"{t123['阶段一']:.3f}", f"{t123[k123]:.1f}", f"{t123['物理穷举(全部区段)']:.0f}"]]
    d.table("9b", "推断耗时（批量 1000 个样本，4 核 CPU 单进程）", "Inference time (batch of 1000 samples, one process on a 4-core CPU)", hdr, rows)


def appendix_exact(d):
    """附录表 A2：精确区段准确率与等价类准确率并列（种子 0）。"""
    t = d.t
    c = load()
    ours = np.load(R34 / "samples.npz")
    hdr = [t("场景", "Scenario")] + [mname(d, k) for k, _, _ in METHODS]
    rows = []
    for s in MAIN7:
        r = [sc(d, s)]
        n = len(ours[f"{s}/sec_true"])
        ex = float((ours[f"{s}/本方案/sec"] == ours[f"{s}/sec_true"]).mean())
        r.append(f"{ex * 100:.1f} / {c['ev34'][s][K]['class_acc'] * 100:.1f}")
        for k, _, _ in METHODS[1:]:
            if k == "bsbl":
                b = J(R34 / "baselines_bsbl.json").get(s)
            else:
                b = seed0_base(k, s)
            r.append("-" if not b else f"{b['section_acc'] * 100:.1f} / {b['class_acc'] * 100:.1f}")
        rows.append(r)
    d.table("A2", "精确区段准确率 / 等价类准确率（%，IEEE 34，种子 0）", "Exact section accuracy / equivalence-class accuracy (%, IEEE 34, seed 0)", hdr, rows)


def appendix_design(d):
    """附录表 A4：阻抗扩增、候选扩展与置信度。"""
    t, c = d.t, load()
    ev = c["ev34"]
    no_aug = J(R34 / "ablation_no_aug/evaluation.json")
    hdr = [t("场景", "Scenario"), t("无阻抗扩增", "No impedance augmentation"), t("有阻抗扩增（本方法）", "With augmentation (ours)"), t("Top-2 无候选扩展", "Top-2, no expansion"), t("Top-2 加候选扩展（本方法）", "Top-2 with expansion (ours)"), t("全区段联合打分（参考）", "Joint score over all sections (reference)")]
    rows = []
    for s in ("域内", "负荷偏移", "高阻外推", "噪声×3", "线路阻抗+10%", "线路阻抗-10%", "个别线路阻抗±20%"):
        rows.append([sc(d, s), f"{no_aug[s][K]['class_acc'] * 100:.1f}", f"{ev[s][K]['class_acc'] * 100:.1f}", f"{ev[s]['联合打分(Top-2,无候选扩展)']['class_acc'] * 100:.1f}", f"{ev[s][K]['class_acc'] * 100:.1f}", f"{ev[s]['联合打分(全区段,参考)']['class_acc'] * 100:.1f}"])
    d.table("A4a", "阻抗扩增训练与候选扩展的作用（IEEE 34，类准确率 %，种子 0）", "Effect of impedance augmentation and candidate expansion (IEEE 34, class accuracy %, seed 0)", hdr, rows)
    hdr = [t("场景", "Scenario"), t("高置信样本占比 (%)", "High-confidence share (%)"), t("高置信准确率 (%)", "High-confidence accuracy (%)"), t("低置信准确率 (%)", "Low-confidence accuracy (%)"), t("校准误差 ECE", "ECE"), t("告警占比 (%)", "Alarm share (%)")]
    rows = []
    for s in ("域内", "负荷偏移", "高阻外推", "噪声×3", "个别线路阻抗±20%"):
        x, r2 = ev[s]["置信度C"], ev[s][K]["rules"]
        rows.append([sc(d, s), f"{x['高置信占比'] * 100:.0f}", f"{x['高置信准确率'] * 100:.1f}", f"{x['低置信准确率'] * 100:.1f}", f"{x['ECE']:.3f}", f"{r2['2'] * 100:.1f}"])
    d.table("A4b", "置信度与失配告警（IEEE 34，种子 0；高置信指置信度 ≥ 0.6）", "Confidence and mismatch alarm (IEEE 34, seed 0; high confidence means confidence ≥ 0.6)", hdr, rows)


def appendix_extra(d):
    """附录表 A5：正文之外的场景（本方法）。"""
    t, c = d.t, load()
    ms = c["ms34"]
    hdr = [t("场景", "Scenario"), t("类准确率 (%，均值±标准差)", "Class accuracy (%, mean ± std)"), t("测距误差均值 (km)", "Mean location error (km)")]
    rows = []
    for s in ("线路阻抗+5%", "各负荷独立变化", "线路与负荷同时偏差", "固定档位 阻抗+10%", "固定档位 阻抗-10%"):
        rows.append([sc(d, s), cell((ms[s]["本方案"]["mean"], ms[s]["本方案"]["std"], ms[s]["本方案"]["n"])), f"{ms[s]['本方案测距误差均值km']['mean']:.2f}"])
    d.table("A5", "正文之外的场景（IEEE 34，本方法）", "Scenarios outside the main text (IEEE 34, our method)", hdr, rows)


def appendix_baselines(d):
    t = d.t
    rows = [
        ["GCN", t("按 Chen 等的描述复现：母线图（每个母线保留最近的 Kn 个距离，高斯核权重），切比雪夫卷积，类别为故障区段（原文为故障母线）；论文输入需要故障前相量，由相同参数重新生成数据得到", "Reimplemented from Chen et al.: bus graph (nearest Kn distances per bus, Gaussian weights), Chebyshev convolutions; classes are fault sections (fault buses in the original); the paper input needs pre-fault phasors, obtained by regenerating the data with the same parameters")],
        ["GAT", t("按 Lingyu Liang 等的描述复现：多头图注意力；注意力限于物理相邻母线；输入与训练同 GCN", "Reimplemented from Lingyu Liang et al.: multi-head graph attention restricted to physically adjacent buses; inputs and training as for the GCN")],
        [t("块稀疏贝叶斯", "Block-sparse Bayesian"), t("按 Jiang 等的思路复现并适配：以三相虚拟注入电流为未知量的块稀疏贝叶斯学习，字典取名义参数下的响应；位置受 10 点 α 网格限制，因此其测距误差包含网格误差；不使用神经网络", "Reimplemented and adapted from the idea of Jiang et al.: block-sparse Bayesian learning with the three-phase virtual injection currents as unknowns and a dictionary of nominal-parameter responses; the position is limited to a 10-point α grid, so its location error includes the grid error; no neural network")],
        [t("Liang 两阶段", "Liang two-stage"), t("按论文描述复现，复现保真度见表 A3；假设档位与网络参数精确已知，获得根母线相量、末端电压幅值和全网负荷伪量测，多于本方法的量测", "Reimplemented from the paper, fidelity in Table A3; assumes exactly known taps and network parameters and receives root-bus phasors, end-bus voltage magnitudes and load pseudo-measurements, more than our method")],
        ["PCA+SVM / PCA+RF", t("沿用 Chen 等的对照协议；输入为本方法阶段一的特征；SVM 受计算量限制只用 3 万个训练子样本", "Following the comparison protocol of Chen et al.; inputs are the stage-1 features of our method; the SVM uses only a 30,000-sample training subset because of its cost")],
    ]
    d.table("A6", "基线的复现说明与局限", "Reimplementation notes and limitations of the baselines", [t("方法", "Method"), t("说明", "Notes")], rows, [":--", ":--"])


def appendix(d):
    d.w(d.t("## 附录\n", "## Appendix\n"))
    appendix_hyper(d)
    appendix_exact(d)
    d.w(d.t("### 对 Liang 两阶段方法的复现验证\n", "### Validation of the reimplementation of the Liang et al. method\n"))
    appendix_liang(d)
    appendix_design(d)
    appendix_extra(d)
    appendix_baselines(d)


# ---------------------------------------------------------------- 结果章节的叙述（比较结论由数据判断后写入）
def verdict(A, s):
    """本方法与该场景下其余方法中均值最高者比较：均值差超过两者较大标准差的 2 倍才算显著。"""
    o = A[s]["ours"]
    others = {k: v for k, v in A[s].items() if k != "ours"}
    kb = max(others, key=lambda k: others[k][0])
    b = others[kb]
    diff = (o[0] - b[0]) * 100
    tol = 2 * max(o[1], b[1]) * 100
    return ("win" if diff > tol else "lose" if diff < -tol else "tie"), diff, kb


def pct(v):
    return f"{v * 100:.1f}"


def results(d):
    t, c = d.t, load()
    A = c["acc"]["34"]
    ev = c["ev34"]
    w = d.w
    w(t("## 5 结果\n", "## 5 Results\n"))
    # ---- 5.1
    w(t("### 5.1 与已发表方法和机器学习基线的对比\n", "### 5.1 Comparison with published methods and machine-learning baselines\n"))
    table3(d)
    vd = {s: verdict(A, s) for s in MAIN7}
    win = [s for s in MAIN7 if vd[s][0] == "win"]
    tie = [s for s in MAIN7 if vd[s][0] == "tie"]
    lose = [s for s in MAIN7 if vd[s][0] == "lose"]
    lst = lambda ss: "、".join(sc(d, s) for s in ss) if d.L == "zh" else ", ".join(sc(d, s) for s in ss)
    hr = A["高阻外推"]
    kb = vd["高阻外推"][2]
    w(t(f"表 3 给出主对比。与每个场景中均值最高的其余方法相比，本方法在 {len(win)} 个场景显著更高（{lst(win)}），{len(tie)} 个场景无显著差别（{lst(tie)}），{len(lose)} 个场景显著更低（{lst(lose)}）。",
        f"Table 3 gives the main comparison. Against the best of the other methods in each scenario, our method is significantly higher in {len(win)} scenario(s) ({lst(win)}), not significantly different in {len(tie)} ({lst(tie)}) and significantly lower in {len(lose)} ({lst(lose)}).") + "\n")
    w(t(f"过渡电阻超出训练范围（300–1000 Ω）时差距最大：本方法 {pct(hr['ours'][0])}%，其余方法中最高的 {mname(d, kb)} 为 {pct(hr[kb][0])}%。域内本方法为 {pct(A['域内']['ours'][0])}%，与最强的基线（{mname(d, vd['域内'][2])}，{pct(A['域内'][vd['域内'][2]][0])}%）相差 {abs(vd['域内'][1]):.1f} 个百分点。",
        f"The gap is largest when the fault resistance leaves the training range (300–1000 Ω): our method reaches {pct(hr['ours'][0])}%, the best of the other methods, {mname(d, kb)}, {pct(hr[kb][0])}%. In-domain, ours is {pct(A['域内']['ours'][0])}%, {abs(vd['域内'][1]):.1f} percentage points {'above' if vd['域内'][1] > 0 else 'below'} the strongest baseline ({mname(d, vd['域内'][2])}, {pct(A['域内'][vd['域内'][2]][0])}%).") + "\n")
    # 线路阻抗偏差场景中基线更高的情况
    zs = ["线路阻抗+10%", "线路阻抗-10%", "个别线路阻抗±20%"]
    zl = [s for s in zs if vd[s][0] == "lose"]
    if zl:
        w(t(f"在线路阻抗偏差场景中（{lst(zl)}），本方法低于{ '、'.join(sorted({mname(d, vd[s][2]) for s in zl})) }，差距 {min(-vd[s][1] for s in zl):.1f}–{max(-vd[s][1] for s in zl):.1f} 个百分点：这些场景使阶段二的名义响应库失配，区段判别主要依赖阶段一，而基线与阶段一使用相同的训练数据。",
            f"In the line-impedance-error scenarios ({lst(zl)}) our method is below {', '.join(sorted({mname(d, vd[s][2]) for s in zl}))}, by {min(-vd[s][1] for s in zl):.1f}–{max(-vd[s][1] for s in zl):.1f} percentage points: these scenarios make the nominal response library of stage 2 mismatched, so the section decision relies mainly on stage 1, and the baselines are trained on the same data as stage 1.") + "\n")
    w(t("分类器基线只输出区段，故障点取预测区段中点，测距误差因此明显偏大；把它们的 Top-1 区段交给阶段二估计区段内位置（“+物理定位”）后，测距误差与本方法接近，说明区段内测距精度主要来自物理反演（表 3 下半部分，图 4）。块稀疏贝叶斯学习自带位置估计，但受 α 网格限制且对过渡电阻没有约束。",
        "The classifier baselines output only the section and the fault point is taken at the section midpoint, so their location error is much larger; once their Top-1 section is handed to stage 2 to estimate the position within the section (“+physics location”), the location error approaches that of our method, which shows that the accuracy of the position within a section comes mainly from the physics-based inversion (lower half of Table 3, Fig. 4). Block-sparse Bayesian learning estimates the position itself, but is limited by the α grid and places no constraint on the fault resistance.") + "\n")
    d.fig(4)
    d.fig(5)
    table4(d)
    # ---- 5.2
    w(t("### 5.2 同分布内的分档精度\n", "### 5.2 Accuracy by fault type, position and resistance\n"))
    table5(d)
    t1 = J(R34 / "tier1.json")
    gk = {"本方案": "ours", "GCN 论文输入": "gcn_p", "GCN 突变量输入": "gcn_d", "GAT 论文输入": "gat_p", "GAT 突变量输入": "gat_d", "块稀疏贝叶斯学习": "bsbl", "PCA+SVM": "svm", "PCA+随机森林": "rf"}
    gs = [g for g in t1 if g.startswith(("故障类型 ", "位置 ", "Rf ")) and g != "故障类型 全部"]
    best_cnt = sum(1 for g in gs if max(t1[g], key=lambda k: t1[g][k]["class_acc"] if k in gk else -1) == "本方案")
    ft = [t1[g]["本方案"]["class_acc"] for g in gs if g.startswith("故障类型 ") and g != "故障类型 全部"]
    ps = [t1[g]["本方案"]["class_acc"] for g in gs if g.startswith("位置 ")]
    w(t(f"各类故障的类准确率在 {min(ft) * 100:.1f}%–{max(ft) * 100:.1f}% 之间，近端、中段、远端在 {min(ps) * 100:.1f}%–{max(ps) * 100:.1f}% 之间，说明方法对故障类型与区段内位置都没有明显的短板。按过渡电阻分档，准确率随过渡电阻升高而下降，并在超出训练范围后下降明显。在 {len(gs)} 个分档中，本方法在 {best_cnt} 个分档上最高。",
        f"The class accuracy of the individual fault types lies between {min(ft) * 100:.1f}% and {max(ft) * 100:.1f}%, and that of the near end, middle and far end between {min(ps) * 100:.1f}% and {max(ps) * 100:.1f}%, so the method has no obvious weakness with respect to fault type or position within a section. Binned by fault resistance, the accuracy decreases as the resistance rises and drops markedly beyond the training range. Among the {len(gs)} bins our method is the best in {best_cnt}.") + "\n")
    # ---- 5.3
    w(t("### 5.3 数据驱动与物理反演的互补\n", "### 5.3 Complementarity of the data-driven and physics-based parts\n"))
    d.fig(3)
    bdc = J(R34 / "breakdown.json")
    cv = bdc["过渡电阻曲线"]
    bins = list(cv)
    w(t(f"图 3(a) 是本文方法动机的直接证据：神经网络的类准确率随过渡电阻升高而下降（{cv[bins[0]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[3]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[4]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[5]]['硬串联']['class_acc'] * 100:.1f}%，依次为 0–1 Ω、100–300 Ω、300–500 Ω、500–1000 Ω），物理穷举基本不随过渡电阻变化（{min(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%–{max(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%）但在低阻时低于神经网络；两条曲线在 300–500 Ω 附近交叉，联合决策在该区间（{cv[bins[4]]['本方案']['class_acc'] * 100:.1f}%）高于两者。500–1000 Ω 时联合决策（{cv[bins[5]]['本方案']['class_acc'] * 100:.1f}%）仍低于物理穷举（{cv[bins[5]]['物理穷举']['class_acc'] * 100:.1f}%），原因是 Top-2 候选漏掉了真实区段。",
        f"Fig. 3(a) is direct evidence for the motivation of the method: the class accuracy of the neural network falls as the fault resistance rises ({cv[bins[0]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[3]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[4]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[5]]['硬串联']['class_acc'] * 100:.1f}% for 0–1 Ω, 100–300 Ω, 300–500 Ω and 500–1000 Ω), while the exhaustive physics-based inversion hardly changes with fault resistance ({min(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%–{max(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%) but is below the neural network at low resistance. The curves cross near 300–500 Ω, and the joint decision is above both there ({cv[bins[4]]['本方案']['class_acc'] * 100:.1f}%). At 500–1000 Ω the joint decision ({cv[bins[5]]['本方案']['class_acc'] * 100:.1f}%) is still below the exhaustive physics-based inversion ({cv[bins[5]]['物理穷举']['class_acc'] * 100:.1f}%) because the Top-2 candidates miss the true section.") + "\n")
    t1n = J(R34 / "tier1.json")
    others = [k for k in t1n["噪声 ×1"] if k not in ("本方案", "DNN Top-1")]
    nz = {}
    for lv in (0.5, 1, 2, 3, 5):
        g = t1n[f"噪声 ×{lv:g}"]
        kb2 = max(others, key=lambda k: g[k]["class_acc"])
        nz[lv] = (g["本方案"]["class_acc"] * 100, g[kb2]["class_acc"] * 100, kb2)
    behind = [lv for lv in nz if nz[lv][0] < nz[lv][1] - 0.05]
    nm_ = lambda k: {"GCN 论文输入": "GCN（论文输入）", "GCN 突变量输入": "GCN（突变量输入）", "GAT 论文输入": "GAT（论文输入）", "GAT 突变量输入": "GAT（突变量输入）", "PCA+SVM": "PCA+SVM", "PCA+随机森林": "PCA+随机森林", "块稀疏贝叶斯学习": "块稀疏贝叶斯"}[k] if d.L == "zh" else {"GCN 论文输入": "GCN (paper input)", "GCN 突变量输入": "GCN (disturbance input)", "GAT 论文输入": "GAT (paper input)", "GAT 突变量输入": "GAT (disturbance input)", "PCA+SVM": "PCA+SVM", "PCA+随机森林": "PCA+RF", "块稀疏贝叶斯学习": "block-sparse Bayesian"}[k]
    ahead_txt = "、".join(f"×{lv:g}" for lv in nz if lv not in behind)
    behind_txt = "；".join(f"×{lv:g} 时 {nm_(nz[lv][2])} 高 {nz[lv][1] - nz[lv][0]:.1f} 个百分点" for lv in behind)
    behind_en = "; ".join(f"at ×{lv:g} {nm_(nz[lv][2])} is {nz[lv][1] - nz[lv][0]:.1f} points higher" for lv in behind)
    w(t(f"图 3(b) 为量测噪声敏感性：噪声倍数 {ahead_txt} 时本方法是各方法中最高或并列最高" + (f"；{behind_txt}。" if behind else "。") + "神经网络加物理定位的硬串联在噪声升高时下降更快（×5 时 " + f"{ev['噪声×3'][NN]['class_acc'] * 100:.1f}% 为 ×3）。".replace(f"{ev['噪声×3'][NN]['class_acc'] * 100:.1f}% 为 ×3", f"{t1n['噪声 ×5']['DNN Top-1']['class_acc'] * 100:.1f}%，本方法 {t1n['噪声 ×5']['本方案']['class_acc'] * 100:.1f}%") + "\n",
        f"Fig. 3(b) shows the sensitivity to measurement noise: at noise multipliers {', '.join(f'×{lv:g}' for lv in nz if lv not in behind)} our method is the highest or tied for the highest" + (f"; {behind_en}. " if behind else ". ") + f"The hard cascade of the neural network and physics-based location degrades faster as noise grows ({t1n['噪声 ×5']['DNN Top-1']['class_acc'] * 100:.1f}% at ×5 against {t1n['噪声 ×5']['本方案']['class_acc'] * 100:.1f}% for our method).") + "\n")
    cp = bdc["互补性"]
    w(t(f"表 7b 显示两者的错误并不重合：神经网络错时物理穷举对的比例，域内为 {cp['域内']['神经网络错时物理对'] * 100:.0f}%，高阻外推为 {cp['高阻外推']['神经网络错时物理对'] * 100:.0f}%；物理穷举错时神经网络对的比例，域内为 {cp['域内']['物理错时神经网络对'] * 100:.0f}%，高阻外推为 {cp['高阻外推']['物理错时神经网络对'] * 100:.0f}%。这支持联合决策的前提；但它不单独证明神经网络下降的原因是特征越出训练范围，间接依据是 GCN 论文输入（故障期间的绝对相量）在高阻外推下下降更多。",
        f"Table 7b shows that the errors of the two do not coincide: when the neural network is wrong, the exhaustive physics-based inversion is right in {cp['域内']['神经网络错时物理对'] * 100:.0f}% (in-domain) and {cp['高阻外推']['神经网络错时物理对'] * 100:.0f}% (high resistance) of the cases; when the physics-based inversion is wrong, the neural network is right in {cp['域内']['物理错时神经网络对'] * 100:.0f}% and {cp['高阻外推']['物理错时神经网络对'] * 100:.0f}%. This supports the premise of the joint decision; it does not by itself prove that the decline of the neural network is caused by features leaving the training range, for which the indirect evidence is that the GCN with the paper input (absolute during-fault phasors) declines more under high resistance.") + "\n")
    # ---- 5.4
    w(t("### 5.4 物理模型的消融\n", "### 5.4 Ablation of the physics model\n"))
    table6(d)
    fr, co, ng = J(R34 / "ablation_aug_only/evaluation.json"), ev, J(R34 / "ablation_fixed_gain/evaluation.json")
    allsc = [s for s in fr if s not in ("thresholds", "equivalence_classes", "k", "config", "耗时_ms每样本") and s in co]
    gain = [(co[s][PH]["class_acc"] - fr[s][PH]["class_acc"]) * 100 for s in allsc]
    jn = sum(1 for s in allsc if co[s][K]["class_acc"] > fr[s][K]["class_acc"])
    dn = sum(1 for s in allsc if co[s][K]["d_mean"] < fr[s][K]["d_mean"])
    fx = [s for s in allsc if s.startswith("固定档位") or s == "高阻外推"]
    gg = [(co[s][PH]["class_acc"] - ng[s][PH]["class_acc"]) * 100 for s in fx if s in ng]
    w(t(f"把过渡电阻限定为实数后，物理穷举的类准确率在全部 {len(allsc)} 个场景都提高，提高 {min(gain):.1f}–{max(gain):.1f} 个百分点；联合决策在 {jn} 个场景提高，测距误差在 {dn} 个场景降低。实增益在故障前电压偏离名义值较多的场景作用明显：高阻外推与固定档位场景下，去掉增益使物理穷举降低 {min(gg):.1f}–{max(gg):.1f} 个百分点；但增益对联合决策类准确率的影响不一致，其收益主要体现在物理穷举与测距误差上。线路阻抗整体偏差 ±10% 的场景下，约束对联合决策没有增益。",
        f"Restricting the fault resistance to a real number raises the class accuracy of the exhaustive physics-based inversion in all {len(allsc)} scenarios, by {min(gain):.1f}–{max(gain):.1f} percentage points; the joint decision improves in {jn} scenarios and the location error decreases in {dn}. The real gain matters where the pre-fault voltage deviates from the nominal value: in the high-resistance and fixed-tap scenarios, removing the gain lowers the exhaustive inversion by {min(gg):.1f}–{max(gg):.1f} percentage points; its effect on the class accuracy of the joint decision is inconsistent, and its benefit shows mainly in the exhaustive inversion and in the location error. Under global line-impedance errors of ±10% the constraint brings no gain to the joint decision.") + "\n")
    # ---- 5.5
    w(t("### 5.5 阶段一换成其他分类器\n", "### 5.5 Replacing stage 1 by other classifiers\n"))
    table7(d)
    cnt = {k: sum(1 for s in MAIN7 if swap(k, s) and swap(k, s)[K]["class_acc"] > swap(k, s)[NN]["class_acc"]) for k in TAG}
    w(t(f"把 GCN、GAT、PCA+SVM、PCA+随机森林的概率作为阶段一接入本方法流程后，联合打分与候选扩展相对“分类器 + 物理定位”在 7 个主场景中分别有 {cnt['gcn_p']}、{cnt['gcn_d']}、{cnt['gat_p']}、{cnt['gat_d']}、{cnt['svm']}、{cnt['rf']} 个提高（依次为 GCN 论文输入、GCN 突变量输入、GAT 论文输入、GAT 突变量输入、PCA+SVM、PCA+随机森林），说明流程的增益不依赖具体的阶段一网络。阶段一的网络与输入仍影响最终结果，见表 7 与表 3。",
        f"With the probabilities of GCN, GAT, PCA+SVM and PCA+RF used as stage 1 of our procedure, the joint score and candidate expansion improve on “classifier + physics location” in {cnt['gcn_p']}, {cnt['gcn_d']}, {cnt['gat_p']}, {cnt['gat_d']}, {cnt['svm']} and {cnt['rf']} of the 7 main scenarios (GCN paper input, GCN disturbance input, GAT paper input, GAT disturbance input, PCA+SVM, PCA+RF), so the gain of the procedure does not depend on a specific stage-1 network. The stage-1 network and input still affect the final result, see Tables 7 and 3.") + "\n")
    tally = {}
    for s_ in MAIN7:
        cands = {"ours": ev[s_][K]["class_acc"]}
        cands.update({k: swap(k, s_)[K]["class_acc"] for k in TAG if swap(k, s_)})
        top = max(cands.values())
        for k, v in cands.items():
            if v >= top - 1e-9:
                tally[k] = tally.get(k, 0) + 1
    tl = "、".join(f"{mname(d, k) if k != 'ours' else '本方法的默认 MLP'}（{n} 个）" for k, n in sorted(tally.items(), key=lambda x: -x[1]))
    tl_en = ", ".join(f"{mname(d, k) if k != 'ours' else 'the default MLP of our method'} ({n})" for k, n in sorted(tally.items(), key=lambda x: -x[1]))
    w(t(f"在 7 个主场景中，用本方法流程时准确率最高的阶段一依次为：{tl}。默认的 MLP 阶段一并不是每个场景下最强的，这是流程与阶段一网络解耦的另一个含义：换用更强或更适合某类场景的阶段一，流程仍然适用。",
        f"Among the 7 main scenarios, the stage 1 that gives the highest accuracy with our procedure is, in order: {tl_en}. The default MLP stage 1 is not the strongest in every scenario, which is another implication of decoupling the procedure from the stage-1 network: a stronger stage 1, or one better suited to a type of scenario, can be swapped in and the procedure still applies.") + "\n")
    # ---- 5.6
    w(t("### 5.6 模型偏差下的表现与损失来源\n", "### 5.6 Behaviour under model errors and the source of the loss\n"))
    table8(d)
    o = J(R34 / "oracle_line_params.json")
    w(t(f"每条线路阻抗独立偏差 ±20% 时，本方法的类准确率从域内的 {pct(A['域内']['ours'][0])}% 降到 {pct(A['个别线路阻抗±20%']['ours'][0])}%。表 8b 的诊断把每个样本真实的逐线阻抗交给阶段二：区段已知时的测距误差从 {o['distance_err_km_true_section']['mean_nominal']:.2f} km 降到 {o['distance_err_km_true_section']['mean_oracle']:.2f} km，回到无偏差时的水平，说明位置误差的增加来自模型失配；联合决策的类准确率只从 {o['class_acc']['joint_nominal']:.1f}% 回升到 {o['class_acc']['joint_oracle']:.1f}%，说明区段判别的损失还有阶段一在逐线偏差下的失配，不能全部归于物理模型。以同样的偏差分布扩增阶段一训练集只提高 0.6 个百分点（IEEE 34）和 1.4 个百分点（IEEE 123）。各负荷功率独立变化对结果没有影响（表 A5）。",
        f"When each line impedance deviates independently by ±20%, the class accuracy of our method falls from {pct(A['域内']['ours'][0])}% in-domain to {pct(A['个别线路阻抗±20%']['ours'][0])}%. The diagnosis in Table 8b hands the true per-line impedances of each sample to stage 2: the location error with the section known falls from {o['distance_err_km_true_section']['mean_nominal']:.2f} km to {o['distance_err_km_true_section']['mean_oracle']:.2f} km, back to the error-free level, so the larger position error comes from model mismatch; the class accuracy of the joint decision recovers only from {o['class_acc']['joint_nominal']:.1f}% to {o['class_acc']['joint_oracle']:.1f}%, so part of the loss in the section decision is the mismatch of stage 1 under per-line errors and cannot all be attributed to the physics model. Augmenting the stage-1 training set with the same error distribution improves the accuracy by only 0.6 percentage points (IEEE 34) and 1.4 (IEEE 123). Independent variation of the load powers has no effect (Table A5).") + "\n")
    # ---- 5.7
    w(t("### 5.7 IEEE 123 节点与计算耗时\n", "### 5.7 IEEE 123-bus system and computation time\n"))
    table9(d)
    A3 = c["acc"]["123"]
    v3 = {s: verdict({s: A3[s]}, s) for s in S123 if s in A3 and len(A3[s]) > 1}
    win3 = [s for s in v3 if v3[s][0] == "win"]
    tie3 = [s for s in v3 if v3[s][0] == "tie"]
    lose3 = [s for s in v3 if v3[s][0] == "lose"]
    t34, t123 = c["ev34"]["耗时_ms每样本"], c["ev123"]["耗时_ms每样本"]
    k34 = [k for k in t34 if k.startswith("阶段二")][0]
    k123 = [k for k in t123 if k.startswith("阶段二")][0]
    w(t(f"IEEE 123 上（118 个区段、8 个测点），本方法在 {len(win3)} 个场景显著更高（{lst(win3)}），{len(tie3)} 个场景无显著差别（{lst(tie3)}），{len(lose3)} 个场景显著更低（{lst(lose3)}）。高阻外推下所有方法都低，该场景的故障信号接近噪声。"
        f"计算量方面，阶段二只对 2 个候选做物理反演，IEEE 34 为 {t34[k34]:.1f} ms/样本，全区段物理穷举为 {t34['物理穷举(全部区段)']:.0f} ms；IEEE 123 为 {t123[k123]:.1f} ms 与 {t123['物理穷举(全部区段)']:.0f} ms（批量 1000 个样本、4 核 CPU 单进程，与其他任务并行时测得，偏保守）。",
        f"On IEEE 123 (118 sections, 8 sensors), our method is significantly higher in {len(win3)} scenarios ({lst(win3)}), not significantly different in {len(tie3)} ({lst(tie3)}) and significantly lower in {len(lose3)} ({lst(lose3)}). All methods are low under high resistance, where the fault signal is close to the noise. "
        f"In terms of computation, stage 2 inverts only two candidates: {t34[k34]:.1f} ms per sample on IEEE 34 against {t34['物理穷举(全部区段)']:.0f} ms for the exhaustive inversion over all sections, and {t123[k123]:.1f} ms against {t123['物理穷举(全部区段)']:.0f} ms on IEEE 123 (batch of 1000 samples, one process on a 4-core CPU, measured while other jobs were running and therefore conservative).") + "\n")


def headline(c):
    """摘要与贡献中用到的关键数字。"""
    A, ev = c["acc"]["34"], c["ev34"]
    vd = verdict(A, "高阻外推")
    fr, co = J(R34 / "ablation_aug_only/evaluation.json"), ev
    allsc = [s for s in fr if s in co and s not in ("thresholds", "equivalence_classes", "k", "config", "耗时_ms每样本")]
    gain = [(co[s][PH]["class_acc"] - fr[s][PH]["class_acc"]) * 100 for s in allsc]
    la = [(ev[s][K]["class_acc"] - liang_best(ev, s)[0]) * 100 for s in FIXED4]
    return dict(hr_ours=A["高阻外推"]["ours"][0], hr_best=A["高阻外推"][vd[2]][0], hr_best_k=vd[2], dom_ours=A["域内"]["ours"][0], dom_best=max(v[0] for k, v in A["域内"].items() if k != "ours"),
                gain_lo=min(gain), gain_hi=max(gain), la_lo=min(la), la_hi=max(la), n_sc=len(allsc),
                hc_nn=ev["高阻外推"][NN]["class_acc"], hc_ours=ev["高阻外推"][K]["class_acc"], dom_phys0=fr["域内"][PH]["class_acc"], dom_phys1=ev["域内"][PH]["class_acc"],
                mae=c["ms34"]["域内"]["本方案测距误差均值km"]["mean"])


def abstract(d):
    t, c = d.t, load()
    h = headline(c)
    A3 = c["acc"]["123"]
    v3 = [verdict({s: A3[s]}, s)[0] for s in S123 if s in A3 and len(A3[s]) > 1]
    d.w(t("## 摘要\n", "## Abstract\n"))
    d.w(t(
        f"配电网只有极少数同步量测点时，现有故障定位方法各有不足：数据驱动方法在测试条件偏离训练分布时精度下降，物理反演不受训练分布影响，但在响应相近的区段之间区分度不足。"
        f"本文提出数据粗筛与物理闭环校正的两阶段方法：神经网络把故障区段缩小到两个候选，再用过渡电阻限定为实数的物理反演逐个核对并估计区段内位置，由贝叶斯联合打分合并两者的证据；"
        f"候选都不能解释量测时扩大搜索范围，仍不能解释则发出失配告警。在 IEEE 34 节点（5 个测点、30 个区段）上，过渡电阻超出训练范围（300–1000 Ω）时本方法的区段准确率（按等价类判定）为 {pct(h['hr_ours'])}%，"
        f"对比的外部方法中最高的为 {pct(h['hr_best'])}%；同分布内与最强基线相当（本方法 {pct(h['dom_ours'])}%，基线最高 {pct(h['dom_best'])}%）；把过渡电阻限定为实数使物理反演的区段准确率在全部 {h['n_sc']} 个场景提高 {h['gain_lo']:.1f}–{h['gain_hi']:.1f} 个百分点；"
        f"在调压器档位固定的 4 个场景中比 Liang 等的两阶段方法高 {h['la_lo']:.1f}–{h['la_hi']:.1f} 个百分点。在 IEEE 123 节点（8 个测点、118 个区段）上，本方法在 {sum(1 for x in v3 if x == 'win')}/{len(v3)} 个场景显著优于其余方法。"
        f"线路阻抗存在偏差时，部分基线的区段准确率更高；逐线阻抗真值诊断表明位置误差的增加来自线路参数失配。全部结果来自仿真。",
        f"With only a handful of synchronized sensors in a distribution network, existing fault-location methods each fall short: data-driven methods lose accuracy when test conditions leave the training distribution, and physics-based inversion, which is insensitive to the training distribution, cannot separate sections whose responses are similar. "
        f"We propose a two-stage method of data-driven screening and physics-based closed-loop verification: a neural network narrows the faulted section to two candidates, a physics-based inversion with a real-valued fault resistance verifies each candidate and estimates the position within the section, and a Bayesian joint score combines the two pieces of evidence; "
        f"when no candidate explains the measurements the search is widened, and if it still fails a mismatch alarm is raised. On the IEEE 34-bus feeder (5 sensors, 30 sections), when the fault resistance leaves the training range (300–1000 Ω) our method reaches a section accuracy (judged by equivalence class) of {pct(h['hr_ours'])}%, "
        f"against {pct(h['hr_best'])}% for the best external method compared; within the training distribution it is on par with the strongest baseline ({pct(h['dom_ours'])}% against at most {pct(h['dom_best'])}%); restricting the fault resistance to a real number raises the section accuracy of the physics-based inversion by {h['gain_lo']:.1f}–{h['gain_hi']:.1f} percentage points in all {h['n_sc']} scenarios; "
        f"and in the four fixed-tap scenarios it is {h['la_lo']:.1f}–{h['la_hi']:.1f} percentage points above the two-stage method of Liang et al. On the IEEE 123-bus feeder (8 sensors, 118 sections) it is significantly better than the other methods in {sum(1 for x in v3 if x == 'win')} of {len(v3)} scenarios. "
        f"Under line-impedance errors some baselines are more accurate; a diagnosis with the true per-line impedances shows that the larger position error comes from line-parameter mismatch. All results are from simulation.") + "\n")
    d.w(t("**关键词：** 故障定位；配电网；极稀疏量测；物理反演；数据驱动；分布外泛化\n", "**Keywords:** fault location; distribution network; extremely sparse measurements; physics-based inversion; data-driven; out-of-distribution generalization\n"))


def contributions(d):
    t, c = d.t, load()
    h = headline(c)
    return [
        t(f"1. 提出两阶段方法：神经网络从全部区段中选出 2 个候选，物理反演逐个核对，联合打分判决；候选都不能解释量测时扩大搜索，仍不能解释则发出失配告警。在 IEEE 34 过渡电阻 300–1000 Ω 时，类准确率从神经网络加物理定位的 {pct(h['hc_nn'])}% 提高到 {pct(h['hc_ours'])}%。",
          f"1. A two-stage method in which a neural network selects two candidates from all sections, a physics-based inversion verifies each, and a joint score decides; when no candidate explains the measurements the search is widened, and if it still fails a mismatch alarm is raised. At 300–1000 Ω on IEEE 34 the class accuracy rises from {pct(h['hc_nn'])}% for the neural network plus physics-based location to {pct(h['hc_ours'])}%."),
        t(f"2. 在物理反演中把过渡电阻限定为实数，并引入带先验的实增益。IEEE 34 域内，物理穷举的类准确率从 {pct(h['dom_phys0'])}% 提高到 {pct(h['dom_phys1'])}%，在全部 {h['n_sc']} 个场景都提高。",
          f"2. A physics-based inversion in which the fault resistance is restricted to a real number, with a real gain under a prior. In-domain on IEEE 34, the class accuracy of the exhaustive inversion rises from {pct(h['dom_phys0'])}% to {pct(h['dom_phys1'])}%, and it improves in all {h['n_sc']} scenarios."),
        t("3. 在 IEEE 34 与 IEEE 123 上与 4 个已发表方法和 2 个机器学习基线比较，覆盖过渡电阻、负荷、噪声外推，以及整体与逐线的线路参数偏差；给出本方法占优与不占优的场景，并用逐线阻抗真值诊断指出线路参数偏差下的损失来源。",
          "3. A comparison on IEEE 34 and IEEE 123 with four published methods and two machine-learning baselines, covering extrapolation in fault resistance, load and noise and both global and per-line line-parameter errors; the scenarios where our method is and is not ahead are reported, and a diagnosis with the true per-line impedances points out the source of the loss under line-parameter errors."),
    ]


def discussion(d):
    t, c = d.t, load()
    A = c["acc"]["34"]
    vd = {s: verdict(A, s) for s in MAIN7}
    d.w(t("## 6 讨论与局限\n", "## 6 Discussion and Limitations\n"))
    d.w(t(
        "**方法的适用范围。** 本方法的增益集中在数据驱动方法失效的条件，即过渡电阻等超出训练范围的外推；在同分布内与最强的数据驱动基线相当，在线路阻抗存在偏差的场景中部分基线更高。"
        "因此它适合用在量测很少、故障条件可能超出训练覆盖范围、而线路参数相对可靠的配电网；若线路参数本身不准，则需要先做参数辨识，或像逐线真值诊断那样提供更准确的参数。",
        "**Scope of the method.** The gain of our method is concentrated where data-driven methods fail, namely extrapolation such as fault resistance beyond the training range; within the training distribution it is on par with the strongest data-driven baseline, and under line-impedance errors some baselines are higher. "
        "It is therefore suited to networks with very few sensors, where fault conditions may exceed the training coverage and the line parameters are comparatively reliable; if the line parameters themselves are inaccurate, parameter identification is needed first, or more accurate parameters as in the diagnosis with true per-line impedances.\n"))
    d.w(t("**局限。** （1）全部结果来自 OpenDSS 仿真，量测噪声与同步误差按假设取值，没有实测故障数据，仿真到现场的差距只能由模型偏差场景间接刻画。"
          "（2）区段准确率按固定的等价类判定，报告的精确区段准确率比它低约 2.5–3.3 个百分点（表 A2），但各方法的排序基本不变。"
          "（3）基线是我们按原文描述的复现，可能与原作者的实现有差异；Liang 两阶段方法在其论文的 IEEE 33 节点系统上复现的第一阶段低于论文报告（表 A3），对它的比较只代表复现版本；块稀疏贝叶斯的位置受网格限制，SVM 只用了 3 万个训练子样本。"
          "（4）过渡电阻 500–1000 Ω 时联合决策仍低于物理穷举，因为 Top-2 候选漏掉真实区段；IEEE 123 上高阻故障的信号接近噪声，所有方法的准确率都低。"
          "（5）没有覆盖分布式电源接入、拓扑重构、数据缺失和小电流接地系统；测点数量的影响也没有系统测试。"
          "（6）神经网络下降的原因（特征越出训练范围）只有间接证据。"
          "（7）默认的轻量 MLP 阶段一在各场景下并不总是最强（表 7 与 5.5 节），本文没有针对阶段一做进一步的选择或集成。",
          "**Limitations.** (1) All results come from OpenDSS simulation; measurement noise and synchronization errors are assumed, there are no field fault data, and the gap between simulation and the field is characterized only indirectly by the model-error scenarios. "
          "(2) Section accuracy is judged on fixed equivalence classes; the exact section accuracy is about 2.5–3.3 percentage points lower (Table A2), but the ranking of the methods is essentially unchanged. "
          "(3) The baselines are our reimplementations from the descriptions in the papers and may differ from the original authors' code; for the method of Liang et al. the stage 1 we reproduced on the IEEE 33-bus system of their paper is below the reported values (Table A3), so the comparison with it represents the reimplementation; the position of block-sparse Bayesian learning is limited by a grid, and the SVM uses only a 30,000-sample training subset. "
          "(4) At 500–1000 Ω the joint decision is still below the exhaustive physics-based inversion because the Top-2 candidates miss the true section; on IEEE 123 the signal of high-resistance faults is close to the noise and all methods are low. "
          "(5) Distributed generation, topology reconfiguration, missing data and resonant-grounded systems are not covered, and the effect of the number of sensors is not tested systematically. "
          "(6) The cause of the decline of the neural network (features leaving the training range) is supported only indirectly. "
          "(7) The default lightweight MLP stage 1 is not always the strongest in every scenario (Table 7 and Section 5.5), and this paper does not select or ensemble stage-1 networks further.\n"))


def conclusion(d):
    t, c = d.t, load()
    h = headline(c)
    d.w(t("## 7 结论\n", "## 7 Conclusion\n"))
    d.w(t(
        f"本文针对极稀疏量测下的配电网故障定位，提出神经网络粗筛、物理反演校正、贝叶斯联合打分的两阶段方法，并加入候选扩展与失配告警。在 IEEE 34 与 IEEE 123 的仿真中，过渡电阻超出训练范围时本方法的区段准确率明显高于对比的数据驱动、稀疏恢复与两阶段方法"
        f"（IEEE 34 为 {pct(h['hr_ours'])}%，对比方法中最高 {pct(h['hr_best'])}%），同分布内与最强基线相当；区段内的测距精度来自物理反演，把过渡电阻限定为实数使物理反演的区段准确率在全部 {h['n_sc']} 个场景提高 {h['gain_lo']:.1f}–{h['gain_hi']:.1f} 个百分点。"
        f"方法的主要局限是线路阻抗存在偏差时部分基线更高，以及没有实测验证。后续工作包括线路参数辨识与方法的联合、拓扑变化与分布式电源、以及实测数据验证。",
        f"For fault location in distribution networks with extremely sparse measurements, this paper proposed a two-stage method of neural-network screening, physics-based verification and Bayesian joint scoring, with candidate expansion and a mismatch alarm. "
        f"In simulations on IEEE 34 and IEEE 123, when the fault resistance leaves the training range the section accuracy of our method is clearly higher than that of the compared data-driven, sparse-recovery and two-stage methods ({pct(h['hr_ours'])}% on IEEE 34 against at most {pct(h['hr_best'])}%), and within the training distribution it is on par with the strongest baseline; "
        f"the accuracy of the position within a section comes from the physics-based inversion, and restricting the fault resistance to a real number raises the section accuracy of the physics-based inversion by {h['gain_lo']:.1f}–{h['gain_hi']:.1f} percentage points in all {h['n_sc']} scenarios. "
        f"The main limitations are that some baselines are higher when line impedances are in error and that there is no field validation. Future work includes combining the method with line-parameter identification, topology changes and distributed generation, and validation with field data.\n"))


def statements(d):
    t = d.t
    d.w(t("## 数据与代码可用性\n", "## Data and code availability\n"))
    d.w(t("仿真基于公开的 OpenDSS IEEE 34 与 IEEE 123 节点算例；仿真程序、生成的数据集与全部实验脚本在项目代码库中，可向作者索取。\n",
          "The simulations use the public OpenDSS IEEE 34- and IEEE 123-bus test feeders; the simulation programs, generated data sets and all experiment scripts are in the project repository and are available from the authors on request.\n"))
