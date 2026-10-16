"""初稿的数据装载、表格与结果章节。所有数字来自 results/improved/ 下的实验结果；文字中的比较结论由代码按数据判断后写入。"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R34, R123 = ROOT / "results/improved/ieee34", ROOT / "results/improved/ieee123"
J = lambda p: json.load(open(p))
_pe = {}


def pe(s, key):
    """种子 0 的测距误差均值（km，沿馈线最短路径距离），由 analyze_path_error.py 统计。"""
    if not _pe:
        _pe.update(J(R34 / "path_error.json"))
    return _pe[s][key]["mean"]

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
           ("bsbl", "简化块稀疏贝叶斯", "Simplified block-sparse Bayesian"), ("svm", "PCA+SVM", "PCA+SVM"), ("rf", "PCA+随机森林", "PCA+RF")]

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
        [t("简化块稀疏贝叶斯", "Simplified block-sparse Bayesian"), t("字典与迭代", "Dictionary and iterations"), t("字典为各区段在 α = 0.05–0.95（10 点）的三相注入电流响应，块大小为区段相数；EM 迭代 60 次；量测噪声方差取与阶段二相同的加权；取后验能量最大的块", "Dictionary of three-phase injection-current responses of each section at α = 0.05–0.95 (10 points), block size equal to the number of phases; 60 EM iterations; noise variance weighted as in stage 2; the block with the largest posterior energy is chosen")],
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

def liang_best(ev, s):
    cands = [(ev[s][f"{L} {p}"]["class_acc"], ev[s][f"{L} {p}"]["d_mean"], f"{L} {p}") for L in ("L1(同测点)", "L1(自身布点)") for p in ("论文流程", "全候选线路细化")]
    return max(cands)

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
    hdr = [t("系统", "System"), t("场景", "Scenario"), t("高置信样本占比 (%)", "High-confidence share (%)"), t("高置信准确率 (%)", "High-confidence accuracy (%)"), t("低置信准确率 (%)", "Low-confidence accuracy (%)"), t("校准误差 ECE", "ECE"), t("置信度 AUROC", "Confidence AUROC"), t("低置信召回率 (%)", "Low-confidence recall (%)"),
           t("告警占比 (%)", "Alarm share (%)"), t("告警样本错误率 (%)", "Error rate among alarms (%)"), t("告警召回率 (%)", "Alarm recall (%)"), t("整体错误率 (%)", "Overall error rate (%)")]
    rows = []
    for sysname, ev_, al in (("IEEE 34", ev, J(R34 / "alarm.json")), ("IEEE 123", c["ev123"], J(R123 / "alarm.json"))):
        for s in ("域内", "负荷偏移", "高阻外推", "噪声×3", "个别线路阻抗±20%"):
            x, a_ = ev_[s]["置信度C"], al[s]
            rows.append([sysname, sc(d, s), f"{x['高置信占比'] * 100:.0f}", f"{x['高置信准确率'] * 100:.1f}", f"{x['低置信准确率'] * 100:.1f}", f"{x['ECE']:.3f}", f"{a_['置信度AUROC']:.2f}", f"{a_['低置信召回率'] * 100:.1f}",
                         f"{a_['告警占比'] * 100:.1f}", f"{a_['告警样本错误率'] * 100:.1f}", f"{a_['告警召回率'] * 100:.1f}", f"{(1 - a_['准确率']) * 100:.1f}"])
    d.table("A4b", "置信度与失配告警的可靠性（种子 0，错误按等价类判定；高置信指置信度 ≥ 0.6，低置信召回率为全部错误中低置信样本的比例；告警召回率为全部错误中被告警的比例）",
            "Reliability of the confidence and the mismatch alarm (seed 0, errors judged by equivalence class; high confidence means confidence ≥ 0.6, low-confidence recall is the share of all errors that have low confidence; alarm recall is the share of all errors that raise the alarm)", hdr, rows)

def appendix_baselines(d):
    t = d.t
    rows = [
        ["GCN", t("按 Chen 等的描述复现：母线图（每个母线保留最近的 Kn 个距离，高斯核权重），切比雪夫卷积，类别为故障区段（原文为故障母线）；论文输入需要故障前相量，由相同参数重新生成数据得到", "Reimplemented from Chen et al.: bus graph (nearest Kn distances per bus, Gaussian weights), Chebyshev convolutions; classes are fault sections (fault buses in the original); the paper input needs pre-fault phasors, obtained by regenerating the data with the same parameters")],
        ["GAT", t("按 Lingyu Liang 等的描述复现：多头图注意力；注意力限于物理相邻母线；输入与训练同 GCN", "Reimplemented from Lingyu Liang et al.: multi-head graph attention restricted to physically adjacent buses; inputs and training as for the GCN")],
        [t("简化块稀疏贝叶斯", "Simplified block-sparse Bayesian"), t("按 Jiang 等的思路复现并适配：以三相虚拟注入电流为未知量的块稀疏贝叶斯学习，字典取名义参数下各区段在 10 个位置上的响应；与原文的差别：原文以节点注入为未知量并使用共享的块内相关矩阵，本复现取区段内位置且各块独立；位置受 α 网格限制，因此其测距误差包含网格误差；不使用神经网络", "Reimplemented and adapted from the idea of Jiang et al.: block-sparse Bayesian learning with three-phase virtual injection currents as unknowns and a dictionary of nominal-parameter responses of each section at 10 positions. Differences from the original: the original takes node injections as unknowns and uses a shared within-block correlation matrix, whereas this reimplementation uses positions within sections and independent blocks; the position is limited to a 10-point alpha grid, so its location error includes the grid error; no neural network")],
        [t("Liang 两阶段", "Liang two-stage"), t("按论文描述复现，复现保真度见表 A3；假设档位与网络参数精确已知，获得根母线相量、末端电压幅值和全网负荷伪量测，多于本方法的量测", "Reimplemented from the paper, fidelity in Table A3; assumes exactly known taps and network parameters and receives root-bus phasors, end-bus voltage magnitudes and load pseudo-measurements, more than our method")],
        ["PCA+SVM / PCA+RF", t("沿用 Chen 等的对照协议；输入为本方法阶段一的特征；SVM 受计算量限制只用 3 万个训练子样本", "Following the comparison protocol of Chen et al.; inputs are the stage-1 features of our method; the SVM uses only a 30,000-sample training subset because of its cost")],
    ]
    d.table("A6", "基线的复现说明与局限", "Reimplementation notes and limitations of the baselines", [t("方法", "Method"), t("说明", "Notes")], rows, [":--", ":--"])

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
                mae=pe("域内", "ours"))

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

def statements(d):
    t = d.t
    d.w(t("## 数据与代码可用性\n", "## Data and code availability\n"))
    d.w(t("仿真基于公开的 OpenDSS IEEE 34 与 IEEE 123 节点算例；仿真程序、生成的数据集与全部实验脚本在项目代码库中，可向作者索取。\n",
          "The simulations use the public OpenDSS IEEE 34- and IEEE 123-bus test feeders; the simulation programs, generated data sets and all experiment scripts are in the project repository and are available from the authors on request.\n"))


# ================================================================ 核心场景与结果表格（精简版）
CORE5 = ["域内", "负荷偏移", "高阻外推", "噪声×3", "个别线路阻抗±20%"]
EXTRA7 = ["线路阻抗+10%", "线路阻抗-10%", "线路阻抗+5%", "各负荷独立变化", "线路与负荷同时偏差", "固定档位 阻抗+10%", "固定档位 阻抗-10%"]
CLS = {"gcn": ("gcn_p", "gcn_d"), "gat": ("gat_p", "gat_d")}   # 同一方法的两种输入（论文输入 / 突变量输入）


def best_in(A_s, fam):
    """该场景下方法族 fam（gcn、gat）两种输入中均值较高者：返回 (键, (均值, 标准差, n))；两种输入都缺时返回 (None, None)。"""
    c = [(k, A_s[k]) for k in CLS[fam] if k in A_s]
    return max(c, key=lambda x: x[1][0]) if c else (None, None)


def tag_of(k):
    return "" if k is None else (" (P)" if k.endswith("_p") else " (D)")


def row_cells(A_s):
    """主表的一行：本方法、GCN、GAT（各取较高的输入并标注）、块稀疏贝叶斯、PCA+SVM、PCA+随机森林。返回 [(显示名键, 值, 后缀)]。"""
    kg, vg = best_in(A_s, "gcn")
    ka, va = best_in(A_s, "gat")
    return [("ours", A_s.get("ours"), ""), ("gcn", vg, tag_of(kg)), ("gat", va, tag_of(ka)), ("bsbl", A_s.get("bsbl"), ""), ("svm", A_s.get("svm"), ""), ("rf", A_s.get("rf"), "")], (kg, ka)


FAM_NAME = {"ours": ("本方法", "Ours"), "gcn": ("GCN", "GCN"), "gat": ("GAT", "GAT"), "bsbl": ("简化块稀疏贝叶斯", "Simplified block-sparse Bayesian"), "svm": ("PCA+SVM", "PCA+SVM"), "rf": ("PCA+随机森林", "PCA+RF")}


def fam_hdr(d):
    return [FAM_NAME[k][0 if d.L == "zh" else 1] for k in ("ours", "gcn", "gat", "bsbl", "svm", "rf")]


def fmt_row(items):
    vals = [v for _, v, _ in items]
    cells = bold_best(vals)
    return [c + suf if v else c for c, (_, v, suf) in zip(cells, items)]


def table_main(d):
    """表 2：核心 5 个场景下与各外部方法（Liang 两阶段除外）的类准确率。"""
    t, c = d.t, load()
    rows = []
    for s in CORE5:
        items, _ = row_cells(c["acc"]["34"][s])
        rows.append([sc(d, s)] + fmt_row(items))
    d.table(2, "IEEE 34 上的主对比：类准确率（%，均值±标准差）。GCN、GAT 各取该场景下较高的输入（P：论文输入，D：突变量输入）；每行最优加粗",
            "Main comparison on IEEE 34: class accuracy (%, mean ± standard deviation). For GCN and GAT the input with the higher accuracy in each scenario is shown (P: paper input, D: disturbance input); the best in each row is in bold",
            [t("场景", "Scenario")] + fam_hdr(d), rows)


def table_paired(d, system="34"):
    """表 2b（IEEE 34）与表 9b（IEEE 123）：本方法与各场景中均值最高的其余方法的配对比较（种子 0，同一批测试样本）。"""
    t = d.t
    pt = J((R34 if system == "34" else R123) / "paired_tests.json")
    fmtp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    rows = []
    for s in [s for s in CORE5 if s in pt]:
        r = pt[s]
        cell = lambda tst: f"{tst['diff']:+.1f} [{tst['lo']:+.1f}, {tst['hi']:+.1f}]"
        rows.append([sc(d, s), mname(d, r["best_other"]), f"{r['ours_class'] * 100:.1f} / {r['other_class'] * 100:.1f}", cell(r["class_test"]), fmtp(r["class_test"]["p"]),
                     f"{r['ours_exact'] * 100:.1f} / {r['other_exact'] * 100:.1f}", cell(r["exact_test"]), fmtp(r["exact_test"]["p"])])
    d.table("2b" if system == "34" else "9b", "本方法与各场景中类准确率均值最高的其余方法的配对比较（种子 0，" + ("同一批 5000 个测试样本" if system == "34" else "同一批测试样本；对方只在保存了种子 0 逐样本预测的方法中取，个别线路阻抗 ±20% 因 GCN 种子 0 未运行该场景而省略") + "）：本方法 / 对方的准确率（%）、差值（百分点）及配对 bootstrap 95% 置信区间、McNemar 精确检验 p 值；左为等价类准确率，右为精确区段准确率",
            "Paired comparison of our method with the other method of highest mean class accuracy in each scenario (seed 0, " + ("the same 5,000 test samples" if system == "34" else "the same test samples; the other method is taken among those with saved seed-0 per-sample predictions, and the individual-line-impedance ±20% scenario is omitted because GCN did not run it at seed 0") + "): accuracy of ours / the other (%), difference (percentage points) with the paired-bootstrap 95% confidence interval, and the exact McNemar p value; class accuracy on the left, exact section accuracy on the right",
            [t("场景", "Scenario"), t("对方", "Other method"), t("类准确率 本方法/对方", "Class acc. ours/other"), t("差值 [95% CI]", "Difference [95% CI]"), "p",
             t("精确区段 本方法/对方", "Exact acc. ours/other"), t("差值 [95% CI]", "Difference [95% CI]"), "p"], rows)
    return pt


def sensors_sweep():
    return J(ROOT / "results/improved/ieee34_sensors/sensor_sweep.json")


def table_sensors(d):
    """附录表 A10：测点数量扫描的数值（IEEE 34，3 个种子）。"""
    t = d.t
    sw = sensors_sweep()
    ks = list(sw)
    m = lambda k, sc_, key, mult=100, prec=1: f"{sw[k][sc_][key]['mean'] * mult:.{prec}f}±{sw[k][sc_][key]['std'] * mult:.{prec}f}"
    rows = []
    for sc_ in ("域内", "高阻外推"):
        for k in ks:
            rows.append([sc(d, sc_), k[1:], str(sw[k]["等价类数"]),
                         m(k, sc_, "本方法 精确区段"), m(k, sc_, "GCN 精确区段"), m(k, sc_, "物理穷举 精确区段"),
                         m(k, sc_, "本方法 测距误差", 1, 2), m(k, sc_, "GCN+物理定位 测距误差", 1, 2), m(k, sc_, "物理穷举 测距误差", 1, 2), m(k, sc_, "真实区段已知 测距误差", 1, 2)])
    d.table("A10", "测点数量扫描的数值（IEEE 34，3 个种子，均值±标准差）：精确区段准确率（%）与测距误差（km，沿馈线最短路径距离）；“区段已知”为已知真实区段时本方法阶段二的位置误差",
            "Numbers of the sensor-count sweep (IEEE 34, three seeds, mean ± standard deviation): exact section accuracy (%) and location error (km, shortest path along the feeder); “section known” is the position error of stage 2 when the true section is given",
            [t("场景", "Scenario"), t("测点数", "Sensors"), t("等价类数", "Classes"), t("精确区段 本方法", "Exact: ours"), t("精确区段 GCN", "Exact: GCN"), t("精确区段 物理穷举", "Exact: exhaustive physics"),
             t("测距 本方法", "Error: ours"), t("测距 GCN+物理定位", "Error: GCN+physics"), t("测距 物理穷举", "Error: exhaustive physics"), t("测距 区段已知", "Error: section known")], rows)


def sensors_section(d):
    """5.3：测点数量扫描。"""
    t = d.t
    sw = sensors_sweep()
    ks = list(sw)
    g = lambda k, sc_, key: sw[k][sc_][key]["mean"]
    k0, k1, k5, kn = ks[0], ks[-1], "n5", ks[-1]
    d.w(t("### 5.3 测点减少时，各项指标如何变化？\n", "### 5.3 How do the metrics change as sensors are removed?\n"))
    d.fig(4)
    acc = lambda k, sc_, key="本方法 精确区段": g(k, sc_, key) * 100
    gain_in = (acc(k5, "域内") - acc(k0, "域内"), acc(k1, "域内") - acc(k5, "域内"))
    gcn_gap = {sc_: [acc(k, sc_) - acc(k, sc_, "GCN 精确区段") for k in ks] for sc_ in ("域内", "高阻外推")}
    km_gap = {sc_: [g(k, sc_, "GCN+物理定位 测距误差") - g(k, sc_, "本方法 测距误差") for k in ks] for sc_ in ("域内", "高阻外推")}
    ph_hr = [acc(k, "高阻外推", "物理穷举 精确区段") - acc(k, "高阻外推") for k in ks]
    d.w(t(f"测点取贪心顺序的前 K 个（K = {'、'.join(k[1:] for k in ks)}；前 5 个即主实验的测点，其余由同一准则继续选取，测点集合是嵌套的，不是对每个 K 单独最优的布置）。每个 K 都重新训练阶段一与 GCN（突变量输入）并重新标定阶段三，等价类按各 K 的测点重新定义（{sw[k0]['等价类数']}–{sw[k1]['等价类数']} 类），"
          "类准确率在不同 K 之间不可比，因此图 4 报精确区段准确率和测距误差；三个种子，其余设置同主实验（附录表 A10；K = 5 的数值与表 2 略有差别，因为扩增训练集重新生成、种子数不同）。",
          f"The sensors are the first K of the greedy order (K = {', '.join(k[1:] for k in ks)}; the first five are those of the main experiments and the rest are chosen by the same criterion, so the sets are nested and not the optimal placement for each K). For every K, stage 1 and the GCN (disturbance input) are retrained and stage 3 is recalibrated, and the equivalence classes are redefined for the sensors of that K ({sw[k0]['等价类数']}–{sw[k1]['等价类数']} classes), "
          "so class accuracy is not comparable across K and Fig. 4 reports the exact section accuracy and the location error; three seeds, everything else as in the main experiments (Table A10; the numbers at K = 5 differ slightly from Table 2 because the augmentation training sets were regenerated and the number of seeds differs).") + "\n")
    d.w(t(f"域内，本方法的精确区段准确率从 K = {k0[1:]} 的 {acc(k0, '域内'):.1f}% 升到 K = 5 的 {acc(k5, '域内'):.1f}% 和 K = {k1[1:]} 的 {acc(k1, '域内'):.1f}%（后一段只增加 {gain_in[1]:.1f} 个百分点，前一段增加 {gain_in[0]:.1f} 个百分点），测距误差从 {g(k0, '域内', '本方法 测距误差'):.2f} km 降到 {g(k1, '域内', '本方法 测距误差'):.2f} km；"
          f"与 GCN 相比，本方法的精确区段准确率在各 K 下相差 {min(gcn_gap['域内']):+.1f} 到 {max(gcn_gap['域内']):+.1f} 个百分点。高阻外推下，本方法比 GCN 高 {min(gcn_gap['高阻外推']):.1f}–{max(gcn_gap['高阻外推']):.1f} 个百分点，测距误差比 GCN 加物理定位低 {min(km_gap['高阻外推']):.2f}–{max(km_gap['高阻外推']):.2f} km，测点从 {k1[1:]} 个减到 {k0[1:]} 个时这一优势依然存在。"
          f"同 5.2 节，高阻外推下全区段物理穷举的精确区段准确率"
          + (f"在 K ≥ 4 时高于本方法 {min(ph_hr[1:]):.1f}–{max(ph_hr[1:]):.1f} 个百分点" if min(ph_hr[1:]) > 0 else "与本方法相比互有高低") + "，原因同样是 Top-2 候选漏掉了真实区段。",
          f"In-domain, the exact section accuracy of our method rises from {acc(k0, '域内'):.1f}% at K = {k0[1:]} to {acc(k5, '域内'):.1f}% at K = 5 and {acc(k1, '域内'):.1f}% at K = {k1[1:]} (the second step adds only {gain_in[1]:.1f} percentage points, the first {gain_in[0]:.1f}), and the location error falls from {g(k0, '域内', '本方法 测距误差'):.2f} km to {g(k1, '域内', '本方法 测距误差'):.2f} km; "
          f"compared with GCN, the exact section accuracy of our method differs by {min(gcn_gap['域内']):+.1f} to {max(gcn_gap['域内']):+.1f} percentage points across K. Under high resistance our method is {min(gcn_gap['高阻外推']):.1f}–{max(gcn_gap['高阻外推']):.1f} percentage points above GCN and its location error is {min(km_gap['高阻外推']):.2f}–{max(km_gap['高阻外推']):.2f} km below that of GCN followed by physics-based location; this advantage persists when the sensors are reduced from {k1[1:]} to {k0[1:]}. "
          "As in Section 5.2, under high resistance the exact section accuracy of the exhaustive physics-based inversion over all sections " + (f"is {min(ph_hr[1:]):.1f}–{max(ph_hr[1:]):.1f} percentage points above that of our method for K ≥ 4" if min(ph_hr[1:]) > 0 else "is above or below that of our method depending on K") + ", again because the Top-2 candidates miss the true section.") + "\n")


def table_breakdown(d):
    """表 3：域内按故障类型、故障位置分档的类准确率（种子 0）。GCN、GAT 固定取域内较高的输入。"""
    t = d.t
    t1 = J(R34 / "tier1.json")
    pick = {f: max((f + " 论文输入", f + " 突变量输入"), key=lambda k: t1["故障类型 全部"][k]["class_acc"]) for f in ("GCN", "GAT")}
    keys = ["本方案", pick["GCN"], pick["GAT"], "块稀疏贝叶斯学习", "PCA+SVM", "PCA+随机森林"]
    groups = ["故障类型 SLG", "故障类型 LL", "故障类型 LLG", "故障类型 LLLG", "位置 近端（α<0.35）", "位置 中段（0.35≤α<0.65）", "位置 远端（α≥0.65）"]
    en_g = {"故障类型 SLG": "Fault type SLG", "故障类型 LL": "Fault type LL", "故障类型 LLG": "Fault type LLG", "故障类型 LLLG": "Fault type LLLG", "位置 近端（α<0.35）": "Position: near end (α<0.35)",
            "位置 中段（0.35≤α<0.65）": "Position: middle (0.35≤α<0.65)", "位置 远端（α≥0.65）": "Position: far end (α≥0.65)"}
    rows = []
    for g in groups:
        vals = [(t1[g][k]["class_acc"], 0.0, 1) for k in keys]
        rows.append([g if d.L == "zh" else en_g[g], str(t1[g]["本方案"]["n"])] + bold_best(vals))
    inp = lambda k: t("论文输入", "paper input") if "论文" in k else t("突变量输入", "disturbance input")
    d.table(3, f"IEEE 34 域内测试集按故障类型与故障位置分档的类准确率（%，种子 0；GCN 取{inp(pick['GCN'])}，GAT 取{inp(pick['GAT'])}）",
            f"Class accuracy (%, seed 0) of the IEEE 34 in-domain test set by fault type and fault position (GCN with the {inp(pick['GCN'])}, GAT with the {inp(pick['GAT'])})",
            [t("分组", "Group"), t("样本数", "Samples")] + fam_hdr(d), rows)
    return pick


def table_loc(d):
    """表 4：测距误差（km）。分类器格式为“区段中点 / +物理定位”。"""
    t, c = d.t, load()
    rows = []
    for s in CORE5:
        _, (kg, ka) = row_cells(c["acc"]["34"][s])
        ours, bs = pe(s, "ours"), pe(s, "bsbl")
        ph = {fam: (pe(s, f"{k}_mid"), pe(s, f"{k}_phys")) for fam, k in (("gcn", kg), ("gat", ka), ("svm", "svm"), ("rf", "rf"))}
        low = min([round(ours, 2), round(bs, 2)] + [round(v[1], 2) for v in ph.values()])
        f = lambda x: f"**{x:.2f}**" if round(x, 2) == low else f"{x:.2f}"
        rows.append([sc(d, s), f(ours), f(bs)] + [f"{ph[k][0]:.2f} / " + f(ph[k][1]) for k in ("gcn", "gat", "svm", "rf")])
    d.table(4, "IEEE 34 的测距误差均值（km，预测点与真实点沿馈线的最短路径距离，种子 0）。分类器只输出区段，格式为“取区段中点 / 接阶段二物理定位”（GCN、GAT 的输入同表 2）；每行最小值加粗",
            "Mean location error on IEEE 34 (km; shortest path along the feeder between the predicted and the true fault point, seed 0). The classifiers output only the section; the format is “section midpoint / followed by the stage-2 physics-based location” (inputs of GCN and GAT as in Table 2); the smallest in each row is in bold",
            [t("场景", "Scenario"), t("本方法", "Ours"), t("简化块稀疏贝叶斯", "Simplified block-sparse Bayesian"), "GCN", "GAT", "PCA+SVM", t("PCA+随机森林", "PCA+RF")], rows)


ABL5 = ["域内", "负荷偏移", "高阻外推", "噪声×3", "固定档位 域内", "固定档位 负荷偏移"]


def table_ablation(d):
    t = d.t
    f = {"free": J(R34 / "ablation_aug_only/evaluation.json"), "con": load()["ev34"], "nog": J(R34 / "ablation_fixed_gain/evaluation.json")}
    names = (t("自由", "free"), t("约束", "con."), t("无增益", "no gain"))
    hdr = [t("场景", "Scenario")] + [f"{a}：{b}" if d.L == "zh" else f"{a}: {b}" for a in (t("物理穷举", "Exhaustive physics"), t("联合决策", "Joint decision")) for b in names]
    rows = []
    for s in ABL5:
        r = [sc(d, s)]
        for key in (PH, K):
            vals = [f[m][s][key]["class_acc"] for m in ("free", "con", "nog")]
            top = max(round(v * 100, 1) for v in vals)
            r += [f"**{v * 100:.1f}**" if round(v * 100, 1) == top else f"{v * 100:.1f}" for v in vals]
        rows.append(r)
    d.table(5, "物理模型的受控消融（IEEE 34，类准确率 %，种子 0；自由：注入电流为自由复数；约束：过渡电阻为实数加实增益；无增益：约束但增益固定为 1）。除物理模型外其余设置相同",
            "Controlled ablation of the physics model (IEEE 34, class accuracy %, seed 0; free: free complex injection current; con.: real fault resistance plus real gain; no gain: constrained with the gain fixed at 1). Everything except the physics model is identical", hdr, rows)


def table_swap(d):
    t = d.t
    scs = ("域内", "高阻外推", "噪声×3")
    hdr = [t("分类器", "Classifier")] + [f"{sc(d, s)}：{t('自身 → +本方法流程', 'alone → +our procedure')}" if d.L == "zh" else f"{sc(d, s)}: alone → +our procedure" for s in scs] + [t("+本方法流程高于+物理定位的场景数（核心 5 个中）", "Scenarios where +our procedure beats +physics location (of 5 core)")]
    rows = []
    for k in ("gcn_p", "gcn_d", "gat_p", "gat_d", "svm", "rf"):
        r = [mname(d, k)]
        for s in scs:
            w = swap(k, s)
            r.append("-" if not w else f"{w[NN]['class_acc'] * 100:.1f} → {w[K]['class_acc'] * 100:.1f}")
        n_better = sum(1 for s in CORE5 if swap(k, s) and swap(k, s)[K]["class_acc"] > swap(k, s)[NN]["class_acc"])
        r.append(str(n_better))
        rows.append(r)
    d.table(6, "各分类器作为阶段一接入本方法流程（IEEE 34，种子 0；类准确率 %）", "Each classifier plugged in as stage 1 of our procedure (IEEE 34, seed 0; class accuracy %)", hdr, rows)


def table_liang(d):
    t, c = d.t, load()
    ev = c["ev34"]
    hdr = [t("场景", "Scenario"), t("本方法", "Ours"), t("Liang 两阶段", "Liang two-stage"), "GCN", "PCA+SVM"]
    rows = []
    for s in FIXED4:
        la = liang_best(ev, s)
        A_s = c["acc"]["34"][s]
        kg, vg = best_in(A_s, "gcn")
        vals = [A_s["ours"], (la[0], 0.0, 1), vg, A_s["svm"]]
        cells = bold_best(vals)
        if vg:
            cells[2] += tag_of(kg)
        rows.append([sc(d, s)] + cells)
    d.table(7, "档位固定场景下与 Liang 两阶段方法的比较（IEEE 34，类准确率 %；Liang 取四个变体中最好者；GCN 取较高的输入，P：论文输入，D：突变量输入）",
            "Comparison with the two-stage method of Liang et al. in the fixed-tap scenarios (IEEE 34, class accuracy %; Liang is the best of its four variants; GCN with the higher input, P: paper input, D: disturbance input)", hdr, rows)
    errs = "; ".join(f"{sc(d, s)}: {t('本方法', 'ours')} {pe(s, 'ours'):.2f} km, Liang {pe(s, liang_best(ev, s)[2]):.2f} km" for s in FIXED4)
    d.w(t("测距误差（均值）：", "Location error (mean): ") + errs + ".\n")


def table_diag(d):
    t = d.t
    o = J(R34 / "oracle_line_params.json")
    hdr = [t("指标", "Metric"), t("名义阻抗模型（本方法）", "Nominal-impedance model (ours)"), t("逐线真值模型", "True per-line model")]
    rows = [[t("联合决策类准确率 (%)", "Joint-decision class accuracy (%)"), f"{o['class_acc']['joint_nominal']:.1f}", f"**{o['class_acc']['joint_oracle']:.1f}**"],
            [t("区段已知时测距误差均值 (km)", "Location error, section known, mean (km)"), f"{o['distance_err_km_true_section']['mean_nominal']:.2f}", f"**{o['distance_err_km_true_section']['mean_oracle']:.2f}**"],
            [t("区段已知时测距误差中位数 (km)", "Location error, section known, median (km)"), f"{o['distance_err_km_true_section']['median_nominal']:.2f}", f"**{o['distance_err_km_true_section']['median_oracle']:.2f}**"]]
    d.table(8, f"个别线路阻抗 ±20% 下的逐线阻抗真值诊断（IEEE 34，前 {o['n']} 个样本；神经网络 Top-1 为 {o['class_acc']['stage1_top1']:.1f}%）",
            f"Diagnosis with the true per-line impedances under individual line impedance ±20% (IEEE 34, first {o['n']} samples; neural-network Top-1 is {o['class_acc']['stage1_top1']:.1f}%)", hdr, rows)


def table_123(d):
    t, c = d.t, load()
    rows = []
    for s in CORE5:
        items, _ = row_cells(c["acc"]["123"][s])
        rows.append([sc(d, s)] + fmt_row(items))
    d.table(9, "IEEE 123 上的对比（类准确率 %；GCN 3 个种子（个别线路阻抗 ±20% 为 2 个），GAT 只运行突变量输入，块稀疏贝叶斯每个场景取 1000 个样本，其余为种子 0）；每行最优加粗",
            "Comparison on IEEE 123 (class accuracy %; GCN 3 seeds (2 for the individual-line scenario), GAT with the disturbance input only, simplified block-sparse Bayesian on 1000 samples per scenario, the rest seed 0); the best in each row is in bold", [t("场景", "Scenario")] + fam_hdr(d), rows)


# ---------------------------------------------------------------- 附录中的大表
def appendix_full(d):
    """附录表 A7：主对比的完整版（两种输入、全部场景、测距误差）。"""
    t, c = d.t, load()
    hdr = [t("场景", "Scenario")] + [mname(d, k) for k, _, _ in METHODS]
    rows = []
    for s in MAIN7:
        rows.append([sc(d, s)] + bold_best([c["acc"]["34"][s].get(k) for k, _, _ in METHODS]))
    d.table("A7a", "主对比的完整版：类准确率（%，均值±标准差；IEEE 34）", "Full version of the main comparison: class accuracy (%, mean ± standard deviation; IEEE 34)", hdr, rows)
    rows = []
    for s in MAIN7:
        cell_ = [f"{pe(s, 'ours'):.2f}"]
        for k, _, _ in METHODS[1:]:
            cell_.append(f"{pe(s, 'bsbl'):.2f}" if k == "bsbl" else f"{pe(s, k + '_mid'):.2f} / {pe(s, k + '_phys'):.2f}")
        rows.append([sc(d, s)] + cell_)
    d.table("A7b", "测距误差均值（km，沿馈线最短路径距离，种子 0）：分类器为“区段中点 / 接物理定位”（IEEE 34）", "Mean location error (km, shortest path along the feeder, seed 0): classifiers as “section midpoint / followed by physics-based location” (IEEE 34)", hdr, rows)


def appendix_extra(d):
    """附录表 A5：核心场景之外的场景，各方法的类准确率。"""
    t, c = d.t, load()
    hdr = [t("场景", "Scenario")] + [mname(d, k) for k, _, _ in METHODS]
    rows = [[sc(d, s)] + bold_best([c["acc"]["34"][s].get(k) for k, _, _ in METHODS]) for s in EXTRA7]
    d.table("A5", "核心场景之外的场景（IEEE 34，类准确率 %）", "Scenarios outside the core set (IEEE 34, class accuracy %)", hdr, rows)


def appendix_comp(d):
    """附录表 A8：神经网络 Top-1 与全区段物理穷举的对错互补。"""
    t = d.t
    bd = J(R34 / "breakdown.json")["互补性"]
    hdr = [t("场景", "Scenario"), t("两者都对", "Both right"), t("仅神经网络对", "NN only right"), t("仅物理对", "Physics only right"), t("两者都错", "Both wrong"), t("神经网络错时物理对 (%)", "Physics right when NN wrong (%)"), t("物理错时神经网络对 (%)", "NN right when physics wrong (%)")]
    rows = []
    for s in ("域内", "负荷偏移", "高阻外推", "噪声×3", "个别线路阻抗±20%", "固定档位 负荷偏移", "固定档位 高阻外推"):
        x = bd[s]
        rows.append([sc(d, s)] + [f"{x[k] * 100:.1f}" for k in ("两者都对", "仅神经网络对", "仅物理对", "两者都错")] + [f"{x['神经网络错时物理对'] * 100:.0f}", f"{x['物理错时神经网络对'] * 100:.0f}"])
    d.table("A8", "神经网络 Top-1 与全区段物理穷举的对错互补（占全部样本的百分比，IEEE 34，种子 0）", "Complementarity of the neural-network Top-1 and the exhaustive physics-based inversion (percent of all samples, IEEE 34, seed 0)", hdr, rows)


def appendix_timing(d):
    t, c = d.t, load()
    t34, t123 = c["ev34"]["耗时_ms每样本"], c["ev123"]["耗时_ms每样本"]
    k34 = [k for k in t34 if k.startswith("阶段二")][0]
    k123 = [k for k in t123 if k.startswith("阶段二")][0]
    hdr = [t("系统", "System"), t("阶段一 (ms/样本)", "Stage 1 (ms/sample)"), t("阶段二，Top-2 候选 (ms/样本)", "Stage 2, Top-2 candidates (ms/sample)"), t("全区段物理穷举 (ms/样本)", "Exhaustive physics over all sections (ms/sample)")]
    rows = [["IEEE 34", f"{t34['阶段一']:.3f}", f"{t34[k34]:.1f}", f"{t34['物理穷举(全部区段)']:.0f}"], ["IEEE 123", f"{t123['阶段一']:.3f}", f"{t123[k123]:.1f}", f"{t123['物理穷举(全部区段)']:.0f}"]]
    d.table("A9", "推断耗时（批量 1000 个样本摊销到每个样本，4 核 CPU 单进程，与其他任务并行时测得，偏保守；不是单个事件的端到端延迟，不含数据采集与通信）", "Inference time (batch of 1000 samples amortized per sample, one process on a 4-core CPU, measured while other jobs were running and therefore conservative; not the end-to-end latency of a single event and excluding data acquisition and communication)", hdr, rows)
    return t34[k34], t34["物理穷举(全部区段)"], t123[k123], t123["物理穷举(全部区段)"]


def appendix(d):
    d.w(d.t("## 附录\n", "## Appendix\n"))
    appendix_hyper(d)
    appendix_exact(d)
    appendix_full(d)
    appendix_extra(d)
    appendix_comp(d)
    appendix_design(d)
    appendix_timing(d)
    table_sensors(d)
    d.w(d.t("### 对 Liang 两阶段方法的复现验证\n", "### Validation of the reimplementation of the Liang et al. method\n"))
    appendix_liang(d)
    appendix_baselines(d)


# ================================================================ 结果章节正文
def results(d):
    t, c = d.t, load()
    A = c["acc"]["34"]
    ev = c["ev34"]
    w = d.w
    lst = lambda ss: "、".join(sc(d, s) for s in ss) if d.L == "zh" else ", ".join(sc(d, s) for s in ss)
    w(t("## 5 结果\n", "## 5 Results\n"))
    w(t("本节按问题组织：每个小节只回答一个问题，类准确率为主指标，测距误差单独成节。核心场景为 5 个（域内、负荷偏移、高阻外推、噪声 ×3、个别线路阻抗 ±20%），其余场景在附录。",
        "This section is organized by question: each subsection answers one question, class accuracy is the main metric and location error has its own subsection. The core scenarios are five (in-domain, load shift, high resistance, noise ×3, individual line impedance ±20%); the other scenarios are in the appendix.\n"))
    # ---- 5.1
    w(t("### 5.1 与已发表方法和基线相比，本方法在哪些条件下更好？\n", "### 5.1 Compared with published methods and baselines, under which conditions is our method better?\n"))
    table_main(d)
    pt = table_paired(d)
    vd = {s: verdict(A, s) for s in CORE5}
    win = [s for s in CORE5 if vd[s][0] == "win"]
    tie = [s for s in CORE5 if vd[s][0] == "tie"]
    lose = [s for s in CORE5 if vd[s][0] == "lose"]
    hr = A["高阻外推"]
    kb = vd["高阻外推"][2]
    w(t(f"与每个场景中均值最高的其余方法相比（均值差超过较大标准差的 2 倍才算显著），本方法在 {len(win)} 个场景显著更高（{lst(win)}），{len(tie)} 个无显著差别（{lst(tie)}），{len(lose)} 个显著更低（{lst(lose)}）。"
        f"差距最大的是过渡电阻超出训练范围（300–1000 Ω）：本方法 {pct(hr['ours'][0])}%，其余方法中最高的{mname(d, kb)}为 {pct(hr[kb][0])}%。同分布内本方法为 {pct(A['域内']['ours'][0])}%，与最强基线（{mname(d, vd['域内'][2])}，{pct(A['域内'][vd['域内'][2]][0])}%）相差 {abs(vd['域内'][1]):.1f} 个百分点。",
        f"Against the best of the other methods in each scenario (a difference counts as significant only if it exceeds twice the larger standard deviation), our method is significantly higher in {len(win)} scenario(s) ({lst(win)}), not significantly different in {len(tie)} ({lst(tie)}) and significantly lower in {len(lose)} ({lst(lose)}). "
        f"The gap is largest when the fault resistance leaves the training range (300–1000 Ω): our method reaches {pct(hr['ours'][0])}%, the best of the other methods, {mname(d, kb)}, {pct(hr[kb][0])}%. In-domain, ours is {pct(A['域内']['ours'][0])}%, {abs(vd['域内'][1]):.1f} percentage points {'above' if vd['域内'][1] > 0 else 'below'} the strongest baseline ({mname(d, vd['域内'][2])}, {pct(A['域内'][vd['域内'][2]][0])}%).") + "\n")
    def pt_group(key):
        up = [s for s in CORE5 if pt[s][key]["p"] < 0.05 and pt[s][key]["diff"] > 0]
        dn = [s for s in CORE5 if pt[s][key]["p"] < 0.05 and pt[s][key]["diff"] < 0]
        return up, dn, [s for s in CORE5 if s not in up + dn]
    cu, cd, cn = pt_group("class_test")
    eu, ed, en_ = pt_group("exact_test")
    cat = lambda up, dn, s: "win" if s in up else "lose" if s in dn else "tie"
    differ = [s for s in CORE5 if cat(cu, cd, s) != vd[s][0]]
    dif_txt = "、".join(sc(d, s) for s in differ) if d.L == "zh" else ", ".join(sc(d, s) for s in differ)
    dtl = lambda ss, key: "、".join(f"{sc(d, s)} {pt[s][key]['diff']:+.1f}" for s in ss) if d.L == "zh" else ", ".join(f"{sc(d, s)} {pt[s][key]['diff']:+.1f}" for s in ss)
    w(t(f"表 2b 在种子 0 的同一批测试样本上做配对检验（对方为各场景均值最高者）。类准确率上，本方法显著更高的有 {len(cu)} 个场景（{dtl(cu, 'class_test') or '无'} 个百分点），显著更低的有 {len(cd)} 个（{dtl(cd, 'class_test') or '无'}），其余 {len(cn)} 个差异不显著；"
        f"精确区段准确率上，显著更高的有 {len(eu)} 个场景（{dtl(eu, 'exact_test') or '无'}），显著更低的有 {len(ed)} 个（{dtl(ed, 'exact_test') or '无'}）。"
        "配对检验只含测试样本的随机性，不含训练随机性，而按种子间标准差的判断只含训练随机性，两者在" + (f"{dif_txt}上的结论不同；这些场景一律按本方法不优于对方处理。" if differ else "各场景的结论一致。"),
        f"Table 2b applies paired tests on the same seed-0 test samples (the other method is the best one of each scenario). For class accuracy, our method is significantly higher in {len(cu)} scenario(s) ({dtl(cu, 'class_test') or 'none'} percentage points), significantly lower in {len(cd)} ({dtl(cd, 'class_test') or 'none'}) and not significantly different in the other {len(cn)}; "
        f"for exact section accuracy it is significantly higher in {len(eu)} ({dtl(eu, 'exact_test') or 'none'}) and significantly lower in {len(ed)} ({dtl(ed, 'exact_test') or 'none'}). "
        "A paired test contains only the randomness of the test samples, while the judgement by the standard deviation across seeds contains only the randomness of training; the two " + (f"disagree in {dif_txt}, and these scenarios are all treated as our method not being better than the other." if differ else "agree in every scenario.")) + "\n")
    zl = [(s, verdict(A, s)) for s in ("线路阻抗+10%", "线路阻抗-10%")]
    gaps = [-v[1] for s, v in zl if v[0] == "lose"]
    w(t(f"线路阻抗存在偏差时，GCN 等基线不低于本方法：个别线路阻抗 ±20% 时论文输入的 GCN 比本方法高 {-vd['个别线路阻抗±20%'][1]:.1f} 个百分点" + (f"；全网线路阻抗整体偏差 ±10% 时（附录表 A5）差距为 {min(gaps):.1f}–{max(gaps):.1f} 个百分点" if gaps else "") +
        "。原因是这类偏差使阶段二的名义响应库失配，区段判别主要依赖阶段一，而基线与阶段一使用相同的训练数据；5.7 节进一步诊断了损失的来源。",
        f"When the line impedances are in error, baselines such as GCN are not below our method: under individual line impedance ±20% the GCN with the paper input is {-vd['个别线路阻抗±20%'][1]:.1f} percentage points higher" + (f"; under global line-impedance errors of ±10% (Table A5) the gap is {min(gaps):.1f}–{max(gaps):.1f} points" if gaps else "") +
        ". This is because such errors make the nominal response library of stage 2 mismatched, so the section decision relies mainly on stage 1, and the baselines are trained on the same data as stage 1; Section 5.7 diagnoses the source of the loss.") + "\n")
    pick = table_breakdown(d)
    t1 = J(R34 / "tier1.json")
    gs = ["故障类型 SLG", "故障类型 LL", "故障类型 LLG", "故障类型 LLLG", "位置 近端（α<0.35）", "位置 中段（0.35≤α<0.65）", "位置 远端（α≥0.65）"]
    best_cnt = sum(1 for g in gs if max(["本方案", pick["GCN"], pick["GAT"], "块稀疏贝叶斯学习", "PCA+SVM", "PCA+随机森林"], key=lambda k: t1[g][k]["class_acc"]) == "本方案")
    ft = [t1[g]["本方案"]["class_acc"] for g in gs[:4]]
    ps = [t1[g]["本方案"]["class_acc"] for g in gs[4:]]
    w(t(f"表 3 给出同分布内的分档：各类故障的类准确率在 {min(ft) * 100:.1f}%–{max(ft) * 100:.1f}% 之间，近端、中段、远端在 {min(ps) * 100:.1f}%–{max(ps) * 100:.1f}% 之间，方法对故障类型与区段内位置都没有明显的短板；本方法在 {len(gs)} 个分档中的 {best_cnt} 个最高。",
        f"Table 3 gives the breakdown within the distribution: the class accuracy of the individual fault types lies between {min(ft) * 100:.1f}% and {max(ft) * 100:.1f}%, and that of the near end, middle and far end between {min(ps) * 100:.1f}% and {max(ps) * 100:.1f}%, so the method has no obvious weakness with respect to fault type or position within a section; our method is the best in {best_cnt} of the {len(gs)} bins.") + "\n")
    # ---- 5.2
    w(t("### 5.2 过渡电阻与量测噪声变化时，准确率如何变化？\n", "### 5.2 How does the accuracy change with fault resistance and measurement noise?\n"))
    d.fig(3)
    bdc = J(R34 / "breakdown.json")
    cv = bdc["过渡电阻曲线"]
    bins = list(cv)
    w(t(f"图 3(a) 是本文方法动机的直接证据：神经网络的类准确率随过渡电阻升高而下降（{cv[bins[0]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[3]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[4]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[5]]['硬串联']['class_acc'] * 100:.1f}%，依次为 0–1 Ω、100–300 Ω、300–500 Ω、500–1000 Ω），物理穷举基本不随过渡电阻变化（{min(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%–{max(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%）但在低阻时低于神经网络；两条曲线在 300–500 Ω 附近交叉，联合决策在该区间（{cv[bins[4]]['本方案']['class_acc'] * 100:.1f}%）高于两者。500–1000 Ω 时联合决策（{cv[bins[5]]['本方案']['class_acc'] * 100:.1f}%）仍低于物理穷举（{cv[bins[5]]['物理穷举']['class_acc'] * 100:.1f}%），原因是 Top-2 候选漏掉了真实区段。",
        f"Fig. 3(a) is direct evidence for the motivation of the method: the class accuracy of the neural network falls as the fault resistance rises ({cv[bins[0]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[3]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[4]]['硬串联']['class_acc'] * 100:.1f}% → {cv[bins[5]]['硬串联']['class_acc'] * 100:.1f}% for 0–1 Ω, 100–300 Ω, 300–500 Ω and 500–1000 Ω), while the exhaustive physics-based inversion hardly changes with fault resistance ({min(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%–{max(cv[b]['物理穷举']['class_acc'] for b in bins) * 100:.1f}%) but is below the neural network at low resistance. The curves cross near 300–500 Ω, and the joint decision is above both there ({cv[bins[4]]['本方案']['class_acc'] * 100:.1f}%). At 500–1000 Ω the joint decision ({cv[bins[5]]['本方案']['class_acc'] * 100:.1f}%) is still below the exhaustive physics-based inversion ({cv[bins[5]]['物理穷举']['class_acc'] * 100:.1f}%) because the Top-2 candidates miss the true section.") + "\n")
    others = [k for k in t1["噪声 ×1"] if k not in ("本方案", "DNN Top-1")]
    nz = {}
    for lv in (0.5, 1, 2, 3, 5):
        g = t1[f"噪声 ×{lv:g}"]
        kb2 = max(others, key=lambda k: g[k]["class_acc"])
        nz[lv] = (g["本方案"]["class_acc"] * 100, g[kb2]["class_acc"] * 100, kb2)
    behind = [lv for lv in nz if nz[lv][0] < nz[lv][1] - 0.05]
    nm_ = lambda k: {"GCN 论文输入": "GCN（论文输入）", "GCN 突变量输入": "GCN（突变量输入）", "GAT 论文输入": "GAT（论文输入）", "GAT 突变量输入": "GAT（突变量输入）", "PCA+SVM": "PCA+SVM", "PCA+随机森林": "PCA+随机森林", "块稀疏贝叶斯学习": "简化块稀疏贝叶斯"}[k] if d.L == "zh" else {"GCN 论文输入": "GCN (paper input)", "GCN 突变量输入": "GCN (disturbance input)", "GAT 论文输入": "GAT (paper input)", "GAT 突变量输入": "GAT (disturbance input)", "PCA+SVM": "PCA+SVM", "PCA+随机森林": "PCA+RF", "块稀疏贝叶斯学习": "simplified block-sparse Bayesian"}[k]
    ahead_txt = "、".join(f"×{lv:g}" for lv in nz if lv not in behind)
    behind_txt = "；".join(f"×{lv:g} 时{nm_(nz[lv][2])}高 {nz[lv][1] - nz[lv][0]:.1f} 个百分点" for lv in behind)
    behind_en = "; ".join(f"at ×{lv:g} {nm_(nz[lv][2])} is {nz[lv][1] - nz[lv][0]:.1f} points higher" for lv in behind)
    w(t(f"图 3(b) 为噪声敏感性：噪声倍数 {ahead_txt} 时本方法是各方法中最高或并列最高" + (f"；{behind_txt}。" if behind else "。") + f"神经网络加物理定位的硬串联在噪声升高时下降更快（×5 时 {t1['噪声 ×5']['DNN Top-1']['class_acc'] * 100:.1f}%，本方法 {t1['噪声 ×5']['本方案']['class_acc'] * 100:.1f}%）。",
        f"Fig. 3(b) shows the sensitivity to measurement noise: at noise multipliers {', '.join(f'×{lv:g}' for lv in nz if lv not in behind)} our method is the highest or tied for the highest" + (f"; {behind_en}. " if behind else ". ") + f"The hard cascade of the neural network and physics-based location degrades faster as noise grows ({t1['噪声 ×5']['DNN Top-1']['class_acc'] * 100:.1f}% at ×5 against {t1['噪声 ×5']['本方案']['class_acc'] * 100:.1f}% for our method).") + "\n")
    sensors_section(d)
    # ---- 5.4
    w(t("### 5.4 预测的故障点离真实故障点有多远？\n", "### 5.4 How far is the predicted fault point from the true one?\n"))
    d.fig(5)
    table_loc(d)
    mid_in = [pe("域内", f"{k}_mid") for k in TAG]
    ph_in = (min(pe("域内", f"{k}_phys") for k in TAG), max(pe("域内", f"{k}_phys") for k in TAG))
    ph_hr = (min(pe("高阻外推", f"{k}_phys") for k in TAG), max(pe("高阻外推", f"{k}_phys") for k in TAG))
    w(t(f"分类器基线只输出区段，故障点取预测区段中点，测距误差因此明显偏大（域内 {min(mid_in):.2f}–{max(mid_in):.2f} km）；把它们的 Top-1 区段交给阶段二估计区段内位置后，域内降到 {ph_in[0]:.2f}–{ph_in[1]:.2f} km，与本方法（{pe('域内', 'ours'):.2f} km）接近，说明区段内测距精度主要来自物理反演。"
        f"高阻外推下，各分类器接物理定位后仍有 {ph_hr[0]:.2f}–{ph_hr[1]:.2f} km，本方法为 {pe('高阻外推', 'ours'):.2f} km，差别来自区段选错（见 5.5 节）。块稀疏贝叶斯自带位置估计，但位置受 α 网格限制，且误判区段时误差很大（图 5、表 4）。",
        f"The classifier baselines output only the section and the fault point is taken at the section midpoint, so their location error is much larger ({min(mid_in):.2f}–{max(mid_in):.2f} km in-domain); once their Top-1 section is handed to stage 2 to estimate the position within the section, the in-domain error falls to {ph_in[0]:.2f}–{ph_in[1]:.2f} km, close to our method ({pe('域内', 'ours'):.2f} km), which shows that the accuracy of the position within a section comes mainly from the physics-based inversion. "
        f"Under high resistance the classifiers followed by physics-based location still have {ph_hr[0]:.2f}–{ph_hr[1]:.2f} km against {pe('高阻外推', 'ours'):.2f} km for our method, the difference coming from wrongly selected sections (Section 5.5). Block-sparse Bayesian learning estimates the position itself, but it is limited by the α grid and its error is large when it selects the wrong section (Fig. 5, Table 4).") + "\n")
    # ---- 5.5
    w(t("### 5.5 各个组成部分各自起了什么作用？\n", "### 5.5 What does each component contribute?\n"))
    table_ablation(d)
    fr, co, ng = J(R34 / "ablation_aug_only/evaluation.json"), ev, J(R34 / "ablation_fixed_gain/evaluation.json")
    allsc = [s for s in fr if s not in ("thresholds", "equivalence_classes", "k", "config", "耗时_ms每样本") and s in co]
    gain = [(co[s][PH]["class_acc"] - fr[s][PH]["class_acc"]) * 100 for s in allsc]
    jn = sum(1 for s in allsc if co[s][K]["class_acc"] > fr[s][K]["class_acc"])
    fx = [s for s in allsc if s.startswith("固定档位") or s == "高阻外推"]
    gg = [(co[s][PH]["class_acc"] - ng[s][PH]["class_acc"]) * 100 for s in fx if s in ng]
    w(t(f"表 5 是物理模型的受控消融。把过渡电阻限定为实数后，物理穷举的类准确率在全部 {len(allsc)} 个评测场景（含附录场景）都提高，提高 {min(gain):.1f}–{max(gain):.1f} 个百分点；联合决策在 {jn} 个场景提高。实增益在故障前电压偏离名义值较多的场景作用明显：高阻外推与固定档位场景下，去掉增益使物理穷举降低 {min(gg):.1f}–{max(gg):.1f} 个百分点；"
        "增益对联合决策类准确率的影响不一致，收益主要体现在物理穷举与测距误差上。线路阻抗整体偏差 ±10% 的场景下，约束对联合决策没有增益。",
        f"Table 5 is a controlled ablation of the physics model. Restricting the fault resistance to a real number raises the class accuracy of the exhaustive physics-based inversion in all {len(allsc)} evaluated scenarios (including those in the appendix), by {min(gain):.1f}–{max(gain):.1f} percentage points; the joint decision improves in {jn} scenarios. The real gain matters where the pre-fault voltage deviates from the nominal value: in the high-resistance and fixed-tap scenarios, removing the gain lowers the exhaustive inversion by {min(gg):.1f}–{max(gg):.1f} percentage points; "
        "its effect on the class accuracy of the joint decision is inconsistent, and its benefit shows mainly in the exhaustive inversion and in the location error. Under global line-impedance errors of ±10% the constraint brings no gain to the joint decision.") + "\n")
    table_swap(d)
    tally = {}
    for s_ in CORE5:
        cands = {"ours": ev[s_][K]["class_acc"]}
        cands.update({k: swap(k, s_)[K]["class_acc"] for k in TAG if swap(k, s_)})
        top = max(cands.values())
        for k, v in cands.items():
            if v >= top - 1e-9:
                tally[k] = tally.get(k, 0) + 1
    cnt = {k: sum(1 for s in CORE5 if swap(k, s) and swap(k, s)[K]["class_acc"] > swap(k, s)[NN]["class_acc"]) for k in TAG}
    cp = bdc["互补性"]
    tl = "、".join(f"{mname(d, k) if k != 'ours' else '本方法的默认 MLP'}（{n} 个）" for k, n in sorted(tally.items(), key=lambda x: -x[1]))
    tl_en = ", ".join(f"{mname(d, k) if k != 'ours' else 'the default MLP of our method'} ({n})" for k, n in sorted(tally.items(), key=lambda x: -x[1]))
    w(t(f"表 6 把 GCN、GAT、PCA+SVM、PCA+随机森林的概率作为阶段一接入本方法流程：联合打分与候选扩展相对“分类器 + 物理定位”在 5 个核心场景中分别有 {cnt['gcn_p']}、{cnt['gcn_d']}、{cnt['gat_p']}、{cnt['gat_d']}、{cnt['svm']}、{cnt['rf']} 个提高（依次为 GCN 论文输入、GCN 突变量输入、GAT 论文输入、GAT 突变量输入、PCA+SVM、PCA+随机森林），说明流程的增益不依赖具体的阶段一网络。"
        f"在 5 个核心场景中，用本方法流程时准确率最高的阶段一依次为：{tl}；默认的轻量 MLP 并不是每个场景下最强的，换用更强的阶段一，流程仍然适用。",
        f"Table 6 plugs the probabilities of GCN, GAT, PCA+SVM and PCA+RF in as stage 1 of our procedure: the joint score and candidate expansion improve on “classifier + physics location” in {cnt['gcn_p']}, {cnt['gcn_d']}, {cnt['gat_p']}, {cnt['gat_d']}, {cnt['svm']} and {cnt['rf']} of the 5 core scenarios (GCN paper input, GCN disturbance input, GAT paper input, GAT disturbance input, PCA+SVM, PCA+RF), so the gain of the procedure does not depend on a specific stage-1 network. "
        f"Among the 5 core scenarios, the stage 1 that gives the highest accuracy with our procedure is, in order: {tl_en}; the default lightweight MLP is not the strongest in every scenario, and a stronger stage 1 can be swapped in while the procedure still applies.") + "\n")
    alm = J(R34 / "alarm.json")
    au_lo, au_hi = min(alm[s_]["置信度AUROC"] for s_ in CORE5), max(alm[s_]["置信度AUROC"] for s_ in CORE5)
    rcs = sorted(alm[s_]["告警召回率"] * 100 for s_ in CORE5)
    rc_hi, rc_2nd = rcs[-1], rcs[-2]
    er = [alm[s_]["告警样本错误率"] / (1 - alm[s_]["准确率"]) for s_ in CORE5]
    a3 = J(R123 / "alarm.json")
    w(t(f"联合决策的前提是两路错误不重合（附录表 A8）：神经网络错时物理穷举对的比例，域内为 {cp['域内']['神经网络错时物理对'] * 100:.0f}%，高阻外推为 {cp['高阻外推']['神经网络错时物理对'] * 100:.0f}%；物理穷举错时神经网络对的比例，域内为 {cp['域内']['物理错时神经网络对'] * 100:.0f}%，高阻外推为 {cp['高阻外推']['物理错时神经网络对'] * 100:.0f}%。"
        f"这不单独证明神经网络下降的原因是特征越出训练范围，间接依据是 GCN 论文输入（故障期间的绝对相量）在高阻外推下下降更多（表 2）。候选扩展只在高阻外推（+2.3 个百分点）与噪声 ×3（+0.9）下有明显作用，置信度与告警的统计见附录表 A4：置信度能在一定程度上区分对错（AUROC 为 {au_lo:.2f}–{au_hi:.2f}，高阻外推下最低），失配告警则几乎不能挑出错误样本（召回率最高 {rc_hi:.1f}%，其余场景不超过 {rc_2nd:.1f}%；告警样本的错误率为整体错误率的 {min(er):.1f}–{max(er):.1f} 倍），它只表示物理模型解释不了量测，不能当作错误预测的检测器（IEEE 34）。IEEE 123 上告警在噪声 ×3 下触发 {a3['噪声×3']['告警占比'] * 100:.0f}%、召回 {a3['噪声×3']['告警召回率'] * 100:.0f}%（残差整体超过标定阈值），在高阻外推下几乎不触发（{a3['高阻外推']['告警占比'] * 100:.1f}%，而准确率只有 {a3['高阻外推']['准确率'] * 100:.1f}%），说明告警反映的是残差是否超过标定阈值，而不是区段判断是否出错。",
        f"The premise of the joint decision is that the errors of the two do not coincide (Table A8): when the neural network is wrong, the exhaustive physics-based inversion is right in {cp['域内']['神经网络错时物理对'] * 100:.0f}% (in-domain) and {cp['高阻外推']['神经网络错时物理对'] * 100:.0f}% (high resistance) of the cases; when the physics-based inversion is wrong, the neural network is right in {cp['域内']['物理错时神经网络对'] * 100:.0f}% and {cp['高阻外推']['物理错时神经网络对'] * 100:.0f}%. "
        f"This does not by itself prove that the decline of the neural network is caused by features leaving the training range; the indirect evidence is that the GCN with the paper input (absolute during-fault phasors) declines more under high resistance (Table 2). Candidate expansion matters clearly only under high resistance (+2.3 percentage points) and noise ×3 (+0.9); statistics on confidence and alarm are in Table A4: the confidence separates correct from wrong decisions to some extent (AUROC {au_lo:.2f}–{au_hi:.2f}, lowest under high resistance), whereas the mismatch alarm hardly picks out the wrong decisions (recall at most {rc_hi:.1f}% and below {rc_2nd:.1f}% in the other scenarios, and the error rate among alarmed samples is {min(er):.1f}–{max(er):.1f} times the overall error rate); it only indicates that the physics model cannot explain the measurements and should not be used as a detector of wrong predictions (IEEE 34). On IEEE 123 the alarm fires for {a3['噪声×3']['告警占比'] * 100:.0f}% of the samples under noise ×3 with a recall of {a3['噪声×3']['告警召回率'] * 100:.0f}% (the residuals as a whole exceed the calibrated threshold) and hardly fires under high resistance ({a3['高阻外推']['告警占比'] * 100:.1f}%, while the accuracy is only {a3['高阻外推']['准确率'] * 100:.1f}%), so the alarm reflects whether the residual exceeds the calibrated threshold, not whether the section decision is wrong.") + "\n")
    # ---- 5.6
    w(t("### 5.6 与结构最接近的 Liang 两阶段方法相比如何？\n", "### 5.6 How does it compare with the structurally closest method of Liang et al.?\n"))
    table_liang(d)
    la = [(ev[s][K]["class_acc"] - liang_best(ev, s)[0]) * 100 for s in FIXED4]
    w(t(f"Liang 两阶段方法假设调压器档位与网络参数精确已知，并获得根母线相量、末端电压幅值和全网负荷伪量测，多于本方法的量测，因此只在档位固定的 4 个场景比较。本方法在这 4 个场景中高 {min(la):.1f}–{max(la):.1f} 个百分点，测距误差也更小。需要注意这是我们按论文复现的版本，第一阶段低于论文报告（附录表 A3），对原方法可能低估。",
        f"The method of Liang et al. assumes exactly known regulator taps and network parameters and receives root-bus phasors, end-bus voltage magnitudes and load pseudo-measurements, more than our method, so it is compared only in the four fixed-tap scenarios. Our method is {min(la):.1f}–{max(la):.1f} percentage points higher in these four scenarios and also has the smaller location error. Note that this is our reimplementation from the paper, whose stage 1 is below the reported values (Table A3), so the original method may be underestimated.") + "\n")
    # ---- 5.7
    w(t("### 5.7 线路参数不准时，损失来自哪里？\n", "### 5.7 When line parameters are inaccurate, where does the loss come from?\n"))
    table_diag(d)
    o = J(R34 / "oracle_line_params.json")
    w(t(f"每条线路阻抗独立偏差 ±20% 时，本方法的类准确率从域内的 {pct(A['域内']['ours'][0])}% 降到 {pct(A['个别线路阻抗±20%']['ours'][0])}%。表 8 的诊断把每个样本真实的逐线阻抗交给阶段二：区段已知时的测距误差从 {o['distance_err_km_true_section']['mean_nominal']:.2f} km 降到 {o['distance_err_km_true_section']['mean_oracle']:.2f} km，回到无偏差时的水平，说明位置误差的增加来自模型失配；联合决策的类准确率只从 {o['class_acc']['joint_nominal']:.1f}% 回升到 {o['class_acc']['joint_oracle']:.1f}%，说明区段判别的损失还有阶段一在逐线偏差下的失配，不能全部归于物理模型。"
        "以同样的偏差分布扩增阶段一训练集只提高 0.6 个百分点（IEEE 34）和 1.4 个百分点（IEEE 123）。各负荷功率独立变化对结果没有影响（附录表 A5）。",
        f"When each line impedance deviates independently by ±20%, the class accuracy of our method falls from {pct(A['域内']['ours'][0])}% in-domain to {pct(A['个别线路阻抗±20%']['ours'][0])}%. The diagnosis in Table 8 hands the true per-line impedances of each sample to stage 2: the location error with the section known falls from {o['distance_err_km_true_section']['mean_nominal']:.2f} km to {o['distance_err_km_true_section']['mean_oracle']:.2f} km, back to the error-free level, so the larger position error comes from model mismatch; the class accuracy of the joint decision recovers only from {o['class_acc']['joint_nominal']:.1f}% to {o['class_acc']['joint_oracle']:.1f}%, so part of the loss in the section decision is the mismatch of stage 1 under per-line errors and cannot all be attributed to the physics model. "
        "Augmenting the stage-1 training set with the same error distribution improves the accuracy by only 0.6 percentage points (IEEE 34) and 1.4 (IEEE 123). Independent variation of the load powers has no effect (Table A5).") + "\n")
    # ---- 5.8
    w(t("### 5.8 在 IEEE 123 节点上结论是否成立，计算量如何？\n", "### 5.8 Do the conclusions hold on the IEEE 123-bus system, and what is the computation cost?\n"))
    table_123(d)
    table_paired(d, "123")
    A3 = c["acc"]["123"]
    v3 = {s: verdict({s: A3[s]}, s) for s in CORE5 if s in A3 and len(A3[s]) > 1}
    win3 = [s for s in v3 if v3[s][0] == "win"]
    tie3 = [s for s in v3 if v3[s][0] == "tie"]
    lose3 = [s for s in v3 if v3[s][0] == "lose"]
    p3 = J(R123 / "paired_tests.json")
    ex_in = (p3["域内"]["ours_exact"] * 100, p3["域内"]["ours_class"] * 100)
    ex_gap = [p3[s_]["exact_test"]["diff"] for s_ in p3]
    cl_gap = [p3[s_]["class_test"]["diff"] for s_ in p3]
    sig3 = [s_ for s_ in p3 if p3[s_]["class_test"]["p"] < 0.05 and p3[s_]["class_test"]["diff"] > 0]
    w(t(f"表 9b 在种子 0 的同一批测试样本上做配对检验：对方是保存了种子 0 逐样本预测的方法中均值最高者（GCN 突变量输入，个别线路阻抗 ±20% 因 GCN 种子 0 没有运行该场景而省略）。类准确率上本方法在 {len(p3)} 个场景中的 {len(sig3)} 个显著更高（{min(cl_gap):+.1f} 到 {max(cl_gap):+.1f} 个百分点）；"
        f"精确区段准确率上高 {min(ex_gap):.1f}–{max(ex_gap):.1f} 个百分点。需要注意，IEEE 123 的 118 个区段在 8 个测点下只能分成 64 个等价类，精确区段准确率远低于类准确率（域内 {ex_in[0]:.1f}% 对 {ex_in[1]:.1f}%），类准确率不能当作对单个区段的定位能力。",
        f"Table 9b applies paired tests on the same seed-0 test samples; the other method is the best one among those with saved seed-0 per-sample predictions (GCN with the disturbance input; the individual-line-impedance ±20% scenario is omitted because GCN did not run it at seed 0). In class accuracy our method is significantly higher in {len(sig3)} of {len(p3)} scenarios ({min(cl_gap):+.1f} to {max(cl_gap):+.1f} percentage points); "
        f"in exact section accuracy it is higher by {min(ex_gap):.1f}–{max(ex_gap):.1f} percentage points. Note that the 118 sections of IEEE 123 fall into only 64 equivalence classes with 8 sensors, so the exact section accuracy is far below the class accuracy ({ex_in[0]:.1f}% against {ex_in[1]:.1f}% in-domain), and the class accuracy must not be read as the ability to locate an individual section.") + "\n")
    a34, e34, a123, e123 = appendix_timing(Doc_dummy(d))
    w(t(f"IEEE 123（118 个区段、8 个测点）上，本方法在 {len(win3)} 个核心场景显著更高（{lst(win3)}），{len(tie3)} 个无显著差别（{lst(tie3)}），{len(lose3)} 个显著更低（{lst(lose3)}）。高阻外推下所有方法都低，该场景的故障信号接近噪声。"
        f"计算量方面，阶段二只对 2 个候选做物理反演：IEEE 34 为 {a34:.1f} ms/样本，全区段物理穷举为 {e34:.0f} ms；IEEE 123 为 {a123:.1f} ms 与 {e123:.0f} ms（附录表 A9）。",
        f"On IEEE 123 (118 sections, 8 sensors), our method is significantly higher in {len(win3)} core scenarios ({lst(win3)}), not significantly different in {len(tie3)} ({lst(tie3)}) and significantly lower in {len(lose3)} ({lst(lose3)}). All methods are low under high resistance, where the fault signal is close to the noise. "
        f"In terms of computation, stage 2 inverts only two candidates: {a34:.1f} ms per sample on IEEE 34 against {e34:.0f} ms for the exhaustive inversion over all sections, and {a123:.1f} ms against {e123:.0f} ms on IEEE 123 (Table A9).") + "\n")


class Doc_dummy:
    """只为取耗时数字而调用 appendix_timing，不输出表。"""

    def __init__(self, d):
        self.L, self.out = d.L, []

    def t(self, zh, en):
        return zh if self.L == "zh" else en

    def table(self, *a, **k):
        pass


def abstract(d):
    t, c = d.t, load()
    h = headline(c)
    A3 = c["acc"]["123"]
    v3 = [verdict({s: A3[s]}, s)[0] for s in CORE5 if s in A3 and len(A3[s]) > 1]
    d.w(t("## 摘要\n", "## Abstract\n"))
    d.w(t(
        f"配电网只有极少数同步量测点时，现有故障定位方法各有不足：数据驱动方法在测试条件偏离训练分布时精度下降，物理反演不受训练分布影响，但在响应相近的区段之间区分度不足。"
        f"本文提出数据粗筛与物理闭环校正的两阶段方法：神经网络把故障区段缩小到两个候选，再用过渡电阻限定为实数的物理反演逐个核对并估计区段内位置，由概率—残差联合打分合并两者的证据；"
        f"候选都不能解释量测时扩大搜索范围，仍不能解释则发出失配告警。在 IEEE 34 节点（5 个测点、30 个区段）上，过渡电阻超出训练范围（300–1000 Ω）时本方法的区段准确率（按等价类判定）为 {pct(h['hr_ours'])}%，"
        f"对比的外部方法中最高的为 {pct(h['hr_best'])}%；同分布内与最强基线相当（本方法 {pct(h['dom_ours'])}%，基线最高 {pct(h['dom_best'])}%）；把过渡电阻限定为实数使物理反演的区段准确率在全部 {h['n_sc']} 个场景提高 {h['gain_lo']:.1f}–{h['gain_hi']:.1f} 个百分点；"
        f"在调压器档位固定的 4 个场景中比 Liang 等的两阶段方法高 {h['la_lo']:.1f}–{h['la_hi']:.1f} 个百分点。在 IEEE 123 节点（8 个测点、118 个区段）的 5 个核心场景中，本方法在 {sum(1 for x in v3 if x == 'win')} 个显著优于其余方法。"
        f"线路阻抗存在偏差时，部分基线的区段准确率更高；逐线阻抗真值诊断表明位置误差的增加来自线路参数失配。全部结果来自仿真。",
        f"With only a handful of synchronized sensors in a distribution network, existing fault-location methods each fall short: data-driven methods lose accuracy when test conditions leave the training distribution, and physics-based inversion, which is insensitive to the training distribution, cannot separate sections whose responses are similar. "
        f"We propose a two-stage method of data-driven screening and physics-based closed-loop verification: a neural network narrows the faulted section to two candidates, a physics-based inversion with a real-valued fault resistance verifies each candidate and estimates the position within the section, and a joint probability–residual score combines the two pieces of evidence; "
        f"when no candidate explains the measurements the search is widened, and if it still fails a mismatch alarm is raised. On the IEEE 34-bus feeder (5 sensors, 30 sections), when the fault resistance leaves the training range (300–1000 Ω) our method reaches a section accuracy (judged by equivalence class) of {pct(h['hr_ours'])}%, "
        f"against {pct(h['hr_best'])}% for the best external method compared; within the training distribution it is on par with the strongest baseline ({pct(h['dom_ours'])}% against at most {pct(h['dom_best'])}%); restricting the fault resistance to a real number raises the section accuracy of the physics-based inversion by {h['gain_lo']:.1f}–{h['gain_hi']:.1f} percentage points in all {h['n_sc']} scenarios; "
        f"and in the four fixed-tap scenarios it is {h['la_lo']:.1f}–{h['la_hi']:.1f} percentage points above the two-stage method of Liang et al. On the IEEE 123-bus feeder (8 sensors, 118 sections) it is significantly better than the other methods in {sum(1 for x in v3 if x == 'win')} of the 5 core scenarios. "
        f"Under line-impedance errors some baselines are more accurate; a diagnosis with the true per-line impedances shows that the larger position error comes from line-parameter mismatch. All results are from simulation.") + "\n")
    d.w(t("**关键词：** 故障定位；配电网；极稀疏量测；物理反演；数据驱动；分布外泛化\n", "**Keywords:** fault location; distribution network; extremely sparse measurements; physics-based inversion; data-driven; out-of-distribution generalization\n"))


def discussion(d):
    t = d.t
    d.w(t("## 6 讨论与局限\n", "## 6 Discussion and Limitations\n"))
    d.w(t(
        "**方法的适用范围。** 本方法的增益集中在数据驱动方法失效的条件，即过渡电阻等超出训练范围的外推；在同分布内与最强的数据驱动基线相当，在线路阻抗存在偏差的场景中部分基线更高。"
        "因此它适合用在量测很少、故障条件可能超出训练覆盖范围、而线路参数相对可靠的配电网；若线路参数本身不准，则需要先做参数辨识，或像逐线真值诊断那样提供更准确的参数。",
        "**Scope of the method.** The gain of our method is concentrated where data-driven methods fail, namely extrapolation such as fault resistance beyond the training range; within the training distribution it is on par with the strongest data-driven baseline, and under line-impedance errors some baselines are higher. "
        "It is therefore suited to networks with very few sensors, where fault conditions may exceed the training coverage and the line parameters are comparatively reliable; if the line parameters themselves are inaccurate, parameter identification is needed first, or more accurate parameters as in the diagnosis with true per-line impedances.\n"))
    d.w(t("**局限。** （1）全部结果来自 OpenDSS 仿真，量测噪声与同步误差按假设取值，没有实测故障数据，仿真到现场的差距只能由模型偏差场景间接刻画。"
          "（2）区段准确率按固定的等价类判定，报告的精确区段准确率比它低约 2.5–3.3 个百分点（附录表 A2），但各方法的排序基本不变。"
          "（3）基线是我们按原文描述的复现，可能与原作者的实现有差异：Liang 两阶段方法在其论文的 IEEE 33 节点系统上复现的第一阶段低于论文报告（附录表 A3），对它的比较只代表复现版本；块稀疏贝叶斯的字典取区段内位置而非原文的节点注入，且没有使用原文的块内相关矩阵，位置还受网格限制；SVM 只用了 3 万个训练子样本。"
          "（4）过渡电阻 500–1000 Ω 时联合决策仍低于物理穷举，因为 Top-2 候选漏掉真实区段；IEEE 123 上高阻故障的信号接近噪声，所有方法的准确率都低。"
          "（5）没有覆盖分布式电源接入、拓扑重构、数据缺失和小电流接地系统；测点数量的扫描只在 IEEE 34 上做了，测点集合是嵌套的贪心序列，没有比较不同布置。"
          "（6）神经网络下降的原因（特征越出训练范围）只有间接证据。"
          "（7）默认的轻量 MLP 阶段一在各场景下并不总是最强（表 6），本文没有针对阶段一做进一步的选择或集成。"
          "（8）失配告警只表示物理模型解释不了量测，召回率很低，不能当作错误预测的检测器；置信度对对错有一定区分能力，但高阻外推下偏弱（附录表 A4b）。",
          "**Limitations.** (1) All results come from OpenDSS simulation; measurement noise and synchronization errors are assumed, there are no field fault data, and the gap between simulation and the field is characterized only indirectly by the model-error scenarios. "
          "(2) Section accuracy is judged on fixed equivalence classes; the exact section accuracy is about 2.5–3.3 percentage points lower (Table A2), but the ranking of the methods is essentially unchanged. "
          "(3) The baselines are our reimplementations from the descriptions in the papers and may differ from the original authors' code: for the method of Liang et al. the stage 1 we reproduced on the IEEE 33-bus system of their paper is below the reported values (Table A3), so the comparison with it represents the reimplementation; our block-sparse Bayesian learning uses a dictionary of positions within sections rather than the node injections of the original paper, does not use its within-block correlation matrix, and its position is limited by a grid; the SVM uses only a 30,000-sample training subset. "
          "(4) At 500–1000 Ω the joint decision is still below the exhaustive physics-based inversion because the Top-2 candidates miss the true section; on IEEE 123 the signal of high-resistance faults is close to the noise and all methods are low. "
          "(5) Distributed generation, topology reconfiguration, missing data and resonant-grounded systems are not covered, and the sweep of the number of sensors was done only on IEEE 34 with a nested greedy sequence of sensors, without comparing different placements. "
          "(6) The cause of the decline of the neural network (features leaving the training range) is supported only indirectly. "
          "(7) The default lightweight MLP stage 1 is not always the strongest in every scenario (Table 6), and this paper does not select or ensemble stage-1 networks further. "
          "(8) The mismatch alarm only indicates that the physics model cannot explain the measurements; its recall is low and it should not be used as a detector of wrong predictions, while the confidence separates correct from wrong decisions to some extent but is weak under high resistance (Table A4b).\n"))


def conclusion(d):
    t, c = d.t, load()
    h = headline(c)
    d.w(t("## 7 结论\n", "## 7 Conclusion\n"))
    d.w(t(
        f"本文针对极稀疏量测下的配电网故障定位，提出神经网络粗筛、物理反演校正、概率—残差联合打分的两阶段方法，并加入候选扩展与失配告警。在 IEEE 34 与 IEEE 123 的仿真中，过渡电阻超出训练范围时本方法的区段准确率明显高于对比的数据驱动、稀疏恢复与两阶段方法"
        f"（IEEE 34 为 {pct(h['hr_ours'])}%，对比方法中最高 {pct(h['hr_best'])}%），同分布内与最强基线相当；区段内的测距精度来自物理反演，把过渡电阻限定为实数使物理反演的区段准确率在全部 {h['n_sc']} 个场景提高 {h['gain_lo']:.1f}–{h['gain_hi']:.1f} 个百分点。"
        f"方法的主要局限是线路阻抗存在偏差时部分基线更高，以及没有实测验证。后续工作包括线路参数辨识与方法的联合、拓扑变化与分布式电源、以及实测数据验证。",
        f"For fault location in distribution networks with extremely sparse measurements, this paper proposed a two-stage method of neural-network screening, physics-based verification and joint probability–residual scoring, with candidate expansion and a mismatch alarm. "
        f"In simulations on IEEE 34 and IEEE 123, when the fault resistance leaves the training range the section accuracy of our method is clearly higher than that of the compared data-driven, sparse-recovery and two-stage methods ({pct(h['hr_ours'])}% on IEEE 34 against at most {pct(h['hr_best'])}%), and within the training distribution it is on par with the strongest baseline; "
        f"the accuracy of the position within a section comes from the physics-based inversion, and restricting the fault resistance to a real number raises the section accuracy of the physics-based inversion by {h['gain_lo']:.1f}–{h['gain_hi']:.1f} percentage points in all {h['n_sc']} scenarios. "
        f"The main limitations are that some baselines are higher when line impedances are in error and that there is no field validation. Future work includes combining the method with line-parameter identification, topology changes and distributed generation, and validation with field data.\n"))
