"""初稿中不依赖实验数字的章节：引言、相关工作、方法、实验设置。d 为 Doc（含语言与引用编号），返回无。"""


def intro(d):
    t, c, w = d.t, d.c, d.w
    w(t("## 1 引言\n", "## 1 Introduction\n"))
    w(t(
        f"配电网区段多、分支多，故障后依靠巡线定位耗时长，缩短定位时间是缩短停电时间的关键环节{c('Stefanidou2022','Rezapour2023')}。"
        f"同步相量量测装置（μPMU 等）的成本使得实际能安装的测点远少于区段数，这就是“极稀疏量测”问题：IEEE 34 节点馈线只有 5 个测点覆盖 30 个区段，IEEE 123 节点只有 8 个测点覆盖 118 个区段。",
        f"Distribution feeders have many sections and branches, and locating a fault by patrol is slow; shortening location time is a key step in shortening outages {c('Stefanidou2022','Rezapour2023')}. "
        f"The cost of synchronized phasor devices (μPMUs) means that far fewer sensors can be installed than there are sections. We call this the extremely sparse measurement setting: "
        f"five sensors cover the 30 sections of the IEEE 34-bus feeder, and eight cover the 118 sections of the IEEE 123-bus feeder.\n"))
    w(t(
        f"现有方法大致分两类。物理方法利用网络模型，包括阻抗法{c('MoraFlorez2008','Chandran2024')}、电压凹陷与响应匹配{c('Buzo2020','Zhang2020')}、稀疏恢复与压缩感知{c('Jia2020','Jiang2021','Shan2024')}，"
        f"以及先筛选候选、再做非线性状态估计的两阶段方法{c('DLiang2025')}。它们不依赖训练数据，但测点很少时不同区段的响应相近，残差的区分度有限。"
        f"数据驱动方法用神经网络学习量测到故障位置的映射，包括支持向量机和集成学习{c('Thukaram2005','Ghaemi2022','Okumus2025')}、卷积网络{c('Siddique2024','Khattak2025')}和图神经网络{c('Chen2019','LLiang2024','Karabulut2026')}，"
        f"训练分布内精度很高，但测试条件偏离训练分布时精度会下降。在我们检索到的文献中，测试集的过渡电阻、负荷水平等通常与训练同分布，超出训练范围的外推测试很少见（见相关工作一节）。",
        f"Existing methods fall into two families. Physics-based methods use the network model: impedance-based methods {c('MoraFlorez2008','Chandran2024')}, voltage-sag and response matching {c('Buzo2020','Zhang2020')}, "
        f"sparse recovery and compressed sensing {c('Jia2020','Jiang2021','Shan2024')}, and two-stage methods that screen candidates and then run a nonlinear state estimation {c('DLiang2025')}. "
        f"They do not depend on training data, but with very few sensors the responses of different sections are similar and the residual separates them poorly. "
        f"Data-driven methods learn a mapping from measurements to the fault location with support vector machines and ensembles {c('Thukaram2005','Ghaemi2022','Okumus2025')}, convolutional networks {c('Siddique2024','Khattak2025')} "
        f"and graph neural networks {c('Chen2019','LLiang2024','Karabulut2026')}. They are accurate within the training distribution but degrade when test conditions leave it. "
        f"In the papers we reviewed, the fault resistance and load level of the test set generally follow the training distribution, and extrapolation beyond the training range is rarely tested (see Section 2).\n"))
    w(t(
        "这两类方法的失效条件不同：数据驱动方法在训练分布外失效，物理方法在响应相近的区段之间失效。本文据此提出“数据粗筛、物理校正”的两阶段方法：神经网络把候选区段缩小到两个，"
        "再用过渡电阻限定为实数的物理反演逐个核对并估计区段内位置，最后用贝叶斯联合打分合并两者的证据；两路证据都不能解释量测时扩大搜索范围，仍不能解释则发出失配告警。",
        "The two families fail under different conditions: data-driven methods fail outside the training distribution, physics-based methods fail between sections with similar responses. "
        "We therefore propose a two-stage method in which a neural network screens the faulted section down to two candidates, a physics-based inversion with a real-valued fault resistance verifies each candidate and estimates the position within the section, "
        "and a Bayesian joint score combines the two pieces of evidence. When neither explains the measurements, the search is widened, and if it still fails a model-mismatch alarm is raised.\n"))
    w(t("本文的贡献：\n", "The contributions of this paper are:\n"))
    d.contrib(intro_contrib)


def related(d):
    t, c, w = d.t, d.c, d.w
    w(t("## 2 相关工作\n", "## 2 Related Work\n"))
    w(t(
        f"**物理方法。** 阻抗法由视在阻抗估计故障距离，对过渡电阻和负荷敏感{c('MoraFlorez2008')}，同步相量与分布式电源下的扩展见{c('Chandran2024')}。"
        f"电压凹陷方法借助阻抗矩阵与分块把方程化为可解形式{c('Buzo2020')}，也有先用分类器选支路、再用仿真库相似度匹配的做法{c('Zhang2020')}。"
        f"稀疏量测方法把故障建模为稀疏的虚拟注入电流，用贝叶斯压缩感知或块稀疏贝叶斯学习求解，并讨论了最少测点数{c('Jia2020','Jia2020b','Jiang2021','Shan2024','Shan2024b')}；"
        f"Jia 等在稀疏混合量测下进一步采用两阶段方法{c('Jia2025')}。Liang 等的两阶段方法第一阶段用虚拟电流的线性模型匹配筛选可疑线路，第二阶段对候选做非线性故障状态估计并定位到线内{c('DLiang2025')}，"
        f"与本文结构最接近；不同之处是其第一阶段不含学习先验，据其摘要未见神经网络概率与物理残差的联合打分，也未见扩大搜索后的失配告警。",
        f"**Physics-based methods.** Impedance-based methods estimate the fault distance from the apparent impedance and are sensitive to fault resistance and load {c('MoraFlorez2008')}; extensions with synchrophasors and distributed generation are given in {c('Chandran2024')}. "
        f"Voltage-sag methods use impedance matrices and block partitioning to make the equations solvable {c('Buzo2020')}; others select a branch with a classifier and then match against a simulated library by similarity {c('Zhang2020')}. "
        f"Sparse-measurement methods model the fault as a sparse virtual injection current and solve it by Bayesian compressed sensing or block-sparse Bayesian learning, and discuss the minimum number of meters {c('Jia2020','Jia2020b','Jiang2021','Shan2024','Shan2024b')}; "
        f"Jia et al. further use a two-stage method with sparse hybrid measurements {c('Jia2025')}. The two-stage method of Liang et al. screens suspect lines with a linear model matching of virtual currents and then runs a nonlinear fault state estimation on the candidates to locate the fault within the line {c('DLiang2025')}; "
        f"it is the closest in structure to ours. It differs in that its first stage has no learned prior; judging from its abstract, it has no joint score of neural-network probability and physics residual, and no mismatch alarm after widening the search.\n"))
    w(t(
        f"**数据驱动方法。** 经典机器学习用电压电流特征训练 SVM、人工神经网络、随机森林或其堆叠{c('Thukaram2005','Jamali2020','Ghaemi2022','Okumus2025')}；深度学习包括卷积网络{c('Siddique2024','Mirshekali2023','Khattak2025')}和图网络。"
        f"Chen 等用切比雪夫图卷积在 IEEE 123 节点上分类故障母线，并与 PCA+SVM、PCA+随机森林和全连接网络比较{c('Chen2019')}；后续工作有图注意力网络{c('LLiang2024')}、时空图网络{c('Karabulut2026')}、"
        f"含边属性的图网络{c('Sun2021')}和深度图学习{c('Hu2023')}。Khattak 和 Choudhry 在改进的 IEEE 34 与 IEEE 123 节点上分别用 5 个和 9 个 PMU 的故障前后差值相量训练卷积网络{c('Khattak2025')}，"
        f"测点数与本文相近，但其评测未含噪声、负荷偏移和超出训练范围的过渡电阻，也没有物理反演。",
        f"**Data-driven methods.** Classical machine learning trains SVMs, artificial neural networks, random forests or their stacks on voltage and current features {c('Thukaram2005','Jamali2020','Ghaemi2022','Okumus2025')}; deep learning includes convolutional networks {c('Siddique2024','Mirshekali2023','Khattak2025')} and graph networks. "
        f"Chen et al. classify the faulted bus of the IEEE 123-bus system with Chebyshev graph convolutions and compare with PCA+SVM, PCA+random forest and a fully connected network {c('Chen2019')}; later work includes graph attention networks {c('LLiang2024')}, spatio-temporal graph networks {c('Karabulut2026')}, "
        f"graph networks with edge attributes {c('Sun2021')} and deep graph learning {c('Hu2023')}. Khattak and Choudhry train a convolutional network on pre/post-fault difference phasors from 5 and 9 PMUs on modified IEEE 34- and 123-bus systems {c('Khattak2025')}; "
        f"the number of sensors is close to ours, but their evaluation includes no noise, no load shift and no fault resistance beyond the training range, and no physics-based inversion.\n"))
    w(t(
        f"**物理与数据结合。** 已有工作把物理结构放进网络：Li 和 Deka 按物理结构构造邻接矩阵并分两个阶段学习嵌入与标签传播{c('Li2023')}；Zheng 等在图注意力网络的损失中加入基尔霍夫电流定律残差项{c('Zheng2026')}；"
        f"Hu 等把物理网络结构作为训练约束{c('Hu2023')}。这些方法中物理信息作用于训练或网络结构，推理时不求解物理反演，也没有基于残差的校验与告警。"
        f"关于线路参数误差，Shan 等的摘要提到方法对线路参数误差具有抗性{c('Shan2024b')}，Yu 等用稳态数据先校正线路参数再定位{c('Yu2022')}。"
        f"低可观测条件下的理论工作指出，测点远少于可观性要求时只能把故障定位到子图而不是单条线路{c('Jamei2018','Li2019')}，这与本文极稀疏设定下需要“候选加核对”的思路一致。",
        f"**Combining physics and data.** Existing work puts physical structure into the network: Li and Deka build the adjacency matrix from the physical structure and learn embeddings and propagate labels in two stages {c('Li2023')}; Zheng et al. add a Kirchhoff current-law residual term to the loss of a graph attention network {c('Zheng2026')}; "
        f"Hu et al. use the physical network structure as a training constraint {c('Hu2023')}. In these methods physics acts on the training or the network structure; no physics-based inversion is solved at inference time, and there is no residual-based verification or alarm. "
        f"On line-parameter errors, the abstract of Shan et al. mentions robustness to them {c('Shan2024b')}, and Yu et al. first calibrate line parameters from steady-state data {c('Yu2022')}. "
        f"Theoretical work on low-observability settings notes that with far fewer sensors than observability requires, a fault can only be localized to a subgraph rather than a single line {c('Jamei2018','Li2019')}, which is consistent with our use of candidates followed by verification.\n"))
    w(t(
        "表 1 归纳了与本文最接近的方法。我们检索到的文献中，未见到把神经网络概率与物理残差联合打分、在候选残差都偏大时扩大搜索并发出失配告警、并系统测试过渡电阻超出训练范围的工作；这一判断基于摘要与部分全文，不能代替完整检索。\n",
        "Table 1 summarizes the closest methods. In the literature we reviewed, we did not find work that jointly scores neural-network probability and physics residual, widens the search and raises a mismatch alarm when all candidate residuals are large, and systematically tests fault resistance beyond the training range; this judgment rests on abstracts and partial full texts and is not a substitute for an exhaustive search.\n"))
    d.table1()


def method(d):
    t, c, w = d.t, d.c, d.w
    w(t("## 3 方法\n", "## 3 Method\n"))
    w(t("### 3.1 量测模型与总体流程\n", "### 3.1 Measurement model and overall procedure\n"))
    w(t(
        "设测点共有 $m$ 个复数量测（各测点母线的三相电压和一条相连线路的三相电流），故障前后的相量之差记为突变量 $\\mathbf z\\in\\mathbb C^{m}$。网络线性化后，位置为 $(s,\\alpha)$（区段 $s$，区段内位置 $\\alpha\\in(0,1)$）、"
        "故障相集合为 $\\tau$ 的故障，其突变量满足 $\\mathbf z=\\mathbf G_{\\tau}(s,\\alpha)\\,\\mathbf i+\\mathbf n$，其中 $\\mathbf G_{\\tau}(s,\\alpha)$ 是故障相单位注入电流的响应（由节点导纳矩阵求得），$\\mathbf i$ 为注入电流，$\\mathbf n$ 为量测噪声。"
        "方法分三阶段：阶段一用神经网络给出区段后验概率并取前 $k=2$ 个候选；阶段二对每个候选做物理反演，得到加权残差 $J$ 和区段内位置；阶段三联合两者判决，并在必要时扩大候选集或发出告警（图 1）。",
        "Let the sensors provide $m$ complex measurements (the three-phase voltages at each sensor bus and the three-phase currents of one adjacent line); the difference of the phasors after and before the fault is the disturbance $\\mathbf z\\in\\mathbb C^{m}$. "
        "After linearization, a fault at position $(s,\\alpha)$ (section $s$, position $\\alpha\\in(0,1)$ within the section) on phase set $\\tau$ satisfies $\\mathbf z=\\mathbf G_{\\tau}(s,\\alpha)\\,\\mathbf i+\\mathbf n$, "
        "where $\\mathbf G_{\\tau}(s,\\alpha)$ is the response to a unit injection current on the faulted phases (obtained from the nodal admittance matrix), $\\mathbf i$ is the injected current and $\\mathbf n$ is the measurement noise. "
        "The method has three stages: stage 1 gives the posterior probability of each section with a neural network and keeps the top $k=2$ candidates; stage 2 runs a physics-based inversion on each candidate and returns a weighted residual $J$ and the position within the section; "
        "stage 3 combines the two to make the decision, widening the candidate set or raising an alarm when necessary (Fig. 1).\n"))
    d.fig(1)
    w(t("### 3.2 阶段一：区段后验概率\n", "### 3.2 Stage 1: section posterior\n"))
    w(t(
        "故障电流的复幅值未知，所以特征取各分量与幅值最大的分量之比 $\\mathbf u=\\mathbf z/z_{\\mathrm{ref}}$ 的实部与虚部，另附各分量的对数信噪比 $\\log_{10}(|z_i|/\\sigma_i)$ 和参考分量的对数信噪比，共 $3m+1$ 维。"
        "分类器是全连接网络（IEEE 34 为 160-160-96，IEEE 123 为 256-256-192），输出各区段的 softmax 概率 $P(s\\mid\\mathbf z)$。训练时每个批次重新叠加量测噪声；"
        "训练集除名义线路阻抗的样本外，加入线路阻抗倍率为 0.88、0.94、1.06、1.12 的样本，使网络对线路参数的整体偏差不过度敏感。",
        "The complex amplitude of the fault current is unknown, so the features are the real and imaginary parts of the ratio of each component to the largest-magnitude component, $\\mathbf u=\\mathbf z/z_{\\mathrm{ref}}$, "
        "plus the log signal-to-noise ratio $\\log_{10}(|z_i|/\\sigma_i)$ of each component and of the reference component, $3m+1$ features in all. "
        "The classifier is a fully connected network (160-160-96 for IEEE 34, 256-256-192 for IEEE 123) that outputs the softmax probability $P(s\\mid\\mathbf z)$ of each section. Noise is re-drawn for every training batch; "
        "besides samples with nominal line impedance, the training set contains samples with line-impedance scales of 0.88, 0.94, 1.06 and 1.12, so that the network is not overly sensitive to a global line-parameter error.\n"))
    w(t("### 3.3 阶段二：过渡电阻为实数的约束物理反演\n", "### 3.3 Stage 2: physics-based inversion with a real fault resistance\n"))
    w(t(
        "若注入电流 $\\mathbf i$ 取自由复数，每个区段多出一个复自由度，在测点很少时足以拟合其他区段的响应，区段之间的残差差距因此缩小。实际上故障电流由过渡电阻 $R$（实数）和故障点自身阻抗决定。"
        "设故障点故障前电压为 $\\mathbf V_{f}^{\\mathrm{pre}}$，故障点自身阻抗矩阵为 $\\mathbf Z_{ff}$，$\\mathbf E$ 为故障相的连接矩阵，则故障电流为 $\\mathbf i_f(R)=(R\\mathbf E+\\mathbf Z_{ff})^{-1}\\mathbf V_f^{\\mathrm{pre}}$，"
        "注入电流 $\\mathbf i=-\\mathbf i_f$（相间故障取相应的差分形式）。再引入实数增益 $g$ 吸收故障前电压相对名义值的偏差，其先验为 $g\\sim\\mathcal N(1,\\sigma_g^2)$（$\\sigma_g=0.05$），于是",
        "If the injected current $\\mathbf i$ is a free complex vector, each section gains a complex degree of freedom that, with few sensors, suffices to fit the response of other sections, and the residual gap between sections shrinks. "
        "In reality the fault current is determined by the fault resistance $R$ (a real number) and the impedance at the fault point. "
        "Let $\\mathbf V_{f}^{\\mathrm{pre}}$ be the pre-fault voltage at the fault point, $\\mathbf Z_{ff}$ the self-impedance matrix at the fault point and $\\mathbf E$ the connection matrix of the faulted phases; the fault current is $\\mathbf i_f(R)=(R\\mathbf E+\\mathbf Z_{ff})^{-1}\\mathbf V_f^{\\mathrm{pre}}$, "
        "and the injection is $\\mathbf i=-\\mathbf i_f$ (a differential form for phase-to-phase faults). A real gain $g$ absorbs the deviation of the pre-fault voltage from its nominal value, with prior $g\\sim\\mathcal N(1,\\sigma_g^2)$ ($\\sigma_g=0.05$), so that\n"))
    w("$$J(s,\\alpha,R,g)=\\big\\|\\mathbf W^{1/2}\\big(\\mathbf z-g\\,\\mathbf G_{\\tau}(s,\\alpha)\\,\\mathbf i(R)\\big)\\big\\|^{2}+\\Big(\\frac{g-1}{\\sigma_g}\\Big)^{2},\\qquad W_{ii}=\\frac{1}{\\sigma_i^{2}+(\\varepsilon_r|z_i|)^{2}+(\\varepsilon_a z_{\\max})^{2}}$$\n")
    w(t(
        "权重中的 $\\varepsilon_r=2\\%$、$\\varepsilon_a=1\\%$ 表示线性化的模型误差，$z_{\\max}$ 为同类量测的最大幅值。对每个候选区段，在 49 个等间隔位置 $\\alpha\\in[0.02,0.98]$ 与 48 点对数网格 $R\\in[0.01,3000]\\,\\Omega$ 上求 $J$（$g$ 有闭式解），再在最优点附近以 13 点细搜 $R$；"
        "对线路阻抗 0.9、0.95、1.0、1.05、1.1 倍的五个假设各算一遍，惩罚残差为 $J_{\\mathrm{pen}}=J+((s_z-1)/0.05)^2$（$s_z$ 为阻抗倍率），取最小者；报告的残差 $J$ 不含这一先验项。输出为 $J_{\\mathrm{pen}}$、$J$ 与位置估计 $\\hat\\alpha$、故障类型。",
        "In the weights, $\\varepsilon_r=2\\%$ and $\\varepsilon_a=1\\%$ represent the linearization model error and $z_{\\max}$ is the largest magnitude among measurements of the same kind. For each candidate section, $J$ is evaluated on 49 equally spaced positions $\\alpha\\in[0.02,0.98]$ and a 48-point logarithmic grid $R\\in[0.01,3000]\\,\\Omega$ ($g$ has a closed form), "
        "and $R$ is then refined on 13 points around the optimum. Five line-impedance hypotheses (0.9, 0.95, 1.0, 1.05 and 1.1 times) are each evaluated, with penalized residual $J_{\\mathrm{pen}}=J+((s_z-1)/0.05)^2$ ($s_z$ is the impedance scale), and the smallest is kept; the reported residual $J$ excludes this prior term. "
        "The outputs are $J_{\\mathrm{pen}}$, $J$, the position estimate $\\hat\\alpha$ and the fault type.\n"))
    w(t("### 3.4 阶段三：联合打分、候选扩展与告警\n", "### 3.4 Stage 3: joint score, candidate expansion and alarm\n"))
    w(t(
        "候选区段的得分为 $S_s=\\ln P(s\\mid\\mathbf z)-\\lambda\\,J_{\\mathrm{pen}}(s)$，取得分最大的候选；$\\lambda$ 在验证集上最大化真实等价类的联合后验似然得到。"
        "设 $\\theta$ 为假设正确时残差 $J$ 的 99% 分位数（验证集上标定）：若两个候选的 $J$ 都大于 $\\theta$，说明阶段一的候选集不可信，把候选扩展到全部区段重新打分；扩展后最小的 $J$ 仍大于 $\\theta$，则标记模型失配告警。"
        "置信度取候选池内 $\\mathrm{softmax}(S)$ 在所选区段所属等价类上的概率。等价类由物理证据定义：真实故障在 $s$ 时，若不少于一半的样本中区段 $s'$ 的惩罚残差不比 $s$ 大出 4，则两者互相不可区分，双向不可区分的区段连通成一类。",
        "The score of a candidate is $S_s=\\ln P(s\\mid\\mathbf z)-\\lambda\\,J_{\\mathrm{pen}}(s)$ and the candidate with the largest score is chosen; $\\lambda$ is fitted on a validation set by maximizing the joint posterior likelihood of the true equivalence class. "
        "Let $\\theta$ be the 99th percentile of the residual $J$ when the hypothesis is correct (calibrated on the validation set). If the $J$ of both candidates exceeds $\\theta$, the candidate set from stage 1 is not trustworthy, and the candidates are expanded to all sections and re-scored; "
        "if the smallest $J$ after expansion still exceeds $\\theta$, a model-mismatch alarm is raised. The confidence is the probability that $\\mathrm{softmax}(S)$ over the candidate pool places on the equivalence class of the chosen section. "
        "Equivalence classes are defined from physical evidence: for a true fault in $s$, if in at least half of the samples the penalized residual of section $s'$ is not larger than that of $s$ by more than 4, the two cannot be distinguished; sections that are mutually indistinguishable are connected into one class.\n"))


def setup(d):
    t, c, w = d.t, d.c, d.w
    w(t("## 4 实验设置\n", "## 4 Experimental Setup\n"))
    w(t("### 4.1 系统、测点与数据\n", "### 4.1 Systems, sensors and data\n"))
    w(t(
        f"测试系统为 IEEE 34 节点馈线（30 个故障区段）和 IEEE 123 节点馈线（118 个故障区段，开关作为短连接线），仿真用 OpenDSS{c('OpenDSS')}。IEEE 34 的 5 个测点位于母线 814、836、802、822、834，"
        "IEEE 123 的 8 个测点位于母线 60、47、95、1、108、27、81、30（图 2）；测点按物理穷举的平均测距误差贪心前向选取。每个测点量测母线的三相电压和一条相连线路的三相电流，IEEE 34 共 26 个复数量测，IEEE 123 共 44 个。"
        "量测噪声：电压为名义电压的 0.1%，电流为 0.2 A，突变量为两次量测之差，噪声标准差再乘 $\\sqrt2$。故障共 10 种类型（单相接地、相间、两相接地、三相接地），位置沿全网均匀，过渡电阻在训练集中对数均匀取 0.01–300 Ω，负荷倍率取 0.6–1.1。"
        "训练集 IEEE 34 为 8 万个名义阻抗样本加线路阻抗倍率 0.88、0.94、1.06、1.12 各 2 万个，IEEE 123 为 12 万个加各 3 万个；每个测试场景 5000 个样本。",
        f"The test systems are the IEEE 34-bus feeder (30 fault sections) and the IEEE 123-bus feeder (118 fault sections; switches are modeled as short connections), simulated with OpenDSS {c('OpenDSS')}. The five sensors of IEEE 34 are at buses 814, 836, 802, 822 and 834, "
        "and the eight of IEEE 123 at buses 60, 47, 95, 1, 108, 27, 81 and 30 (Fig. 2); they were chosen by greedy forward selection on the mean location error of exhaustive physics-based inversion. Each sensor measures the three-phase voltages of its bus and the three-phase currents of one adjacent line, "
        "giving 26 complex measurements for IEEE 34 and 44 for IEEE 123. Measurement noise is 0.1% of the nominal voltage and 0.2 A for current; the disturbance is the difference of two measurements, so the noise standard deviation is multiplied by $\\sqrt2$. "
        "There are 10 fault types (single-phase-to-ground, phase-to-phase, two-phase-to-ground, three-phase-to-ground), the position is uniform over the whole network, the fault resistance in the training set is log-uniform in 0.01–300 Ω and the load multiplier is 0.6–1.1. "
        "The IEEE 34 training set has 80,000 nominal-impedance samples plus 20,000 each at line-impedance scales 0.88, 0.94, 1.06 and 1.12; IEEE 123 has 120,000 plus 30,000 each. Each test scenario has 5000 samples.\n"))
    d.fig(2)
    w(t("### 4.2 测试场景\n", "### 4.2 Test scenarios\n"))
    w(t(
        "所有场景使用同一个训练好的模型，只改变测试条件。核心场景 5 个：域内（与训练同分布）；负荷偏移（全网负荷 1.25–1.5 倍，训练为 0.6–1.1）；高阻外推（过渡电阻 300–1000 Ω，训练为 0.01–300 Ω）；噪声 ×3（量测噪声放大 3 倍）；"
        "个别线路阻抗 ±20%（每条线路阻抗独立取名义值的 0.8–1.2 倍，模型仍用名义值）。此外，噪声敏感性取 ×0.5、×1、×2、×3、×5（图 3b）；与要求调压器档位已知的 Liang 两阶段方法比较时，用调压器档位固定的域内、负荷偏移、高阻外推、噪声 ×3 四个场景（各 3000 样本）；"
        "全网线路阻抗整体偏差 ±5%/±10%、各负荷功率独立变化、线路与负荷同时偏差等补充场景见附录表 A5。",
        "All scenarios use the same trained model and only change the test conditions. There are five core scenarios: in-domain (same distribution as training); load shift (network-wide load 1.25–1.5 times, trained 0.6–1.1); high resistance (fault resistance 300–1000 Ω, trained 0.01–300 Ω); noise ×3 (measurement noise tripled); "
        "and individual line impedance ±20% (each line impedance independently 0.8–1.2 times its nominal value while the model keeps nominal values). In addition, the noise sensitivity uses ×0.5, ×1, ×2, ×3 and ×5 (Fig. 3b); for the comparison with the method of Liang et al., which requires known regulator taps, the four fixed-tap scenarios (in-domain, load shift, high resistance, noise ×3; 3000 samples each) are used; "
        "supplementary scenarios such as global line-impedance errors of ±5%/±10%, independent load variation and combined line and load errors are in Table A5 of the appendix.\n"))
    w(t("### 4.3 对比方法\n", "### 4.3 Compared methods\n"))
    w(t(
        f"共 6 个外部方法，其中 4 个为已发表方法，2 个为机器学习基线，均在相同的训练集上训练、使用相同的量测。已发表方法：（1）GCN{c('Chen2019')}，切比雪夫图卷积，输入有“论文输入”（故障期间相量的幅值与相角）和“突变量输入”（与阶段一相同的特征）两种；"
        f"（2）GAT{c('LLiang2024')}，3 层 4 头图注意力，注意力限于物理相邻母线，两种输入同 GCN；（3）块稀疏贝叶斯学习{c('Jiang2021')}，以各候选位置的三相虚拟注入电流为未知量，用 EM 迭代估计块方差，取能量最大的块；位置受 10 点 $\\alpha$ 网格限制，不使用神经网络，线路参数取名义值；"
        f"（4）Liang 两阶段方法{c('DLiang2025')}，线性模型匹配筛选加非线性故障状态估计，假设调压器档位与网络参数精确已知，因此只在档位固定的场景比较，并获得多于本方法的量测（根母线相量、末端电压幅值、全网负荷伪量测）。"
        f"机器学习基线：（5）PCA+SVM 和（6）PCA+随机森林，沿用 Chen 等的对照协议{c('Chen2019')}，输入为阶段一的特征，标准化后做 PCA（保留 99% 方差）；SVM 为 RBF 核，受核方法计算量限制只用 3 万个训练子样本。"
        f"除 Liang 两阶段方法和块稀疏贝叶斯学习外，所有分类器都只输出区段，故障点取预测区段中点；我们另外把每个分类器的 Top-1 区段交给阶段二估计区段内位置（记作“+物理定位”），或把它的概率作为阶段一接入完整流程（记作“+本方法流程”），以分开评价区段选择与位置估计。"
        f"内部对照还有神经网络 Top-1 加物理定位（硬串联）、全区段物理穷举、已知区段的 Oracle。所有基线均为我们按原文复现，与原作者的实现可能有差异，详见附录表 A6。",
        f"There are six external methods, four published methods and two machine-learning baselines, all trained on the same training set and using the same measurements. The published methods are: (1) GCN {c('Chen2019')}, Chebyshev graph convolution, with two inputs, the “paper input” (magnitude and angle of the during-fault phasors) and the “disturbance input” (the same features as stage 1); "
        f"(2) GAT {c('LLiang2024')}, three layers with four attention heads restricted to physically adjacent buses, with the same two inputs; (3) block-sparse Bayesian learning {c('Jiang2021')}, which takes the three-phase virtual injection current at each candidate position as the unknown and estimates the block variances by EM, choosing the block with the largest energy; the position is limited to a 10-point $\\alpha$ grid, no neural network is used and line parameters are nominal; "
        f"(4) the two-stage method of Liang et al. {c('DLiang2025')}, linear model matching for screening followed by a nonlinear fault state estimation, which assumes exactly known regulator taps and network parameters and is therefore compared only in the fixed-tap scenarios, and which receives more measurements than our method (root-bus phasors, end-bus voltage magnitudes, load pseudo-measurements). "
        f"The machine-learning baselines are (5) PCA+SVM and (6) PCA+random forest, following the comparison protocol of Chen et al. {c('Chen2019')}, with the stage-1 features standardized and reduced by PCA (99% of the variance); the SVM has an RBF kernel and, because of the cost of kernel methods, is trained on a random subset of 30,000 samples. "
        f"Except for the method of Liang et al. and block-sparse Bayesian learning, all classifiers output only the section, and the fault point is taken at the section midpoint. To separate section selection from position estimation we also hand each classifier's Top-1 section to stage 2 to estimate the position within the section (“+physics location”), "
        f"or feed its probability into the full procedure as stage 1 (“+our procedure”). Internal references are neural-network Top-1 followed by physics-based location (hard cascade), exhaustive physics-based inversion over all sections, and an oracle with the section known. All baselines are our reimplementations from the original papers and may differ from the original authors' code; see Table A6.\n"))
    w(t("### 4.4 评价指标与统计\n", "### 4.4 Metrics and statistics\n"))
    w(t(
        "区段准确率按等价类判定（IEEE 34 为 22 类，IEEE 123 为 64 类；等价类由物理证据定义并固定，使不同方法同口径），同时报告精确区段准确率。测距误差为预测故障点与真实故障点沿馈线到变电站的距离之差的绝对值（km），报告均值，必要时报告中位数、P90 与分位数。"
        "纠错率 CR 与误改率 FCR 以阶段一 Top-1 的对错为参照。神经网络训练的随机性用多个种子衡量：IEEE 34 上本方法与 GCN 为 5 个种子，GAT 与 PCA 基线为 3 个种子；块稀疏贝叶斯没有随机性，Liang 两阶段为单次结果；IEEE 123 上本方法 5 个种子、GCN 3 个种子，其余基线为单次结果；报告均值±标准差，均值差超过两者较大标准差的 2 倍才称为显著。",
        "Section accuracy is judged at the level of equivalence classes (22 classes for IEEE 34, 64 for IEEE 123; the classes are defined from physical evidence and fixed so that all methods are scored the same way), and exact section accuracy is also reported. "
        "The location error is the absolute difference (km) between the predicted and true fault points measured along the feeder from the substation; we report the mean and, where useful, the median, P90 and quantiles. "
        "The correction rate CR and false-correction rate FCR are defined relative to the correctness of the stage-1 Top-1. Training randomness is measured with several seeds: on IEEE 34, five seeds for our method and GCN and three for GAT and the PCA baselines (block-sparse Bayesian learning has no randomness and the method of Liang et al. is a single run); "
        "on IEEE 123 five seeds for our method, three for GCN and single runs for the other baselines; results are reported as mean ± standard deviation; a difference of means is called significant only if it exceeds twice the larger standard deviation.\n"))


def intro_contrib(d):
    pass
