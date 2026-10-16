"""生成论文初稿：python paper/build.py → 论文初稿_中文.md、Paper_Draft_EN.md（在仓库根目录运行；需先跑完实验并生成图）。
静态章节见 static.py，结果章节与表格见 results_sections.py，参考文献见 refs.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from refs import REFS  # noqa: E402
import static  # noqa: E402
import results_sections as RS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CAPTION = {
    1: ("图 1 方法框图。神经网络从全部区段中选出 Top-2 候选，物理反演逐个核对并估计区段内位置，联合打分判决；候选残差都偏大时扩展到全部区段，仍偏大则发出失配告警。",
        "Fig. 1 Block diagram. A neural network selects the Top-2 candidate sections from all sections, a physics-based inversion verifies each candidate and estimates the position within the section, and a joint score makes the decision; when all candidate residuals are large the search is widened to all sections, and if it is still large a mismatch alarm is raised."),
    2: ("图 2 测试系统的拓扑（以变电站为根、横轴为沿馈线到变电站的距离）与测点位置。",
        "Fig. 2 Topology of the test systems (rooted at the substation, horizontal axis is the distance along the feeder to the substation) and sensor locations."),
    3: ("图 3 (a) IEEE 34 上类准确率随过渡电阻的变化（域内为 0.01–300 Ω，高阻外推为 300–1000 Ω，虚线右侧超出训练范围）；(b) 类准确率随量测噪声倍数的变化（GCN、GAT 的论文输入为实线、突变量输入为虚线；块稀疏贝叶斯学习从 ×3 起低于 50%，未画出）。",
        "Fig. 3 (a) Class accuracy on IEEE 34 versus fault resistance (in-domain 0.01–300 Ω, high-resistance 300–1000 Ω; right of the dotted line is beyond the training range); (b) class accuracy versus measurement noise multiplier (solid: paper input, dashed: disturbance input for GCN and GAT; block-sparse Bayesian learning, below 50% from ×3 on, is not drawn)."),
    4: ("图 4 IEEE 34 上测距误差（沿馈线的最短路径距离）的累积分布：(左) 域内，(右) 高阻外推。",
        "Fig. 4 Cumulative distribution of the location error (shortest path along the feeder) on IEEE 34: (left) in-domain, (right) high resistance."),
}


class Doc:
    def __init__(self, lang):
        self.L, self.order, self.out = lang, [], []
        self.n_fig = self.n_tab = 0

    def t(self, zh, en):
        return zh if self.L == "zh" else en

    def c(self, *keys):
        nums = []
        for k in keys:
            if k not in REFS:
                raise KeyError(k)
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        return "[" + ",".join(map(str, sorted(set(nums)))) + "]"

    def w(self, x):
        self.out.append(x)

    def fig(self, n):
        self.w(f"![{CAPTION[n][0 if self.L == 'zh' else 1]}](figures/{self.L}/fig{n}.png)\n")
        self.w(f"*{CAPTION[n][0 if self.L == 'zh' else 1]}*\n")

    def table(self, num, caption_zh, caption_en, header, rows, align=None):
        cap = caption_zh if self.L == "zh" else caption_en
        label = f"表 {num}" if self.L == "zh" else f"Table {num}"
        self.w(f"**{label} {cap}**\n")
        align = align or (["---"] + ["---:"] * (len(header) - 1))
        self.w("| " + " | ".join(header) + " |\n|" + "|".join(a if a.startswith(":") or a.endswith(":") else ":--" for a in align) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")

    def contrib(self, fn):
        for line in RS.contributions(self):
            self.w(line)
        self.w("")

    def table1(self):
        RS.table1(self)



def build(lang):
    d = Doc(lang)
    d.w(f"# {d.t('极稀疏量测下的配电网故障定位：数据驱动粗筛与物理闭环校正', 'Fault Location in Distribution Networks under Extremely Sparse Measurements: Data-Driven Screening with Physics-Based Closed-Loop Verification')}\n")
    d.w(d.t("*初稿（实验结果均来自 OpenDSS 仿真，未使用实测数据）*\n", "*First draft (all results are from OpenDSS simulations; no field data are used)*\n"))
    RS.abstract(d)
    static.intro(d)
    static.related(d)
    static.method(d)
    static.setup(d)
    RS.results(d)
    RS.discussion(d)
    RS.conclusion(d)
    RS.statements(d)
    RS.appendix(d)
    d.w(d.t("## 参考文献\n", "## References\n"))
    for i, k in enumerate(d.order, 1):
        d.w(f"[{i}] {REFS[k]}\n")
    return "\n".join(d.out)


if __name__ == "__main__":
    (ROOT / "论文初稿_中文.md").write_text(build("zh"))
    (ROOT / "Paper_Draft_EN.md").write_text(build("en"))
    print("written")
