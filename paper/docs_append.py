"""把新增基线、分档与噪声档位、阶段一互换、IEEE 123 对比等结果追加进《实验结果与分析.md》。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import results_sections as RS  # noqa: E402
from build import Doc  # noqa: E402

d = Doc("zh")
w = d.w
w("## 十五、补充对比：新增基线、同分布分档与噪声档位\n")
w("本节的表与论文初稿（`论文初稿_中文.md`）中的表一致，表号沿用论文初稿（正文表 2、2b、3、4、6、7、9，图 4，附录表 A5、A7、A2、A10）。外部方法为 6 个：4 个已发表方法（GCN、GAT、块稀疏贝叶斯学习、Liang 两阶段，后者即前文的“L1”）和 2 个机器学习基线（PCA+SVM、PCA+随机森林）。"
  "新增基线的复现说明见论文初稿附录表 A6；GAT 的 IEEE 123 结果只运行了突变量输入。`baseline_ml.py`、`baseline_gcn.py --arch gat`、`baseline_bsbl.py` 生成各基线结果，`aggregate_seeds.py` 汇总多种子，`analyze_samples.py`、`analyze_tier1.py` 生成分档统计。\n")
w("**测距误差的口径**：一至十四节沿用当时的定义，即预测故障点与真实故障点到变电站的距离之差（“根距离误差”），预测到另一条支线时会被低估；本节与论文初稿改用二者沿馈线的最短路径距离（`analyze_path_error.py` → `results/improved/ieee34/path_error.json`，由逐样本区段与物理反演缓存复原，先复原根距离误差并与已存值核对一致）。区段已知时二者相同。\n")
RS.table_main(d)
RS.table_paired(d)
RS.table_breakdown(d)
RS.table_loc(d)
RS.sensors_section(d)
RS.table_swap(d)
RS.table_liang(d)
RS.table_123(d)
RS.appendix_full(d)
RS.appendix_extra(d)
RS.appendix_exact(d)
RS.table_sensors(d)
text = "\n".join(d.out)
doc = Path(__file__).resolve().parents[1] / "实验结果与分析.md"
s = doc.read_text()
marker = "## 十五、复现"
if "## 十五、补充对比" in s:
    head = s[: s.index("## 十五、补充对比")]
    tail = s[s.index("## 十六、复现"):]
else:
    head, tail = s[: s.index(marker)], s[s.index(marker):].replace(marker, "## 十六、复现", 1)
doc.write_text(head + text + "\n" + tail)
print("ok", len(text))
