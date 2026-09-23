# 树麻雀声学数据工具 1.0.1

原创代码采用 [MIT 许可](LICENSE)，配套数值输入采用 [CC BY 4.0](ASSET_LICENSE.md)。
科学环境按 Python 3.10 与 `requirements-observed.txt` 安装；采集和绘图组件分别使用自己的环境。

在 `code/analysis/` 中运行：

```bash
python -m pip install -r requirements-observed.txt
python -m pip install --no-deps .
sparrow-data inspect --package ../../data
sparrow-data reconstruct --package ../../data
sparrow-data export-raven --package ../../data --output-dir ../../outputs/raven
sparrow-data assessment inspect
```

示例假设 `data/` 与 `code/` 同级；数据位于其他位置时更换路径。
工具核对七表、源身份、27个family／41个叶类别和138份Raven表；按整数帧验证
2,556音节的精确重切。`reconstruct` 可用 `--syllable-id` 选单条，增加新
`--output-dir` 导出FLOAT WAV。Raven原行号不是Selection序号，未核实的原MP3偏移不能代替裁剪坐标。

主要功能：

- `features`：声学参数、105维特征或log-mel；默认输出计划，`--run`才计算。
- `assessment taxonomy/songiness`：使用匹配数值输入运行内部标签评估和鸣唱排序；默认输出计划。
- `python -m sparrow_dataset.cluster ... --dry-run`：检查UMAP/HDBSCAN参数和2,893行输入；去掉`--dry-run`才聚类。
- `python -m sparrow_dataset.annotation.archive ... --dry-run`：检查人工映射，将2,560条已审阅输入映射为2,556条保留音节；不自动再生此前人工判断。未提供音频根时仅输出CSV。
- `python -m sparrow_dataset.assessments.mp3_assessment ... --dry-run`：核对11组固定WAV/MP3及参考结果；去掉`--dry-run`复算频谱测量，不重新编码MP3。
- `process`／`stage-help`：库存、QC、Raven规范化、数据装配与录音级特征。装配所需额外输入见 [INPUTS.md](INPUTS.md)。

计算均使用新输出目录。标签评估输入边界与公开边界的对应关系见
`data/validation/label_assessment/assessment_input_bounds.csv`；不能直接用不同边界新提特征替换评估数组。
鸣唱排序输入中的辅助八类词表与最终family标签不同；已审阅正负样本不代表公开录音的鸣唱流行率。
已处理WAV不应再次带通，公开子集也不能还原缺失的原录音或人工决策。

完整参数和可直接修改路径的示例见 [English README](README.md)。
回归检查：`python -m unittest discover -s tests -q`。

本版本默认标签评估对应R1边界＋R2分类，使用127对结构内family参照；`retry-seed`仅处理训练折类别覆盖可行性，完整记录试种子及train/test成员。人工精炼stage的输入仍是历史43叶、2560音节；新增`structure_reclassification`明确迁移74条，且同步结构和family字段。其旧研究边界不替换R1改界。包内保留历史map供追溯，默认当前map为R2。

历史聚类改从独立的`assets/clustering/historical_run/`读取原2893行，与原RMS一一匹配；当前R1标签评估数组不覆盖该历史输入。两套原始/改界特征的用途分别标明。
