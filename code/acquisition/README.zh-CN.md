# 完整录音采集与留存组件 — 1.0.0

[English](README.md)

本组件采集 30 秒录音，调用固定 BirdNET 模型，将满足条件的整段录音保存为 MP3，并保留检测 CSV 和分析凭据。一条合格检测即可保留整段录音，不截取为三秒片段。

| 模块 | 功能 |
| --- | --- |
| `pipeline.py` | ALSA 采集、采集凭据和串行模型分析 |
| `model_bridge.py` | 固定源文件、模型、标签校验及上游模型函数适配 |
| `retention.py` | 输入验证、筛选、编码验证及原子归档 |
| `manage.py` | 独立安装、完整性验证及可恢复卸载 |

## 环境与参数

维护层使用 Python 3.9+；采集需要 Linux ALSA `arecord`，编码需要带 `libmp3lame` 的 FFmpeg/ffprobe。推理需要相容的 NumPy、librosa、SciPy 和 TFLite 环境；参考组合为 Python 3.9、NumPy 1.19.5、librosa 0.9.2、SciPy 1.10.1、tflite-runtime 2.6.0。

只支持 BirdNET-Pi 提交 `6334367ac0257514967025487026d6fe6b351afe` 和固定 `BirdNET_6K_GLOBAL_MODEL`。模型权重需按适用许可另行获取；相应中文标签已附于 `vendor/labels.txt`。`model_bridge.py` 校验源文件、模型和标签 SHA-256，不接受其他版本；`upstream_contract.json` 列出所需上游文件。适配器仅加载八个允许的模型函数，不启动所附源文件中的服务器。

将 `config.example.json` 复制为私有配置文件，填写以下内容：

| 配置项 | 含义 |
| --- | --- |
| `runtime_user`、`runtime_python` | 非 root 运行账户及运行环境 Python 的绝对路径 |
| `recording_device`、`device_id`、`timezone` | ALSA 输入设备、唯一设备标识及 IANA 时区 |
| `latitude`、`longitude` | 模型使用的录音站点坐标 |
| `model_path`、`labels_path` | 模型和匹配标签文件的绝对路径 |
| `pending_dir`、`archive_dir`、`quarantine_dir` | 独立的待处理、归档和可选隔离目录 |

安装后可将 `labels_path` 指向 `/path/to/installation/vendor/labels.txt`；安装器会包含并校验该文件。三个数据目录必须互不相同、互不包含，且不能与源包、上游检出目录或安装目录相互包含。使用仅操作者可写的目录；程序拒绝符号链接及重解析路径。

默认规则要求科学名 `Passer montanus` 和原模型标签 `麻雀` 同时精确匹配，置信度 `>= 0.2`（`comparison: gte`；`gt` 表示严格大于）。这是采集筛选阈值，与数据集后续发布筛选分开。默认 sensitivity 为 1.25、overlap 为 1.5 秒、privacy_threshold 为 0；不支持其他隐私阈值。

## 安装、验证与回退

从本目录运行以下命令。安装父目录须已存在，安装目录只能为空或尚不存在；上游检出目录须匹配固定提交及文件合同。不加 `--apply` 时只输出安装或卸载计划。

```sh
python -B manage.py install --upstream /path/to/BirdNET-Pi --prefix /path/to/installation --config /path/to/config.json
python -B manage.py install --upstream /path/to/BirdNET-Pi --prefix /path/to/installation --config /path/to/config.json --apply
python -B manage.py verify --prefix /path/to/installation
python -B manage.py uninstall --prefix /path/to/installation
python -B manage.py uninstall --prefix /path/to/installation --apply
```

安装先准备固定文件集合，再通过目录重命名完成；相同安装可重复执行。程序生成供操作者启用的 systemd 建议配置，不修改现有服务。`verify` 校验安装文件内容及权限。需要更改安装配置时，应先卸载未改动的原安装，再用新配置重新安装。

卸载先校验受管文件并创建恢复事务，再逐个原子移走当前文件后验证。发现用户修改或替代文件时保留相关内容及恢复事务；不删除未知文件和录音。中断后再次执行 `uninstall --apply` 可恢复后续处理；存在冲突的恢复材料会保留供检查。

## 运行与数据留存

安装后使用安装目录中的脚本及 `config.json`：

```sh
python -B pipeline.py --config /path/to/installation/config.json capture-once
python -B pipeline.py --config /path/to/installation/config.json analyze-pending
python -B pipeline.py --config /path/to/installation/config.json run
python -B retention.py --config /path/to/installation/config.json --once
```

`capture-once` 生成一段 WAV 和采集凭据；`analyze-pending` 处理具有采集凭据的录音；`run` 同时执行采集和串行分析；仅留存命令处理生产者明确发布的分析完成标记。每段采集重新打开设备，可能存在段间间隙；使用前协调麦克风占用及已有清理任务。

调用链依次校验 48 kHz、单声道 PCM16 的 1,440,000 帧完整录音，执行模型，验证五列分号 CSV 的全部行，再发布完成标记。阳性录音编码为 48 kHz、单声道、320 kb/s MP3，解码后仍须恰为 1,440,000 帧才可归档。模型窗口补零不延长或裁切实际录音。

默认保留源 WAV 和阴性输入。`source_wav_policy: delete_after_commit` 仅在已提交归档和匹配源文件哈希通过验证后允许删除该源 WAV；`negative_policy: quarantine` 生成验证后的副本并保留输入。无效输入、转码失败、超时及空间不足都保留源录音；没有自动清空归档策略。磁盘预留量由 `min_free_bytes` 和 `capacity_buffer_bytes` 控制。

完成凭据绑定推理设置、目标标签和模型哈希。改变这些设置后拒绝复用旧过滤 CSV 和分析缓存，应保留原产物并在独立位置明确重新分析。仅改变留存策略可以继续使用匹配的已完成分析。

## 测试与许可

```sh
python -B -m unittest discover -s tests -v
```

测试使用临时录音并包含真实 FFmpeg 编解码检查；缺少 FFmpeg 时相应测试明确跳过。研究者已报告采集代码主体在树莓派上手动运行通过。本发布版新增的标签随安装分发由本地回归测试覆盖；该实机报告不证明新增安装步骤、重启行为或某一时长的无人值守运行已获验证。

原创维护代码采用 MIT；派生模型适配器及所附 BirdNET 材料保留 CC BY-NC-SA 4.0 和完整上游声明。具体范围见 [LICENSE](LICENSE)、[LICENSE-MIT](LICENSE-MIT) 和 [vendor/LICENSE](vendor/LICENSE)。
