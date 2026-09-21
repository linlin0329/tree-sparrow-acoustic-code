"""
配置参数模块
Configuration Parameters Module

包含所有可调节的参数，便于在 Jupyter Notebook 中进行调参实验。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class AudioConfig:
    """音频处理参数"""

    sample_rate: int = 48000  # 采样率 (Hz)
    freq_min: float = 1000  # 最低频率 (Hz) - 麻雀鸣声下限
    freq_max: float = 15000  # 最高频率 (Hz) - 麻雀鸣声上限
    min_syllable_duration: float = (
        0.03  # 最短音节时长 (秒) - 提高到30ms以确保足够的帧数
    )
    max_syllable_duration: float = 1.0  # 最长音节时长 (秒)


@dataclass
class FeatureConfig:
    """特征提取参数"""

    # 梅尔频谱参数
    n_mels: int = 128  # 梅尔频带数量
    n_fft: int = 2048  # FFT 窗口大小
    hop_length: int = 512  # 步长

    # MFCC 参数
    n_mfcc: int = 20  # MFCC 系数数量

    # PCEN 参数 (Per-Channel Energy Normalization)
    use_pcen: bool = True  # 是否使用 PCEN (对低信噪比有效)
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5
    pcen_time_constant: float = 0.4

    # 图像特征参数 (用于统一音节长度)
    fixed_width: int = 128  # 将所有音节缩放到统一的时间宽度

    # 特征类型选择
    feature_types: List[str] = field(
        default_factory=lambda: [
            "f0",  # 基频特征 (音高)
            "mfcc",  # MFCC 均值和标准差
            "spectral",  # 频谱特征 (质心、带宽、滚降点等)
            "temporal",  # 时域特征 (持续时间、ZCR等)
            "amplitude",  # 振幅包络特征 (新增)
        ]
    )


@dataclass
class UMAPConfig:
    """UMAP 降维参数"""

    n_neighbors: int = 15  # 邻居数量 (越小越关注局部结构)
    min_dist: float = 0.1  # 最小距离 (越小聚类越紧凑)
    n_components: int = 2  # 降维目标维度
    metric: str = "euclidean"  # 距离度量
    random_state: int = 42  # 随机种子，确保可复现


@dataclass
class HDBSCANConfig:
    """HDBSCAN 聚类参数"""

    min_cluster_size: int = 5  # 最小簇大小
    min_samples: int = 3  # 核心点最小样本数 (越大越保守)
    cluster_selection_epsilon: float = 0.0  # 聚类选择阈值
    cluster_selection_method: str = "eom"  # 聚类选择方法: "eom" 或 "leaf"
    allow_single_cluster: bool = False  # 是否允许单一簇


@dataclass
class OutputConfig:
    """输出配置"""

    output_dir: str = "output"  # 输出目录
    save_features: bool = True  # 是否保存特征矩阵
    save_spectrograms: bool = True  # 是否保存音节频谱图
    figure_format: str = "png"  # 图片格式 (png, pdf, svg)
    figure_dpi: int = 300  # 图片分辨率


@dataclass
class Config:
    """主配置类，包含所有子配置"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "audio": self.audio.__dict__,
            "feature": {
                k: v for k, v in self.feature.__dict__.items() if k != "feature_types"
            }
            | {"feature_types": self.feature.feature_types},
            "umap": self.umap.__dict__,
            "hdbscan": self.hdbscan.__dict__,
            "output": self.output.__dict__,
        }

    def save(self, path: str):
        """保存配置到 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = cls()
        if "audio" in data:
            config.audio = AudioConfig(**data["audio"])
        if "feature" in data:
            config.feature = FeatureConfig(**data["feature"])
        if "umap" in data:
            config.umap = UMAPConfig(**data["umap"])
        if "hdbscan" in data:
            config.hdbscan = HDBSCANConfig(**data["hdbscan"])
        if "output" in data:
            config.output = OutputConfig(**data["output"])
        return config

    def __repr__(self):
        return f"""Config(
  Audio: sr={self.audio.sample_rate}, freq=[{self.audio.freq_min}-{self.audio.freq_max}]Hz
  Feature: n_mels={self.feature.n_mels}, n_mfcc={self.feature.n_mfcc}, pcen={self.feature.use_pcen}
  UMAP: n_neighbors={self.umap.n_neighbors}, min_dist={self.umap.min_dist}, n_components={self.umap.n_components}
  HDBSCAN: min_cluster_size={self.hdbscan.min_cluster_size}, min_samples={self.hdbscan.min_samples}
)"""


def get_default_config() -> Config:
    """获取默认配置（针对麻雀鸣声优化）"""
    return Config()


def get_conservative_config() -> Config:
    """获取保守配置（更少的簇，更严格的噪声过滤）"""
    config = Config()
    config.hdbscan.min_cluster_size = 10
    config.hdbscan.min_samples = 5
    config.umap.n_neighbors = 30
    return config


def get_sensitive_config() -> Config:
    """获取敏感配置（更多的簇，更细粒度的分类）"""
    config = Config()
    config.hdbscan.min_cluster_size = 3
    config.hdbscan.min_samples = 2
    config.umap.n_neighbors = 10
    config.umap.min_dist = 0.05
    return config


# 预定义的参数网格，用于调参
PARAM_GRID = {
    "umap_n_neighbors": [5, 10, 15, 20, 30],
    "umap_min_dist": [0.0, 0.05, 0.1, 0.2, 0.5],
    "hdbscan_min_cluster_size": [3, 5, 10, 15, 20],
    "hdbscan_min_samples": [1, 3, 5, 10],
}
