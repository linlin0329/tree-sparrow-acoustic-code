"""
音频处理与特征提取模块
Audio Processing and Feature Extraction Module

包含:
- Raven Pro 标注表读取
- 音节切割
- 多维声学特征提取
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from skimage.transform import resize
import warnings

from .config import Config, AudioConfig, FeatureConfig


@dataclass
class Syllable:
    """音节数据类"""

    source_file: str  # 源音频文件
    selection_id: int  # Raven 选择 ID
    begin_time: float  # 起始时间 (秒)
    end_time: float  # 结束时间 (秒)
    low_freq: float  # 低频边界 (Hz)
    high_freq: float  # 高频边界 (Hz)
    waveform: Optional[np.ndarray] = None  # 波形数据
    features: Optional[np.ndarray] = None  # 提取的特征向量

    @property
    def duration(self) -> float:
        """音节持续时间 (秒)"""
        return self.end_time - self.begin_time

    @property
    def bandwidth(self) -> float:
        """频率带宽 (Hz)"""
        return self.high_freq - self.low_freq

    @property
    def center_freq(self) -> float:
        """中心频率 (Hz)"""
        return (self.low_freq + self.high_freq) / 2


class RavenTableReader:
    """
    Raven Pro 选择表读取器

    支持读取 Raven Pro 导出的 .txt 选择表文件
    """

    # Raven Pro 标准列名映射
    COLUMN_MAPPING = {
        "Selection": "selection_id",
        "Begin Time (s)": "begin_time",
        "End Time (s)": "end_time",
        "Low Freq (Hz)": "low_freq",
        "High Freq (Hz)": "high_freq",
        "Channel": "channel",
        "View": "view",
    }

    def __init__(self):
        pass

    def read_table(self, table_path: str, audio_path: str = None) -> pd.DataFrame:
        """
        读取 Raven Pro 选择表

        Parameters
        ----------
        table_path : str
            Raven 选择表文件路径 (.txt)
        audio_path : str, optional
            关联的音频文件路径

        Returns
        -------
        pd.DataFrame
            包含音节信息的 DataFrame
        """
        table_path = Path(table_path)
        if not table_path.exists():
            raise FileNotFoundError(f"选择表文件不存在: {table_path}")

        # 读取制表符分隔的文件
        df = pd.read_csv(table_path, sep="\t")

        # 重命名列
        df = df.rename(columns=self.COLUMN_MAPPING)

        # 添加源文件信息
        if audio_path:
            df["source_file"] = str(audio_path)
        else:
            # 尝试从表格路径推断音频路径
            audio_name = table_path.stem.replace(".Table.1.selections", "")
            df["source_file"] = audio_name

        # 验证必要的列
        required_cols = ["begin_time", "end_time"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"选择表缺少必要的列: {col}")

        # 过滤无效行 (持续时间为0的行)
        df = df[df["end_time"] > df["begin_time"]].copy()

        return df

    def read_multiple_tables(
        self, table_audio_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        读取多个选择表并合并

        Parameters
        ----------
        table_audio_pairs : List[Tuple[str, str]]
            (选择表路径, 音频路径) 的列表

        Returns
        -------
        pd.DataFrame
            合并后的 DataFrame
        """
        all_dfs = []
        for table_path, audio_path in table_audio_pairs:
            df = self.read_table(table_path, audio_path)
            all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # 添加全局唯一 ID
        combined_df["global_id"] = range(len(combined_df))

        return combined_df


class AudioProcessor:
    """
    音频处理器

    负责音频加载、音节切割和特征提取
    """

    def __init__(self, config: Config = None):
        """
        初始化音频处理器

        Parameters
        ----------
        config : Config
            配置对象，如果为 None 则使用默认配置
        """
        self.config = config or Config()
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}

    def load_audio(
        self, audio_path: str, use_cache: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Parameters
        ----------
        audio_path : str
            音频文件路径
        use_cache : bool
            是否使用缓存

        Returns
        -------
        Tuple[np.ndarray, int]
            (波形数据, 采样率)
        """
        audio_path = str(audio_path)

        if use_cache and audio_path in self._audio_cache:
            return self._audio_cache[audio_path]

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        y, sr = librosa.load(audio_path, sr=self.config.audio.sample_rate)

        if use_cache:
            self._audio_cache[audio_path] = (y, sr)

        return y, sr

    def clear_cache(self):
        """清除音频缓存"""
        self._audio_cache.clear()

    def extract_syllable_waveform(
        self, audio_path: str, begin_time: float, end_time: float
    ) -> np.ndarray:
        """
        从音频中提取音节波形

        Parameters
        ----------
        audio_path : str
            音频文件路径
        begin_time : float
            起始时间 (秒)
        end_time : float
            结束时间 (秒)

        Returns
        -------
        np.ndarray
            音节波形
        """
        y, sr = self.load_audio(audio_path)

        start_sample = int(begin_time * sr)
        end_sample = int(end_time * sr)

        # 确保在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        return y[start_sample:end_sample]

    def extract_features(self, waveform: np.ndarray, sr: int = None) -> np.ndarray:
        """
        从波形中提取多维声学特征

        Parameters
        ----------
        waveform : np.ndarray
            音节波形
        sr : int
            采样率，默认使用配置中的值

        Returns
        -------
        np.ndarray
            特征向量
        """
        if sr is None:
            sr = self.config.audio.sample_rate

        cfg = self.config.feature
        audio_cfg = self.config.audio

        # 检查波形长度
        if len(waveform) < cfg.n_fft:
            # 如果太短，进行零填充
            waveform = np.pad(waveform, (0, cfg.n_fft - len(waveform)))

        features = []

        # ========== 0. 基频 (F0) 特征 ==========
        if "f0" in cfg.feature_types:
            try:
                # 使用 pyin 算法提取基频 (对鸟类鸣声更鲁棒)
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y=waveform,
                    sr=sr,
                    fmin=audio_cfg.freq_min,
                    fmax=audio_cfg.freq_max,
                    frame_length=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    fill_na=0.0,
                )

                # 只保留有声音的部分 (voiced)
                f0_voiced = f0[voiced_flag]

                if len(f0_voiced) > 0:
                    # 基本统计
                    features.append(np.mean(f0_voiced))  # 平均基频
                    features.append(np.std(f0_voiced))  # 基频标准差
                    features.append(np.min(f0_voiced))  # 最低音高
                    features.append(np.max(f0_voiced))  # 最高音高
                    features.append(np.max(f0_voiced) - np.min(f0_voiced))  # 音高范围

                    # 基频变化率 (一阶差分的标准差)
                    f0_diff = np.diff(f0_voiced)
                    features.append(np.mean(f0_diff))  # 平均变化率
                    features.append(np.std(f0_diff))  # 变化率标准差

                    # 基频变化方向性
                    if len(f0_diff) > 0:
                        features.append(
                            np.mean(np.sign(f0_diff))
                        )  # 上升/下降倾向 (-1到1)
                    else:
                        features.append(0)

                    # 有声帧比例
                    features.append(len(f0_voiced) / len(f0))
                else:
                    # 没有检测到基频，用零填充
                    features.extend([0] * 9)

            except Exception as e:
                # 基频提取失败，用零填充
                features.extend([0] * 9)

        # ========== 1. 振幅包络特征 (新增) ==========
        if "amplitude" in cfg.feature_types:
            # 计算振幅包络
            envelope = librosa.onset.onset_strength(
                y=waveform, sr=sr, hop_length=cfg.hop_length
            )

            if len(envelope) > 0:
                features.append(np.max(envelope))  # 最大振幅
                features.append(np.mean(envelope))  # 平均振幅
                features.append(np.std(envelope))  # 振幅标准差
                features.append(np.ptp(envelope))  # 峰峰值 (max-min)

                # 振幅变化率
                envelope_diff = np.diff(envelope)
                features.append(np.mean(np.abs(envelope_diff)))  # 平均变化率
            else:
                features.extend([0] * 5)

        # ========== 2. MFCC 特征 ==========
        if "mfcc" in cfg.feature_types:
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=sr,
                n_mfcc=cfg.n_mfcc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                fmin=audio_cfg.freq_min,
                fmax=audio_cfg.freq_max,
            )
            # 计算每个 MFCC 系数的均值和标准差
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)

        # ========== 2. MFCC Delta 特征 ==========
        if "mfcc_delta" in cfg.feature_types:
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=sr,
                n_mfcc=cfg.n_mfcc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                fmin=audio_cfg.freq_min,
                fmax=audio_cfg.freq_max,
            )
            # 一阶差分 - 动态调整 width 以适应短音节
            n_frames = mfcc.shape[1]

            if n_frames >= 3:
                # width 必须是奇数，且不超过帧数
                delta_width = min(9, n_frames)
                if delta_width % 2 == 0:  # 如果是偶数，减1变成奇数
                    delta_width = max(3, delta_width - 1)

                mfcc_delta = librosa.feature.delta(mfcc, width=delta_width)
                mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
                mfcc_delta_std = np.std(mfcc_delta, axis=1)
                features.extend(mfcc_delta_mean)
                features.extend(mfcc_delta_std)
            else:
                # 帧数太少，用零填充
                features.extend([0] * cfg.n_mfcc * 2)

        # ========== 3. 频谱特征 ==========
        if "spectral" in cfg.feature_types:
            # 频谱质心 (Spectral Centroid)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=waveform, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))

            # 频谱带宽 (Spectral Bandwidth)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=waveform, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))

            # 频谱滚降点 (Spectral Rolloff)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=waveform, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))

            # 频谱平坦度 (Spectral Flatness)
            spectral_flatness = librosa.feature.spectral_flatness(
                y=waveform, n_fft=cfg.n_fft, hop_length=cfg.hop_length
            )
            features.append(np.mean(spectral_flatness))
            features.append(np.std(spectral_flatness))

            # 频谱对比度 (Spectral Contrast)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    spectral_contrast = librosa.feature.spectral_contrast(
                        y=waveform,
                        sr=sr,
                        n_fft=cfg.n_fft,
                        hop_length=cfg.hop_length,
                        fmin=audio_cfg.freq_min,
                    )
                    features.extend(np.mean(spectral_contrast, axis=1))
                except:
                    # 如果频谱对比度计算失败，填充零
                    features.extend([0] * 7)

        # ========== 4. 时域特征 ==========
        if "temporal" in cfg.feature_types:
            # 持续时间
            duration = len(waveform) / sr
            features.append(duration)

            # 过零率 (Zero Crossing Rate)
            zcr = librosa.feature.zero_crossing_rate(
                waveform, hop_length=cfg.hop_length
            )
            features.append(np.mean(zcr))
            features.append(np.std(zcr))

            # RMS 能量
            rms = librosa.feature.rms(y=waveform, hop_length=cfg.hop_length)
            features.append(np.mean(rms))
            features.append(np.std(rms))

        # ========== 5. PCEN 图像特征 ==========
        if "pcen_image" in cfg.feature_types:
            # 计算梅尔频谱
            S = librosa.feature.melspectrogram(
                y=waveform,
                sr=sr,
                n_mels=cfg.n_mels,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                fmin=audio_cfg.freq_min,
                fmax=audio_cfg.freq_max,
            )

            if cfg.use_pcen:
                # PCEN 归一化 - 对低信噪比录音更鲁棒
                S_norm = librosa.pcen(
                    S * (2**31),
                    sr=sr,
                    hop_length=cfg.hop_length,
                    gain=cfg.pcen_gain,
                    bias=cfg.pcen_bias,
                    power=cfg.pcen_power,
                    time_constant=cfg.pcen_time_constant,
                )
            else:
                # 使用对数梅尔频谱
                S_norm = librosa.power_to_db(S, ref=np.max)

            # 缩放到固定尺寸
            S_resized = resize(
                S_norm,
                (cfg.n_mels, cfg.fixed_width),
                mode="reflect",
                anti_aliasing=True,
            )

            # 展平
            features.extend(S_resized.flatten())

        return np.array(features, dtype=np.float32)

    def process_syllables(
        self, syllable_df: pd.DataFrame, verbose: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        处理所有音节，提取特征

        Parameters
        ----------
        syllable_df : pd.DataFrame
            音节信息 DataFrame (来自 RavenTableReader)
        verbose : bool
            是否显示进度信息

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            (特征矩阵, 有效音节的 DataFrame)
        """
        features_list = []
        valid_indices = []

        total = len(syllable_df)

        for idx, row in syllable_df.iterrows():
            if verbose and (idx + 1) % 10 == 0:
                print(f"\r处理进度: {idx + 1}/{total}", end="", flush=True)

            try:
                # 验证持续时间
                duration = row["end_time"] - row["begin_time"]
                if duration < self.config.audio.min_syllable_duration:
                    continue
                if duration > self.config.audio.max_syllable_duration:
                    continue

                # 提取波形
                waveform = self.extract_syllable_waveform(
                    row["source_file"], row["begin_time"], row["end_time"]
                )

                # 检查波形是否有效
                if len(waveform) < self.config.feature.hop_length:
                    continue

                # 提取特征
                features = self.extract_features(waveform)

                # 检查特征是否有效 (无 NaN 或 Inf)
                if not np.isfinite(features).all():
                    continue

                features_list.append(features)
                valid_indices.append(idx)

            except Exception as e:
                if verbose:
                    print(f"\n警告: 处理音节 {idx} 时出错: {e}")
                continue

        if verbose:
            print(f"\n完成! 有效音节: {len(valid_indices)}/{total}")

        if not features_list:
            raise ValueError("没有成功提取任何音节特征!")

        features_matrix = np.array(features_list)
        valid_df = syllable_df.loc[valid_indices].copy().reset_index(drop=True)

        return features_matrix, valid_df

    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表

        Returns
        -------
        List[str]
            特征名称列表
        """
        cfg = self.config.feature
        names = []

        if "f0" in cfg.feature_types:
            names.extend(
                [
                    "f0_mean",
                    "f0_std",
                    "f0_min",
                    "f0_max",
                    "f0_range",
                    "f0_diff_mean",
                    "f0_diff_std",
                    "f0_trend",
                    "voiced_ratio",
                ]
            )

        if "amplitude" in cfg.feature_types:
            names.extend(
                [
                    "envelope_max",
                    "envelope_mean",
                    "envelope_std",
                    "envelope_ptp",
                    "envelope_diff_rate",
                ]
            )

        if "mfcc" in cfg.feature_types:
            for i in range(cfg.n_mfcc):
                names.append(f"mfcc_{i}_mean")
            for i in range(cfg.n_mfcc):
                names.append(f"mfcc_{i}_std")

        if "mfcc_delta" in cfg.feature_types:
            for i in range(cfg.n_mfcc):
                names.append(f"mfcc_delta_{i}_mean")
            for i in range(cfg.n_mfcc):
                names.append(f"mfcc_delta_{i}_std")

        if "spectral" in cfg.feature_types:
            names.extend(
                [
                    "spectral_centroid_mean",
                    "spectral_centroid_std",
                    "spectral_bandwidth_mean",
                    "spectral_bandwidth_std",
                    "spectral_rolloff_mean",
                    "spectral_rolloff_std",
                    "spectral_flatness_mean",
                    "spectral_flatness_std",
                ]
            )
            for i in range(7):
                names.append(f"spectral_contrast_{i}")

        if "temporal" in cfg.feature_types:
            names.extend(
                [
                    "duration",
                    "zcr_mean",
                    "zcr_std",
                    "rms_mean",
                    "rms_std",
                ]
            )

        if "pcen_image" in cfg.feature_types:
            for i in range(cfg.n_mels * cfg.fixed_width):
                names.append(f"pcen_pixel_{i}")

        return names
