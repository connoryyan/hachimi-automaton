"""WORLD vocoder 语音合成：元音/辅音分离、基频变换、时长拉伸。"""

import numpy as np
import pyworld as pw
import soundfile as sf
import librosa
import logging

from hachimi.config import SR, FRAME_PERIOD

logger = logging.getLogger(__name__)


def synthesize_note(
    preproc: list[dict],
    target_pitch: int,
    dur_samples: int,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
    ap_scale: float = 0.05,
    sp_scale: float = 1.0,
) -> np.ndarray:
    """用 WORLD 合成单个音符，支持音高变换和时长拉伸。

    Args:
        preproc: preprocess_sample() 的返回值，包含 f0/sp/ap/is_vowel。
        target_pitch: 目标 MIDI 音高。
        dur_samples: 目标时长（采样点数）。
        sr: 采样率。
        frame_period: 帧周期（ms）。
        ap_scale: 非周期分量缩放。
        sp_scale: 频谱包络缩放。

    Returns:
        shape (dur_samples, 2) 的立体声音频。
    """
    channels_out = []
    for ch in range(2):
        orig = preproc[ch]
        f0_orig = orig["f0"]
        sp_orig = orig["sp"] * sp_scale
        ap_orig = orig["ap"] * ap_scale
        is_vowel = orig["is_vowel"]

        # 目标基频
        target_f0 = 440.0 * 2 ** ((target_pitch - 69) / 12)
        voiced = f0_orig[f0_orig > 0]
        f0_mean = np.mean(voiced) if len(voiced) > 0 else target_f0
        f0_new = f0_orig.copy()
        f0_new[f0_orig > 0] += target_f0 - f0_mean

        vowel_idx = np.where(is_vowel)[0]
        cons_idx = np.where(~is_vowel)[0]

        f0_v, sp_v, ap_v = f0_new[vowel_idx], sp_orig[vowel_idx], ap_orig[vowel_idx]
        f0_u, sp_u, ap_u = f0_new[cons_idx], sp_orig[cons_idx], ap_orig[cons_idx]

        # 时长拉伸
        dur_ms = dur_samples / sr * 1000
        total_frames_target = int(dur_ms / frame_period)
        n_u = len(f0_u)
        n_v_target = max(1, total_frames_target - n_u)

        if len(f0_v) > 1:
            idx_v = np.linspace(0, len(f0_v) - 1, n_v_target)
        else:
            idx_v = np.zeros(n_v_target)

        if len(f0_u) > 1:
            idx_u = np.linspace(0, len(f0_u) - 1, n_u)
        else:
            idx_u = np.zeros(n_u)

        f0_v_new = np.interp(idx_v, np.arange(len(f0_v)), f0_v)
        sp_v_new = _interp_matrix(sp_v, idx_v)
        ap_v_new = _interp_matrix(ap_v, idx_v)

        f0_final = np.concatenate([f0_u, f0_v_new], axis=0)
        sp_final = np.concatenate([sp_u, sp_v_new], axis=0)
        ap_final = np.concatenate([ap_u, ap_v_new], axis=0)

        y_ch = pw.synthesize(
            f0_final.astype(np.float64),
            sp_final.astype(np.float64),
            ap_final.astype(np.float64),
            sr,
            frame_period,
        )

        # pad 或裁剪
        if len(y_ch) > dur_samples:
            y_ch = y_ch[:dur_samples]
        else:
            y_ch = np.pad(y_ch, (0, dur_samples - len(y_ch)))

        channels_out.append(y_ch)

    return np.stack(channels_out, axis=1)


def _interp_matrix(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """沿第 0 轴插值矩阵。"""
    if mat.ndim == 1:
        return np.interp(idx, np.arange(len(mat)), mat)
    return np.vstack([
        np.interp(idx, np.arange(mat.shape[0]), mat[:, i])
        for i in range(mat.shape[1])
    ]).T


def preprocess_sample(
    sample_path: str,
    cons_frame: int = 8,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
) -> list[dict]:
    """预处理采样：加载音频 → WORLD 分析 → 元音/辅音分离。

    Args:
        sample_path: 音频文件路径。
        cons_frame: 前 N 帧归辅音。
        sr: 目标采样率。
        frame_period: 帧周期（ms）。

    Returns:
        list[dict] 每个声道的 WORLD 参数及元音/辅音掩码。
    """
    x, orig_sr = sf.read(sample_path)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.shape[1] == 1:
        x = np.concatenate([x, x], axis=1)

    if orig_sr != sr:
        x = np.stack([
            librosa.resample(x[:, ch], orig_sr=orig_sr, target_sr=sr)
            for ch in range(2)
        ], axis=1)

    preproc = []
    for ch in range(2):
        channel = x[:, ch].copy().astype(np.float64)
        f0, t = pw.harvest(channel, sr, frame_period=frame_period)
        sp = pw.cheaptrick(channel, f0, t, sr)
        ap = pw.d4c(channel, f0, t, sr)

        is_vowel = np.ones_like(f0, dtype=bool)
        is_consonant = np.zeros_like(f0, dtype=bool)
        buffer = cons_frame

        if len(f0) > buffer:
            is_vowel[:buffer] = False
            is_consonant[:buffer] = True

        preproc.append({
            "f0": f0,
            "sp": sp,
            "ap": ap,
            "t": t,
            "is_vowel": is_vowel,
            "f0_v": f0[is_vowel],
            "sp_v": sp[is_vowel],
            "ap_v": ap[is_vowel],
            "f0_u": f0[is_consonant],
            "sp_u": sp[is_consonant],
            "ap_u": ap[is_consonant],
        })

    return preproc
