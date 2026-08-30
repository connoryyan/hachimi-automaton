"""音频后处理效果：延迟、混响、均衡、声场展开。"""

import numpy as np
from scipy.signal import fftconvolve, butter, lfilter


def multitap_delay(
    y: np.ndarray,
    sr: int = 44100,
    decay: float = 0.4,
    delay_ms: float = 50,
    num_taps: int = 5,
) -> np.ndarray:
    """多级延迟效果，产生柔和回声。

    Args:
        y: (N, 2) 立体声音频。
        sr: 采样率。
        decay: 每级衰减。
        delay_ms: 回声间隔（ms）。
        num_taps: 回声级数。

    Returns:
        (N, 2) 处理后的音频。
    """
    output = y.copy()
    delay_samples = int(sr * delay_ms / 1000.0)

    for i in range(1, num_taps + 1):
        atten = decay ** i
        shift = delay_samples * i

        if shift >= len(y):
            break

        echo = np.zeros_like(y)
        echo[shift:] = y[:-shift] * atten
        output += echo

    return output


def vocal_low_cut(
    audio: np.ndarray,
    sr: int = 44100,
    cutoff: float = 80,
    order: int = 4,
) -> np.ndarray:
    """高通滤波，切除人声低频。

    Args:
        audio: (N,) 或 (N, 2) 音频。
        sr: 采样率。
        cutoff: 截止频率（Hz）。
        order: 滤波器阶数。

    Returns:
        滤波后的音频。
    """
    nyq = sr * 0.5
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, audio)


def stereo_spread(
    stereo_audio: np.ndarray,
    spread_amount: float = 0.2,
) -> np.ndarray:
    """Mid/Side 声场展开。

    Args:
        stereo_audio: (2, N) 立体声。
        spread_amount: 展开量（0~1）。

    Returns:
        (2, N) 展开后的立体声。
    """
    L, R = stereo_audio
    mid = (L + R) * 0.5
    side = (L - R) * 0.5
    side *= (1 + spread_amount)
    return np.vstack([mid + side, mid - side])


def hf_damping(
    audio: np.ndarray,
    sr: int = 44100,
    cutoff: float = 6000,
) -> np.ndarray:
    """高频衰减（低通滤波）。

    Args:
        audio: (N,) 或 (N, 2) 音频。
        sr: 采样率。
        cutoff: 截止频率（Hz）。

    Returns:
        滤波后的音频。
    """
    nyq = sr * 0.5
    norm_cutoff = cutoff / nyq
    b, a = butter(2, norm_cutoff, btype="low")
    return lfilter(b, a, audio)


def cathedral_reverb(
    audio: np.ndarray,
    sr: int = 44100,
    mix: float = 0.25,
    decay: float = 3.5,
) -> np.ndarray:
    """大教堂风格长混响。

    Args:
        audio: (N,) 或 (N, 2) 音频。
        sr: 采样率。
        mix: 干湿比（0~1）。
        decay: 混响尾音长度（秒）。

    Returns:
        混响后的音频。
    """
    ir_len = int(sr * decay)
    t = np.linspace(0, decay, ir_len)
    ir = np.exp(-t * 5)
    ir += 0.3 * np.random.randn(ir_len)

    nyq = sr * 0.5
    cutoff = 5000 / nyq
    b, a = butter(1, cutoff, btype="low")
    ir = lfilter(b, a, ir)
    ir /= np.max(np.abs(ir) + 1e-9)

    mono_input = audio.ndim == 1
    if mono_input:
        audio = audio[:, None]

    n_channels = audio.shape[1]
    wet = np.zeros_like(audio)
    for ch in range(n_channels):
        wet[:, ch] = fftconvolve(audio[:, ch], ir, mode="full")[: len(audio)]

    out = (1 - mix) * audio + mix * wet
    if mono_input:
        out = out[:, 0]
    return out


def soft_normalize(
    audio: np.ndarray,
    target_peak: float = 0.9,
    clip: bool = True,
) -> np.ndarray:
    """温和归一化：sqrt 增益保持动态，防止削波。

    Args:
        audio: (N,) 或 (N, 2) 音频。
        target_peak: 目标峰值。
        clip: 是否硬削波。

    Returns:
        归一化后的音频。
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio

    gain = target_peak / peak
    soft_gain = gain ** 0.5
    audio = audio * soft_gain

    if clip:
        audio = np.clip(audio, -1.0, 1.0)
    return audio
