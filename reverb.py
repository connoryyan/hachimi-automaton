import numpy as np
from scipy.signal import fftconvolve, butter, lfilter

def multitap_delay(y, sr=44100, decay=0.4, delay_ms=50, num_taps=5):
    """
    为 stereo 音频添加柔和人声混响（轻混音）
    
    y: (N,2) stereo numpy array
    sr: 采样率
    decay: 每次回声衰减
    delay_ms: 每级回声间隔（毫秒）
    num_taps: 回声级数（越高越空旷）
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

import numpy as np

def soft_convolution(y, sr=44100, decay=0.3, delay_ms=60):
    """
    为音频添加柔和人声混响（简单卷积方式）
    
    y: 音频 (numpy array) shape = (n_samples, 2) 或 (n_samples,)
    sr: 采样率
    decay: 衰减系数（0.1 ~ 0.6）值越大混响越明显
    delay_ms: 延迟（毫秒），人声适合 30~100ms

    return:
        y_out: 添加混响后的音频
    """

    if y.ndim == 1:
        y = y[:, None]  # 转单声道为两声道视作 shape(N,1)

    # 计算延迟样本数
    delay_samples = int(sr * delay_ms / 1000)

    # 构造脉冲响应（IR）—— 简单指数衰减
    ir_length = delay_samples * 6     # 混响长度 = 约 6 次延迟
    ir = np.zeros(ir_length)
    for i in range(6):
        ir[i * delay_samples] = (decay ** i)

    # 对每个声道卷积
    y_out = np.zeros((len(y) + ir_length - 1, y.shape[1]))
    for ch in range(y.shape[1]):
        y_out[:, ch] = np.convolve(y[:, ch], ir)

    return y_out

def vocal_low_cut(audio, sr, cutoff=80, order=4):
    """
    High-pass filter for vocal low-frequency rolloff.
    cutoff: 高频截止（Hz）
    """
    nyq = sr * 0.5
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio)

def stereo_spread(stereo_audio, spread_amount=0.2):
    """
    stereo_audio: shape (2, N)
    spread_amount: 扩散比例 0~1
    """
    L, R = stereo_audio

    mid = (L + R) * 0.5
    side = (L - R) * 0.5

    side *= (1 + spread_amount)  # 加宽

    L_new = mid + side
    R_new = mid - side

    return np.vstack([L_new, R_new])

def hf_damping(audio, sr, cutoff=6000):
    """
    高频混响衰减（Low-pass filter）
    cutoff: 高频滚降点
    """
    nyq = sr * 0.5
    norm_cutoff = cutoff / nyq
    b, a = butter(2, norm_cutoff, btype='low')
    return lfilter(b, a, audio)

def cathedral_reverb(audio, sr, mix=0.25, decay=3.5):
    """
    Cathedral-style long reverb, 支持 mono 或 stereo。
    
    audio: (N,) 或 (N,2)
    sr: 采样率
    mix: 干湿比例（0~1）
    decay: 尾音长度（秒），越大越像大教堂
    """

    # -------------------
    # 1. 生成简易 IR（脉冲响应）
    # -------------------
    ir_len = int(sr * decay)
    t = np.linspace(0, decay, ir_len)

    # 指数衰减
    ir = np.exp(-t * 5)

    # 加入随机扩散（多级反射）
    ir += 0.3 * np.random.randn(ir_len)

    # 高频轻微衰减（空气吸收）
    nyq = sr * 0.5
    cutoff = 5000 / nyq
    b, a = butter(1, cutoff, btype='low')
    ir = lfilter(b, a, ir)

    # 归一化 IR
    ir /= np.max(np.abs(ir) + 1e-9)

    # -------------------
    # 2. 确保 audio 是二维 (N, C)
    # -------------------
    mono_input = False
    if audio.ndim == 1:
        audio = audio[:, None]  # 转成 (N,1)
        mono_input = True

    n_channels = audio.shape[1]
    wet = np.zeros_like(audio)

    # -------------------
    # 3. 对每个声道卷积
    # -------------------
    for ch in range(n_channels):
        wet[:, ch] = fftconvolve(audio[:, ch], ir, mode='full')[:len(audio)]

    # -------------------
    # 4. 干湿混合
    # -------------------
    out = (1 - mix) * audio + mix * wet

    # -------------------
    # 5. 如果原本是 mono，转回一维
    # -------------------
    if mono_input:
        out = out[:, 0]

    return out

def soft_normalize(audio, target_peak=0.9, clip=True):
    """
    温和正规化音频：
    - 保持动态
    - 最大值接近 target_peak
    - clip=True 时防止溢出

    audio: (N,) 或 (N,2)
    target_peak: 最大允许峰值
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio  # 全零音频

    gain = target_peak / peak

    # 使用 sqrt(gain) 缓和动态变化
    soft_gain = gain ** 0.5
    audio = audio * soft_gain

    if clip:
        audio = np.clip(audio, -1.0, 1.0)

    return audio