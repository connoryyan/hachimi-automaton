import numpy as np
import pyworld as pw
import soundfile as sf
import os

# ------------------------
# CONFIG
# ------------------------

cons_frame = {
    "哈": 3,
    "基": 8,
    "米": 8,
    "摸": 8,
    "南": 8,
    "北": 8,
    "绿": 8,
    "豆": 8,
    "阿": 8,
    "西": 30,
    "噶": 8,
    "呀": 8,
    "库": 8,
    "那": 8,
    "路": 8,
    "曼": 8,
    "波": 8
}

sample_path = "./assets/samples/波.wav"
out_path    = "./output/preview_pitchshift.wav"
out_vowel   = "./output/preview_vowel.wav"
out_consonant = "./output/preview_consonant.wav"

sr = 44100
frame_period = 10
target_midi = 36
target_sec  = 3

buffer = 8  # 前几帧归辅音

def midi_to_hz(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)

target_f0 = midi_to_hz(target_midi)

# ------------------------
# load audio
# ------------------------
x, orig_sr = sf.read(sample_path)
if x.ndim == 1:
    x = np.stack([x, x], axis=1)
elif x.shape[1] == 1:
    x = np.concatenate([x, x], axis=1)

if orig_sr != sr:
    import librosa
    x = np.stack([librosa.resample(x[:, ch], orig_sr=orig_sr, target_sr=sr) for ch in range(2)], axis=1)

# ------------------------
# WORLD synthesis
# ------------------------
n_samples = int(target_sec * sr)
y_stereo = np.zeros((n_samples, 2))

for ch in range(2):
    channel = x[:, ch].copy()

    f0, t = pw.harvest(channel, sr, frame_period=frame_period)
    sp = pw.cheaptrick(channel, f0, t, sr) * 1
    ap = pw.d4c(channel, f0, t, sr) * 0.05

    # 假设 voiced 全部为元音
    is_vowel = np.ones_like(f0, dtype=bool)
    is_consonant = np.zeros_like(f0, dtype=bool)

    if len(f0) > buffer:
        # 前 buffer 帧归辅音
        is_vowel[:buffer] = False
        is_consonant[:buffer] = True

    f0_v = f0[is_vowel]
    sp_v = sp[is_vowel]
    ap_v = ap[is_vowel]

    f0_u = f0[is_consonant]
    sp_u = sp[is_consonant]
    ap_u = ap[is_consonant]

    # ------------------------
    # 导出元音 / 辅音的声音
    # ------------------------
    # 只导出第一个声道
    if ch == 0:
        y_vowel = pw.synthesize(
            f0_v,
            sp_v,
            ap_v,
            sr,
            frame_period=frame_period
        )
        y_cons = pw.synthesize(
            f0_u if len(f0_u) > 0 else np.array([0]),
            sp_u if len(f0_u) > 0 else sp[:1],
            ap_u if len(f0_u) > 0 else ap[:1],
            sr,
            frame_period=frame_period
        )

        os.makedirs(os.path.dirname(out_vowel), exist_ok=True)
        sf.write(out_vowel, y_vowel, sr)
        sf.write(out_consonant, y_cons, sr)
        print("已导出元音:", out_vowel)
        print("已导出辅音:", out_consonant)

    n_v = len(f0_v)
    n_u = len(f0_u)
    total_target_frames = int(target_sec * 1000 / frame_period)

    target_v_frames = max(total_target_frames - n_u, 5)
    target_u_frames = n_u

    if n_v > 1:
        idx_v = np.linspace(0, n_v-1, target_v_frames)
        f0_v2 = np.interp(idx_v, np.arange(n_v), f0_v)
        sp_v2 = np.array([np.interp(idx_v, np.arange(n_v), sp_v[:, i]) for i in range(sp_v.shape[1])]).T
        ap_v2 = np.array([np.interp(idx_v, np.arange(n_v), ap_v[:, i]) for i in range(ap_v.shape[1])]).T
    else:
        f0_v2 = f0_v
        sp_v2 = sp_v
        ap_v2 = ap_v

    f0_final = np.concatenate([f0_u, f0_v2])
    sp_final = np.concatenate([sp_u, sp_v2], axis=0)
    ap_final = np.concatenate([ap_u, ap_v2], axis=0)

    f0_final = np.ascontiguousarray(f0_final)
    sp_final = np.ascontiguousarray(sp_final)
    ap_final = np.ascontiguousarray(ap_final)

    y = pw.synthesize(f0_final, sp_final, ap_final, sr, frame_period=frame_period)

    y_stereo[:, ch] = y[:n_samples] if len(y) >= n_samples else np.pad(y, (0, n_samples - len(y)))

# ------------------------
# 保存主合成结果
# ------------------------
os.makedirs(os.path.dirname(out_path), exist_ok=True)
sf.write(out_path, y_stereo, sr)
print("已生成:", out_path)
