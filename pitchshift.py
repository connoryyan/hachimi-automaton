import numpy as np
import pyworld as pw
import soundfile as sf
import pretty_midi
import os
import librosa
from tqdm import tqdm


def synthesize_note(preproc, target_pitch, dur_samples, sr=44100, frame_period=5.0, ap_scale=0.05, sp_scale=1.0):
    channels_out = []
    for ch in range(2):
        orig = preproc[ch]

        f0_orig = orig['f0']
        sp_orig = orig['sp'] * sp_scale
        ap_orig = orig['ap'] * ap_scale

        is_vowel = orig['is_vowel']

        # -------------- 调音 --------------
        target_f0 = 440.0 * 2 ** ((target_pitch - 69) / 12)
        voiced = f0_orig[f0_orig > 0]

        if len(voiced) > 0:
            f0_mean = np.mean(voiced)
        else:
            f0_mean = target_f0

        f0_new = f0_orig.copy()
        f0_new[f0_orig > 0] += (target_f0 - f0_mean)

        vowel_idx = np.where(is_vowel)[0]
        cons_idx  = np.where(~is_vowel)[0]

        f0_v = f0_new[vowel_idx]
        sp_v = sp_orig[vowel_idx]
        ap_v = ap_orig[vowel_idx]

        f0_u = f0_new[cons_idx]
        sp_u = sp_orig[cons_idx]
        ap_u = ap_orig[cons_idx]

        # -------------- 拉伸 --------------
        dur_ms = dur_samples / sr * 1000
        total_frames_target = int(dur_ms / frame_period)

        # 不拉伸辅音，保持原本帧数
        n_u = len(f0_u)

        # 剩余帧全部给元音
        n_v_target = max(1, total_frames_target - n_u)

        # 计算插值目标索引
        if len(f0_v) > 1:
            idx_v = np.linspace(0, len(f0_v)-1, n_v_target)
        else:
            idx_v = np.zeros(n_v_target)

        if len(f0_u) > 1:
            idx_u = np.linspace(0, len(f0_u)-1, n_u)
        else:
            idx_u = np.zeros(n_u)

        def interp_matrix(mat, idx):
            if mat.ndim == 1:
                return np.interp(idx, np.arange(len(mat)), mat)
            return np.vstack([
                np.interp(idx, np.arange(mat.shape[0]), mat[:, i])
                for i in range(mat.shape[1])
            ]).T

        f0_v_new = np.interp(idx_v, np.arange(len(f0_v)), f0_v)
        sp_v_new = interp_matrix(sp_v, idx_v)
        ap_v_new = interp_matrix(ap_v, idx_v)

        f0_u_new = f0_u
        sp_u_new = sp_u
        ap_u_new = ap_u

        f0_final = np.concatenate([f0_u_new, f0_v_new], axis=0)
        sp_final = np.concatenate([sp_u_new, sp_v_new], axis=0)
        ap_final = np.concatenate([ap_u_new, ap_v_new], axis=0)

        y_ch = pw.synthesize(f0_final.astype(np.float64),
                             sp_final.astype(np.float64),
                             ap_final.astype(np.float64),
                             sr, frame_period)

        # pad 或裁剪到目标长度（样本数）
        if len(y_ch) > dur_samples:
            y_ch = y_ch[:dur_samples]
        else:
            y_ch = np.pad(y_ch, (0, dur_samples - len(y_ch)))

        channels_out.append(y_ch)

    y = np.stack(channels_out, axis=1)
    return y

def preprocess_sample(sample_path, cons_frame, sr=44100, frame_period=5.0):
    x, orig_sr = sf.read(sample_path)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.shape[1] == 1:
        x = np.concatenate([x, x], axis=1)

    if orig_sr != sr:
        x = np.stack([librosa.resample(x[:, ch], orig_sr=orig_sr, target_sr=sr) for ch in range(2)], axis=1)

    preproc = []
    for ch in range(2):
        channel = x[:, ch].copy()
        f0, t = pw.harvest(channel, sr, frame_period=frame_period)
        sp = pw.cheaptrick(channel, f0, t, sr)
        ap = pw.d4c(channel, f0, t, sr)

        is_vowel = np.ones_like(f0, dtype=bool)
        is_consonant = np.zeros_like(f0, dtype=bool)

        buffer = cons_frame

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
        
        preproc.append({'f0': f0, 'sp': sp, 'ap': ap, 't': t, 'is_vowel': is_vowel, 'f0_v': f0_v, 'sp_v': sp_v, 'ap_v': ap_v,
                        'f0_u': f0_u, 'sp_u': sp_u, 'ap_u': ap_u})
    return preproc

# sample_path = "./assets/samples/哈.wav"
# midi_path   = "./assets/midi/tchop35a/tchop35a.mid"
# out_path    = "./output/test2.wav"
# sr = 44100
# frame_period = 5.0

# preproc = preprocess_sample(sample_path, sr=sr, frame_period=frame_period)
# midi = pretty_midi.PrettyMIDI(midi_path)

# octave_shift = [0, 0, 1]
# volume_factor = [1.0, 0.8, 0.6]

# y_out = synthesize_midi(preproc, midi, sr=sr, frame_period=frame_period,
#                         start_time=0.0, end_time=2000.0,
#                         octave_shift=octave_shift,
#                         volume_factor=volume_factor,
#                         ap_scale=0.05, sp_scale=1.1)

# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# sf.write(out_path, y_out, sr)
# print(f"已生成: {out_path}")
