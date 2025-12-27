import librosa
import soundfile as sf
import numpy as np
import pretty_midi
import random
from scipy.signal import butter, lfilter

sample_path = "./assets/samples/米.wav"
midi_path = "./assets/midi/haydnop76/haydn_string_quartet_76_3_score_1_(c)unknown.mid"
out_path = "./output/test.wav"

start_time = 0 # seconds
end_time = 120.0

octave_shift = [0,0,0]
factor = [1,1,1]

def pitch_shift_formant_lock(y, sr, steps):
    if steps == 0:
        return y

    # 1. 原声的 mel 包络
    S = np.abs(librosa.stft(y, n_fft=1024))
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_fft=1024)
    mel_env = librosa.feature.inverse.mel_to_stft(mel, sr=sr, n_fft=1024)

    # 2. 普通 pitch shift（音高变但音色乱）
    y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    # 3. 再取频谱并替换 envelope
    S2 = np.abs(librosa.stft(y2, n_fft=1024))
    S2_corrected = S2 * (mel_env / (np.abs(S) + 1e-6))
    y_final = librosa.istft(S2_corrected * np.exp(1j*np.angle(librosa.stft(y2))))

    return y_final

def estimate_pitch(audio, sr, frame_length=2048, hop_length=256):
    f0 = librosa.yin(
        audio,
        fmin=50,
        fmax=2000,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )

    f0 = f0[np.isfinite(f0)]

    if len(f0) == 0:
        return None

    f0 = f0[(f0 > 50) & (f0 < 3000)]
    if len(f0) == 0:
        return None

    median = np.median(f0)
    lower = median * 0.8
    upper = median * 1.2
    stable_f0 = f0[(f0 > lower) & (f0 < upper)]

    if len(stable_f0) == 0:
        stable_f0 = f0
        
    mean_f0 = np.mean(stable_f0)
    midi = librosa.hz_to_midi(mean_f0)

    return midi

def stable_time_stretch(y, sr, dur_samples, segment=0.18):
    if dur_samples <= len(y):
        return y[:dur_samples]

    seg_len = int(segment * sr)

    pieces = []
    pos = 0
    while len(np.concatenate(pieces)) < dur_samples:
        end = min(pos + seg_len, len(y))
        chunk = y[pos:end]

        # 随机相位扰动（减少金属声）
        stft = librosa.stft(chunk)
        mag, phase = np.abs(stft), np.angle(stft)
        phase += np.random.uniform(-0.05, 0.05, phase.shape)
        chunk = librosa.istft(mag * np.exp(1j*phase))

        pieces.append(chunk)
        pos = random.randint(0, max(1, len(y)-seg_len))

    out = np.concatenate(pieces)[:dur_samples]
    return out

def apply_jitter(y, amount=0.015):
    t = np.linspace(0, len(y)/44100, len(y))
    jitter = 1 + amount * np.sin(2*np.pi * random.uniform(3,7) * t)
    return y * jitter[:len(y)]

def precompute_pitch_shifts(sample, sr, midi, detune_steps=None):
    pitches = set()
    for instrument in midi.instruments:
        for note in instrument.notes:
            pitches.add(note.pitch)

    steps = set()

    for p in pitches:
        step = p - sample_pitch
        if detune_steps is not None:
            for detune in detune_steps:
                steps.add(step + detune)

    pitch_cache = {}
    for step in steps:
        pitch_cache[step] = librosa.effects.pitch_shift(y=sample, sr=sr, n_steps=step)

    return pitch_cache

def soft_limiter(audio, threshold=0.95):
    audio_clipped = np.copy(audio)
    over = np.abs(audio_clipped) > threshold
    audio_clipped[over] = np.sign(audio_clipped[over]) * threshold + \
                          (audio_clipped[over] - np.sign(audio_clipped[over]) * threshold) * 0.1
    return audio_clipped

def lowpass_filter(audio, sr, cutoff=6000, order=4):
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def tiny_time_shift(audio, sr, max_shift_ms=15):
    shift_samples = int(random.uniform(0, max_shift_ms / 1000 * sr))
    return np.pad(audio, (shift_samples, 0), mode='constant')[:len(audio)]


sample, sr = librosa.load(sample_path, sr=None)
midi = pretty_midi.PrettyMIDI(midi_path)
sample_pitch = estimate_pitch(sample, sr)
# sample_pitch = 70.26350551031037
print("Estimated sample pitch (MIDI):", sample_pitch)

for i in range(len(octave_shift)):
    semitone_shift = octave_shift[i] * 12
    for note in midi.instruments[i].notes:
        note.pitch += semitone_shift

pitch_cache = precompute_pitch_shifts(
    sample, sr,
    midi,
    detune_steps=[-0.02, 0, 0.02]
)

audio_length = int((end_time - start_time) * sr)
output_track = []
output = np.zeros(audio_length)
for instrument in midi.instruments:
    output_track.append(np.zeros(audio_length))

# ----------------------- render note --------------------------
for i in range(len(output_track)):
    # notes = midi.instruments[i].notes
    # pitch_map = np.full(audio_length, -np.inf)

    for note in midi.instruments[i].notes:
        if note.start < start_time or note.end > end_time:
            continue

        start = int((note.start - start_time) * sr)
        end   = int((note.end - start_time) * sr)
        dur   = end - start

        detune = 0.0
        steps = note.pitch - sample_pitch + detune

        shifted = pitch_shift_formant_lock(sample, sr, steps)

        stretched = stable_time_stretch(shifted, sr, dur)

        stretched = apply_jitter(stretched, amount=0.01)

        ratio = len(shifted) / max(dur, 10)
        stretched = librosa.effects.time_stretch(y=shifted, rate=ratio)[:dur]
        stretched = lowpass_filter(stretched, sr, cutoff=6000)
        stretched = tiny_time_shift(stretched, sr, max_shift_ms=10)

        # for l in range(dur):
        #     t = start + l
        #     if note.pitch > pitch_map[t]:
        #         pitch_map[t] = note.pitch
        #         output_track[i][t] = stretched[l]

        output_track[i][start:start + len(stretched)] += stretched

    output_track[i] /= np.max(np.abs(output_track[i])) + 1e-6
    output_track[i] *= factor[i]
    output += output_track[i]


output /= np.max(np.abs(output)) + 1e-6
sf.write(out_path, output, sr)
