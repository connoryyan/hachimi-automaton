"""MIDI 合成：将 MIDI 音符渲染为合成人声音频。"""

import numpy as np
import logging
from tqdm import tqdm

import pretty_midi

from hachimi.synthesis import synthesize_note
from hachimi.config import SR, FRAME_PERIOD

logger = logging.getLogger(__name__)


def synthesize_midi(
    preproc: dict[str, list[dict]],
    note2word: list[dict],
    midi: pretty_midi.PrettyMIDI,
    sr: int = SR,
    frame_period: float = FRAME_PERIOD,
    start_time: float = 0.0,
    end_time: float | None = None,
    octave_shift: list[int] | None = None,
    volume_factor: list[float] | None = None,
    ap_scale: float = 0.05,
    sp_scale: float = 1.0,
) -> np.ndarray:
    """将 MIDI 全部轨道合成为立体声音频。

    Args:
        preproc: {word → preprocess_sample 结果}。
        note2word: 每个轨道的 {note → word} 映射。
        midi: PrettyMIDI 对象。
        sr: 采样率。
        frame_period: 帧周期（ms）。
        start_time: 开始时间（秒）。
        end_time: 结束时间（秒），None = MIDI 总长。
        octave_shift: 每个轨道的八度偏移（半音）。
        volume_factor: 每个轨道的音量系数。
        ap_scale: 非周期分量缩放。
        sp_scale: 频谱包络缩放。

    Returns:
        (n_samples, 2) 立体声音频。
    """
    n_tracks = len(midi.instruments)
    if octave_shift is None:
        octave_shift = [0] * n_tracks
    if volume_factor is None:
        volume_factor = [1.0] * n_tracks

    total_time = midi.get_end_time() if end_time is None else end_time
    n_samples = int(total_time * sr)
    y_stereo = np.zeros((n_samples, 2))

    valid_notes = []
    for idx, inst in enumerate(midi.instruments):
        if volume_factor[idx] == 0:
            continue
        for note in inst.notes:
            if note.end > start_time and note.start < total_time:
                valid_notes.append(note)

    pbar = tqdm(total=len(valid_notes), desc="Rendering MIDI")

    for idx, instrument in enumerate(midi.instruments):
        shift = octave_shift[idx] * 12
        vol = volume_factor[idx]

        if vol == 0:
            continue

        for note in instrument.notes:
            if note.end <= start_time or note.start >= total_time:
                continue

            word = note2word[idx].get(note)
            if word is None or word not in preproc:
                continue

            pbar.update(1)

            start = max(int((note.start - start_time) * sr), 0)
            end = min(int((note.end - start_time) * sr), n_samples)
            dur = end - start
            if dur <= 0:
                continue

            target_pitch = note.pitch + shift

            y_note = synthesize_note(
                preproc[word],
                target_pitch,
                dur,
                sr=sr,
                frame_period=frame_period,
                ap_scale=ap_scale,
                sp_scale=sp_scale,
            )
            y_stereo[start:end, :] += y_note * vol

    pbar.close()

    max_amp = np.max(np.abs(y_stereo))
    if max_amp > 1.0:
        y_stereo /= max_amp

    return y_stereo
