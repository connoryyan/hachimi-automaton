"""歌词生成：将 MIDI 音符映射为马尔科夫链生成的字符序列。"""

import logging

import pretty_midi
import numpy as np
from tqdm import tqdm

from hachimi.base import LyricsModel
from hachimi.config import SIMULTANEOUS_THRESHOLD, START_TIME

logger = logging.getLogger(__name__)


def generate_lyrics(
    midi: pretty_midi.PrettyMIDI,
    models: list[LyricsModel | None],
    start_time: float = START_TIME,
    end_time: float | None = None,
) -> tuple[list[list[str]], list[dict]]:
    """根据 MIDI 和歌词模型为每个音符分配字符。

    Args:
        midi: PrettyMIDI 对象。
        models: 每个轨道的歌词模型（Markov 或 Transformer，None 表示静音）。
        start_time: 开始时间。
        end_time: 结束时间（None = MIDI 总长）。

    Returns:
        (track_words_all, note2word)
        - track_words_all: 每个轨道的字符序列。
        - note2word: 每个轨道的 {note → word} 映射。
    """
    note2word: list[dict] = []
    track_words_all: list[list[str]] = []

    valid_notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            if note.end > start_time and (end_time is None or note.start < end_time):
                valid_notes.append(note)

    pbar = tqdm(total=len(valid_notes), desc="编写歌词中")

    for idx, inst in enumerate(midi.instruments):
        notes = sorted(inst.notes, key=lambda n: (n.start, n.end))
        if not notes:
            note2word.append({})
            track_words_all.append([])
            continue

        # 和弦分组
        groups = []
        current_group = [notes[0]]
        current_start = notes[0].start
        for note in notes[1:]:
            if note.start - current_start <= SIMULTANEOUS_THRESHOLD:
                current_group.append(note)
            else:
                groups.append(current_group)
                current_group = [note]
                current_start = note.start
        groups.append(current_group)

        model = models[idx] if idx < len(models) else models[0]
        if not model:
            note2word.append({})
            track_words_all.append([])
            continue

        context = ""
        mapping: dict = {}
        track_words: list[str] = []

        for group in groups:
            word = model.sample(context)
            track_words.append(word)
            main_note = max(group, key=lambda n: n.pitch)
            pbar.update(len(group))
            mapping[main_note] = word

            # 累积上文（Markov 只看最后一位；Transformer 内部会截断到 block_size）
            context = (context + word)[-256:]

        note2word.append(mapping)
        track_words_all.append(track_words)

    pbar.close()
    return track_words_all, note2word


def build_events(note2word: list[dict]) -> list[dict]:
    """从 note2word 构建时间排序的事件列表。

    Args:
        note2word: 每个轨道的 {note → word} 映射。

    Returns:
        按 start 排序的事件列表。
    """
    events = []
    for track_idx, track_map in enumerate(note2word):
        for note, word in track_map.items():
            events.append({
                "start": float(note.start),
                "end": float(note.end),
                "state": word,
                "track": track_idx,
                "pitch": int(note.pitch),
            })
    events.sort(key=lambda e: (e["start"], e["end"]))
    return events


def sec_to_srt(t: float) -> str:
    """秒数转换为 SRT 的时间戳格式 (HH:MM:SS,mmm)。"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_track_srt(
    note2word: list[dict],
    track_idx: int,
    output_path: str,
    reset_gap: float = 1.0,
    max_duration: float = 2.5,
    max_len: int = 12,
) -> None:
    """为指定轨道生成 SRT 字幕文件。

    Args:
        note2word: 每个轨道的 {note → word} 映射。
        track_idx: 轨道索引。
        output_path: 输出 .srt 路径。
        reset_gap: 超过此间隔（秒）则断句。
        max_duration: 字幕最大显示时长（秒）。
        max_len: 字幕最大字符数。
    """
    if track_idx >= len(note2word) or not note2word[track_idx]:
        logger.warning("SRT: 轨道 %d 不存在或为空，跳过", track_idx)
        return

    items = [
        {"start": n.start, "end": n.end, "word": w}
        for n, w in note2word[track_idx].items()
    ]
    items.sort(key=lambda x: x["start"])

    subtitles = []
    text = ""
    seg_start = None
    prev_t = None
    idx = 1

    for i, it in enumerate(items):
        start = it["start"]
        word = it["word"]

        if prev_t is not None and start - prev_t > reset_gap:
            text = ""
            seg_start = None

        if seg_start is None:
            seg_start = start

        text += word
        if len(text) > max_len:
            text = word
            seg_start = start

        if i + 1 < len(items):
            next_start = items[i + 1]["start"]
            end_time = min(next_start, start + max_duration)
        else:
            end_time = start + max_duration

        if end_time - seg_start < 0.08:
            end_time = seg_start + 0.08

        subtitles.append((idx, seg_start, end_time, text))
        idx += 1
        prev_t = start

    with open(output_path, "w", encoding="utf-8-sig") as f:
        for i, s, e, t in subtitles:
            f.write(f"{i}\n{sec_to_srt(s)} --> {sec_to_srt(e)}\n{t}\n\n")

    logger.info("SRT written: %s (%d subtitles)", output_path, len(subtitles))
