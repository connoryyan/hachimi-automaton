"""哈基米音乐自动生成工具 — 主入口。

将 MIDI 文件转成带歌词的哈基米音乐：
  1. 交互式选择 MIDI、歌词生成模型和轨道配置
  2. 用统一接口的歌词模型（马尔科夫 / Transformer）为每个音符生成歌词字符
  3. 用 WORLD vocoder 合成人声音频
  4. 输出 WAV 音频 + JSON 歌词 + SRT 字幕
"""

import json
import os
import logging

import pretty_midi

from hachimi.base import LyricsModel
from hachimi.config import (
    SAMPLE_PATH,
    OUTPUT_DIR,
    MODEL_DIR,
    CONS_FRAMES,
    TRANSFORMER,
    SR,
    FRAME_PERIOD,
    START_TIME,
    END_TIME_BUFFER,
    AP_SCALE,
    SP_SCALE,
    LOW_CUT_FREQ,
    STEREO_SPREAD,
    MULTITAP_DECAY,
    MULTITAP_DELAY_MS,
    MULTITAP_TAPS,
    HF_DAMPING_CUTOFF,
)
from hachimi.models import create_model, load_model
from hachimi.synthesis import preprocess_sample
from hachimi.audio import vocal_low_cut, stereo_spread, multitap_delay, hf_damping
from hachimi.lyrics import generate_lyrics, build_events, make_track_srt
from hachimi.midi import synthesize_midi
from hachimi.cli import select_midi, select_model_kind, configure_tracks
from hachimi.animate import draw_markov_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # ---- 1. 选择 MIDI 与歌词生成模型 ----
    midi_path = select_midi()
    kind = select_model_kind()
    project_name = os.path.splitext(os.path.basename(midi_path))[0]
    output_dir = os.path.join(OUTPUT_DIR, project_name)
    os.makedirs(output_dir, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(midi_path)

    # ---- 2. 显示轨道信息 ----
    n_tracks = len(midi.instruments)
    print(f"\n成功加载 MIDI: {midi_path}")
    print(f"检测到 {n_tracks} 个轨道:")
    for i, inst in enumerate(midi.instruments):
        name = inst.name or "未命名轨道"
        notes_count = len(inst.notes)
        drum = " (打击乐)" if inst.is_drum else ""
        print(f"  [{i}] {name} - {notes_count} 个音符{drum}")

    # ---- 3. 轨道配置 ----
    instrument_names = [inst.name for inst in midi.instruments]
    is_drum_flags = [inst.is_drum for inst in midi.instruments]
    track_classes, octave_shift, volume_factor = configure_tracks(
        n_tracks, instrument_names, is_drum_flags
    )

    # ---- 4. 加载或训练歌词模型（每个类别只处理一次；已有缓存则直接加载） ----
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_cache: dict[str, LyricsModel] = {}
    track_models: list[LyricsModel] = []

    for cls in track_classes:
        if cls not in model_cache:
            if kind == "markov":
                model_path = os.path.join(MODEL_DIR, f"{cls}.json")
                if os.path.exists(model_path):
                    model = load_model("markov", model_path)
                    logger.info("加载已有模型 [markov] %s", cls)
                else:
                    model = create_model("markov", allowed_classes=[cls])
                    model.save(model_path)
                    draw_markov_graph(
                        model.to_dict(), save_path=os.path.join(MODEL_DIR, f"{cls}.png")
                    )
            else:
                model_path = os.path.join(MODEL_DIR, f"{cls}.npz")
                if os.path.exists(model_path):
                    model = load_model("transformer", model_path)
                    logger.info("加载已有模型 [transformer] %s", cls)
                else:
                    model = create_model("transformer", allowed_classes=[cls], **TRANSFORMER)
                    model.save(model_path)
            model_cache[cls] = model
        track_models.append(model_cache[cls])

    # ---- 5. 生成歌词 ----
    midi_duration = midi.get_end_time()
    end_time = midi_duration + END_TIME_BUFFER
    print(f"音频长度预计: {end_time:.2f} 秒")

    track_words, note2word = generate_lyrics(midi, track_models, start_time=START_TIME)

    events = build_events(note2word)
    out_lyrics_path = os.path.join(output_dir, f"{project_name}_lyrics.json")
    with open(out_lyrics_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    # ---- 6. 预处理采样 ----
    preproc = {}
    for cls in track_classes:
        from hachimi.config import ALPHABET
        for sample in ALPHABET[cls]:
            if sample in preproc:
                continue
            path = os.path.join(SAMPLE_PATH, f"{sample}.wav")
            preproc[sample] = preprocess_sample(path, cons_frame=CONS_FRAMES.get(sample, 8))

    # ---- 7. 合成音频 ----
    y_out = synthesize_midi(
        preproc,
        note2word,
        midi,
        start_time=START_TIME,
        end_time=end_time,
        octave_shift=octave_shift,
        volume_factor=volume_factor,
        ap_scale=AP_SCALE,
        sp_scale=SP_SCALE,
    )

    # ---- 8. 后处理 ----
    y_out = vocal_low_cut(y_out, SR, cutoff=LOW_CUT_FREQ)
    y_out = stereo_spread(y_out.T, spread_amount=STEREO_SPREAD).T
    y_out = multitap_delay(y_out, SR, decay=MULTITAP_DECAY, delay_ms=MULTITAP_DELAY_MS, num_taps=MULTITAP_TAPS)
    y_out = hf_damping(y_out, SR, cutoff=HF_DAMPING_CUTOFF)

    # ---- 9. 生成 SRT 字幕（选择第一个可听且有歌词的轨道） ----
    srt_track_idx = None
    for idx, (words_map, vol) in enumerate(zip(note2word, volume_factor)):
        if words_map and vol > 0:
            srt_track_idx = idx
            break
    if srt_track_idx is None:
        for idx, words_map in enumerate(note2word):
            if words_map:
                srt_track_idx = idx
                break

    if srt_track_idx is not None:
        make_track_srt(
            note2word,
            track_idx=srt_track_idx,
            output_path=os.path.join(output_dir, f"track{srt_track_idx}.srt"),
            reset_gap=1.0,
            max_len=12,
        )
    else:
        logger.warning("没有可生成字幕的轨道，跳过 SRT")

    out_path = os.path.join(output_dir, f"{project_name}.wav")
    print(f"\n音频已生成: {out_path}")
    print(f"歌词已保存: {out_lyrics_path}")


if __name__ == "__main__":
    main()
