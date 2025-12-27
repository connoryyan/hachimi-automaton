from character_markov import *
from pitchshift import *
from reverb import *
from animate import *
import pretty_midi
import mido
import soundfile as sf
import os

midi_path = "./assets/midi/tchop35a/tchop35a.mid"
out_path = "./output/tchop35a/tchop35a.wav"
sample_path = "./assets/samples/"
out_lyrics_path = "./output/tchop35a/tchop35a_lyrics.json"
out_model_path = "./models/character_markov/"

track_classes = ["哈基米", "哈基米only", "曼波only"]

SIMULTANEOUS_THRESHOLD = 0.05

start_time = 0
end_time = 60

sr = 44100
frame_period = 5.0

octave_shift = [0, 0, 2]
volume_factor = [1, 0.6, 0.7]

cons_frames = {
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

def generate_lyrics(midi, models, start_time=0.0, end_time=None):
    """
    midi: pretty_midi.PrettyMIDI 对象
    models: list of model 对象，每个轨道对应一个 model
    返回:
        track_words: List[List[str]] 每轨词序列
        note2word: List[Dict[note, word]] 每轨音符到词的映射
    """
    note2word = []
    track_words_all = []

    valid_notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            if note.end > start_time:
                if end_time is None or note.start < end_time:
                    valid_notes.append(note)

    pbar = tqdm(total=len(valid_notes), desc="编写歌词中：")

    for idx, inst in enumerate(midi.instruments):
        
        notes = sorted(inst.notes, key=lambda n: (n.start, n.end))
        if not notes:
            note2word.append({})
            track_words_all.append([])
            continue

        # 分组
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

        # 当前轨道使用的模型
        model = models[idx] if idx < len(models) else models[0]
        if not model:
            note2word.append({})
            track_words_all.append([])
            continue

        # 生成歌词
        current_token = random.choice(list(model.keys()))
        mapping = {}
        track_words = []

        for group in groups:
            word = current_token
            track_words.append(word)
            for note in group:
                pbar.update(1)
                mapping[note] = word
            # 下一个词
            next_options = model.get(current_token, {})
            if next_options:
                current_token = random.choices(list(next_options.keys()), weights=list(next_options.values()), k=1)[0]
            else:
                current_token = random.choice(list(model.keys()))

        note2word.append(mapping)
        track_words_all.append(track_words)

    return track_words_all, note2word

def synthesize_midi(preproc, note2word, midi, sr=44100, frame_period=5.0, start_time=0.0, end_time=60, octave_shift=None, volume_factor=None, ap_scale=0.05, sp_scale=1.0):
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
            if note.end > start_time and note.start < end_time:
                valid_notes.append(note)

    pbar = tqdm(total=len(valid_notes), desc="Rendering MIDI")

    for idx, instrument in enumerate(midi.instruments):
        shift = octave_shift[idx] * 12
        vol = volume_factor[idx]

        if vol == 0:
            continue;

        for note in instrument.notes:
            if note.end <= start_time or note.start >= total_time:
                continue
            
            pbar.update(1)
            start = max(int((note.start-start_time)*sr), 0)
            end   = min(int((note.end-start_time)*sr), n_samples)
            dur   = end - start
            if dur <= 0:
                continue

            target_pitch = note.pitch + shift
            y_note = synthesize_note(preproc[note2word[idx][note]], target_pitch, dur,
                                     sr=sr, frame_period=frame_period,
                                     ap_scale=ap_scale, sp_scale=sp_scale)
            y_stereo[start:end, :] += y_note * vol

    max_amp = np.max(np.abs(y_stereo))
    if max_amp > 1.0:
        y_stereo /= max_amp

    return y_stereo

def build_events(note2word):
    """
    note2word: List[Dict[pretty_midi.Note, str]]

    return:
        events: List[Dict]
    """
    events = []

    for track_idx, track_map in enumerate(note2word):
        for note, word in track_map.items():
            events.append({
                "start": float(note.start),
                "end": float(note.end),
                "state": word,
                "track": track_idx,
                "pitch": int(note.pitch)
            })

    # 按时间排序（非常重要）
    events.sort(key=lambda e: (e["start"], e["end"]))

    return events

midi = pretty_midi.PrettyMIDI(midi_path)

track_models = []

for cls in track_classes:
    model = train(allowed_classes=[cls])
    track_models.append(model)
    draw(model, save_path=out_model_path + f"{cls}.png")
    with open(f"{out_model_path}{cls}.json", "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

track_words, note2word = generate_lyrics(midi, track_models)

events = build_events(note2word)
with open(out_lyrics_path, "w", encoding="utf-8") as f:
    json.dump(events, f, ensure_ascii=False, indent=2)

# preproc = {}
# for cls in track_classes:
#     for sample in alphabet[cls]:
#         if sample in preproc:
#             continue
#         path = sample_path + f"{sample}.wav"
#         preproc[sample] = preprocess_sample(path, cons_frame=cons_frames.get(sample, 8))

# y_out = synthesize_midi(preproc, note2word, midi, start_time=start_time, end_time=end_time, octave_shift = octave_shift, volume_factor=volume_factor, ap_scale=0.05, sp_scale=1.0)

# y_out = vocal_low_cut(y_out, sr)
# y_out = stereo_spread(y_out.T, spread_amount=0.15).T
# y_out = multitap_delay(y_out, sr)
# y_out = hf_damping(y_out, sr, cutoff=10000)

# y_out = soft_normalize(y_out, 0.95)
# sf.write(out_path, y_out, sr)
# print(f"音频已保存至 {out_path}")

