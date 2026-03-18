from character_markov import *
from pitchshift import *
from reverb import *
from animate import draw
import pretty_midi
import mido
import soundfile as sf
import os

import glob

def select_midi():
    midi_files = sorted(glob.glob("./assets/midi/**/*.mid", recursive=True))
    if not midi_files:
        print("未在 assets/midi 中找到 MIDI 文件。")
        exit(1)
    
    print("\n找到以下 MIDI 文件：")
    for i, path in enumerate(midi_files):
        print(f"[{i}] {path}")
    
    while True:
        try:
            choice = int(input("\n请输入要处理的 MIDI 文件编号: "))
            if 0 <= choice < len(midi_files):
                return midi_files[choice]
        except ValueError:
            pass
        print("输入无效，请输入正确的数字。")

midi_path = select_midi()
# 获取不带扩展名的文件名作为项目名
project_name = os.path.splitext(os.path.basename(midi_path))[0]
output_dir = f"./output/{project_name}/"
os.makedirs(output_dir, exist_ok=True)

out_path = os.path.join(output_dir, f"{project_name}.wav")
sample_path = "./assets/samples/"
out_lyrics_path = os.path.join(output_dir, f"{project_name}_lyrics.json")
out_model_path = "./models/character_markov/"
os.makedirs(out_model_path, exist_ok=True)

midi = pretty_midi.PrettyMIDI(midi_path)

# --- 自动匹配轨道信息 ---
n_tracks = len(midi.instruments)
print(f"\n成功加载 MIDI: {midi_path}")
print(f"检测到 {n_tracks} 个轨道:")

for i, inst in enumerate(midi.instruments):
    name = inst.name if inst.name else "未命名轨道"
    notes_count = len(inst.notes)
    is_drum = " (打击乐)" if inst.is_drum else ""
    print(f"  [{i}] {name} - {notes_count} 个音符{is_drum}")

# --- 轨道配置交互 ---
available_classes = ["哈基米", "哈基米only", "曼波only"]
print("\n可选角色:")
for i, cls in enumerate(available_classes):
    print(f"  [{i}] {cls}")

track_classes = []
octave_shift = []
volume_factor = []

print("\n--- 请配置各轨道参数 (直接回车使用默认值) ---")
for i, inst in enumerate(midi.instruments):
    name = inst.name if inst.name else "未命名轨道"
    is_drum = inst.is_drum
    
    if is_drum:
        print(f"\n轨道 [{i}] {name}: 检测为打击乐，默认静音。")
        track_classes.append(available_classes[0])
        octave_shift.append(0)
        volume_factor.append(0.0)
        continue

    print(f"\n配置轨道 [{i}] {name}:")
    
    # 选择角色
    def_cls_idx = i % len(available_classes)
    cls_input = input(f"  选择角色编号 (0-{len(available_classes)-1}, 默认 {def_cls_idx} [{available_classes[def_cls_idx]}]): ")
    if cls_input.strip() == "":
        track_classes.append(available_classes[def_cls_idx])
    else:
        try:
            idx = int(cls_input)
            track_classes.append(available_classes[idx if 0 <= idx < len(available_classes) else def_cls_idx])
        except:
            track_classes.append(available_classes[def_cls_idx])

    # 八度偏移
    shift_input = input("  八度偏移 (例如 -1, 0, 1, 默认 0): ")
    try:
        octave_shift.append(int(shift_input) if shift_input.strip() != "" else 0)
    except:
        octave_shift.append(0)

    # 音量
    vol_input = input("  音量比例 (0.0-1.0, 默认 1.0): ")
    try:
        volume_factor.append(float(vol_input) if vol_input.strip() != "" else 1.0)
    except:
        volume_factor.append(1.0)

SIMULTANEOUS_THRESHOLD = 0.05

# 自动确定结束时间（MIDI 长度 + 2秒缓冲）
start_time = 0
midi_duration = midi.get_end_time()
end_time = midi_duration + 2.0 

print(f"音频长度预计: {end_time:.2f} 秒")

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

preproc = {}
for cls in track_classes:
    for sample in alphabet[cls]:
        if sample in preproc:
            continue
        path = sample_path + f"{sample}.wav"
        preproc[sample] = preprocess_sample(path, cons_frame=cons_frames.get(sample, 8))

y_out = synthesize_midi(preproc, note2word, midi, start_time=start_time, end_time=end_time, octave_shift = octave_shift, volume_factor=volume_factor, ap_scale=0.05, sp_scale=1.0)

y_out = vocal_low_cut(y_out, sr)
y_out = stereo_spread(y_out.T, spread_amount=0.15).T
y_out = multitap_delay(y_out, sr)
y_out = hf_damping(y_out, sr, cutoff=10000)

y_out = soft_normalize(y_out, 0.95)
sf.write(out_path, y_out, sr)
print(f"音频已保存至 {out_path}")

