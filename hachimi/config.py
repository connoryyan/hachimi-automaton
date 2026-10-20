"""全局配置常量。"""

# ---- 路径 ----
SAMPLE_PATH = "./assets/samples/"
MIDI_GLOB = "./assets/midi/**/*.mid"
OUTPUT_DIR = "./output/"
MODEL_DIR = "./models/character_markov/"
DATA_PATH = "./data/corpus.json"
ANIMATION_DIR = "./assets/animation/"

# ---- 字符集 ----
ALPHABET: dict[str, str] = {
    "哈基米": "哈基米摸南北绿豆阿西噶呀库那路",
    "哈基米only": "哈基米",
    "曼波": "曼波欧马吉利哇夏达不耶",
    "曼波only": "曼波",
}

# ---- 辅音帧数（元音/辅音分离参数） ----
CONS_FRAMES: dict[str, int] = {
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
    "波": 8,
}

# ---- 角色名称 ----
# 注意：ALPHABET 中的 "曼波"（含 欧马吉利哇夏达不耶）因缺少对应采样音频，
# 暂不放入可合成列表；此处仅保留有采样支持的类别。
AVAILABLE_CLASSES = ["哈基米", "哈基米only", "曼波only"]

# ---- 歌词生成模型 ----
MODEL_KIND = "markov"  # "markov" 或 "transformer"

# ---- Transformer 模型参数 ----
TRANSFORMER = {
    "block_size": 32,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 3,
    "steps": 1500,
    "batch_size": 32,
    "lr": 1e-3,
    "temperature": 1.0,
    "seed": 1337,
}

# ---- 默认轨道配置 ----
# 索引对应 MIDI 轨道号
DEFAULT_TRACK_CLASSES = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,   # 0-8 静音
    0,                              # 9  Piano
    2,                              # 10 Violin I Synth
    2,                              # 11 Violin I Pizz
    2,                              # 12 Violin I Main
    2,                              # 13 Violin II Synth
    2,                              # 14 Violin II Main
    2,                              # 15 Violin II Pizz
    0, 0, 0, 0, 0, 0, 0, 0,        # 16-23 静音
]

DEFAULT_OCTAVE_SHIFT = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1,    # Piano
    -1,   # Vln I Synth
    -1,   # Vln I Pizz
    -1,   # Vln I Main
    -1,   # Vln II Synth
    -1,   # Vln II Main
    -1,   # Vln II Pizz
    0, 0, 0, 0, 0, 0, 0, 0,
]

DEFAULT_VOLUME_FACTOR = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.7,  # Piano
    1,    # Vln I Synth
    1,    # Vln I Pizz
    1,    # Vln I Main
    0,    # Vln II Synth
    0,    # Vln II Main
    0,    # Vln II Pizz
    0, 0, 0, 0, 0, 0, 0, 0,
]

# ---- 合成参数 ----
SR = 44100
FRAME_PERIOD = 5.0
SIMULTANEOUS_THRESHOLD = 0.05
START_TIME = 500
END_TIME_BUFFER = 2.0

# AP / SP 缩放
AP_SCALE = 0.05
SP_SCALE = 1.0

# ---- 混响参数 ----
LOW_CUT_FREQ = 80
STEREO_SPREAD = 0.15
MULTITAP_DECAY = 0.4
MULTITAP_DELAY_MS = 50
MULTITAP_TAPS = 5
HF_DAMPING_CUTOFF = 10000

# ---- GM 乐器名称 ----
GM_PROGRAMS = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "SynthStrings 1", "SynthStrings 2",
    "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "SynthBrass 1", "SynthBrass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bag pipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]
