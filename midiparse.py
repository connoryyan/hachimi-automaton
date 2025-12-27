import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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
    "Telephone Ring", "Helicopter", "Applause", "Gunshot"
]

folder_path = "./assets/midi/haydnop76/"
midi_path = folder_path + "haydn_string_quartet_76_3_score_1_(c)unknown.mid"

pm = pretty_midi.PrettyMIDI(midi_path)

print("===== Instrument Info =====")
for i, inst in enumerate(pm.instruments):
    if inst.is_drum:
        instr_name = "Drum Kit"
    else:
        instr_name = GM_PROGRAMS[inst.program] if 0 <= inst.program < 128 else f"Program {inst.program}"
    print(f"Track {i}: name = {inst.name}, "
          f"program = {inst.program} ({instr_name}), "
          f"is_drum = {inst.is_drum}, "
          f"notes = {len(inst.notes)}")
print("===========================\n")

def plot_piano_roll():
    for i, inst in enumerate(pm.instruments):
        piano_roll = inst.get_piano_roll(fs=50)

        if np.max(piano_roll) > 0:
            img = (piano_roll / np.max(piano_roll) * 255).astype(np.uint8)
        else:
            img = piano_roll.astype(np.uint8)

        plt.figure(figsize=(100, 10))
        plt.imshow(img, aspect='auto', origin='lower', cmap='gray')
        plt.tight_layout()
        out_file = folder_path + f"pianoroll_track{i}_{GM_PROGRAMS[inst.program].replace(' ', '_')}.jpg"
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved piano roll for track {i}: {out_file}")

def audio_preview():
    out_path=folder_path + "preview.wav"
    fs=44100
    audio = pm.fluidsynth(fs=fs, sf2_path="C:/soundfonts/FluidR3_GM.sf2")
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wavfile.write(out_path, fs, audio_int16)
    print("Audio saved", out_path)
    for i, inst in enumerate(pm.instruments):
        single_pm = pretty_midi.PrettyMIDI()
        single_pm.instruments.append(inst)
        audio = single_pm.fluidsynth(fs=fs)
        audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
        
        if inst.is_drum:
            name_safe = f"track{i}_drum.wav"
        else:
            name_safe = f"track{i}_{GM_PROGRAMS[inst.program].replace(' ', '_')}.wav"
        
        out_path = folder_path + "preview_" + name_safe
        wavfile.write(out_path, fs, audio_int16)
        print("Saved track:", out_path)

plot_piano_roll()
audio_preview()
