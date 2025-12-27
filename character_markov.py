import json
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

data_path = "./data/corpus.json"

alphabet = {
    "哈基米": "哈基米摸南北绿豆阿西噶呀库那路",
    "哈基米only": "哈基米",
    "曼波": "曼波欧马吉利哇夏达不耶",
    "曼波only": "曼波"
}

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def train(allowed_classes):
    with open(data_path, "r", encoding="utf-8") as f:
        songs = json.load(f)

    allowed_chars = {}
    for cls in allowed_classes:
        for char in alphabet[cls]:
            allowed_chars[char] = True

    model = {c: {} for c in allowed_chars}
    
    for song in songs:
        text = song.get("lines", [])
        prev = None
        for ch in text:
            if not allowed_chars.get(ch):
                prev = None
                continue

            if prev:
                model[prev][ch] = model[prev].get(ch, 0) + song.get("weight", 1.0)

            prev = ch
    
    for a in model:
        total = sum(model[a].values())
        if total > 0:
            for b in model[a]:
                model[a][b] /= total
    
    return model

def generate(model, start, length=50):
    chars = list(model.keys())
    cur = start
    out = "" + cur

    for _ in range(length):
        next_probs = model[cur]
        if not next_probs:
            cur = random.choice(chars)
        else:
            nxt = random.choices(
                population=list(next_probs.keys()),
                weights=list(next_probs.values())
            )[0]
            cur = nxt
        out += cur

    return "".join(out)
