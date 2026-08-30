"""马尔科夫链训练与歌词生成。"""

import json
import random
import logging

from hachimi.config import ALPHABET, DATA_PATH

logger = logging.getLogger(__name__)


def train(allowed_classes: list[str]) -> dict[str, dict[str, float]]:
    """训练马尔科夫链模型。

    Args:
        allowed_classes: 允许的角色类别列表，如 ["哈基米", "曼波only"]。

    Returns:
        转移概率矩阵 model[char][next_char] = probability。
    """
    allowed_chars: dict[str, bool] = {}
    for cls in allowed_classes:
        for char in ALPHABET[cls]:
            allowed_chars[char] = True

    model: dict[str, dict[str, float]] = {c: {} for c in allowed_chars}

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        songs = json.load(f)

    for song in songs:
        text = song.get("lines", [])
        prev = None
        for ch in text:
            if not allowed_chars.get(ch):
                prev = None
                continue
            if prev:
                model[prev][ch] = model[prev].get(ch, 0.0) + song.get("weight", 1.0)
            prev = ch

    # 归一化为概率
    for a in model:
        total = sum(model[a].values())
        if total > 0:
            for b in model[a]:
                model[a][b] /= total

    logger.info("Markov model trained: %d states", len(model))
    return model


def generate(model: dict, start: str | None = None, length: int = 50) -> str:
    """从马尔科夫模型生成字符序列。

    Args:
        model: 转移概率矩阵。
        start: 起始字符，为 None 则随机选取。
        length: 生成长度。

    Returns:
        生成的字符序列。
    """
    chars = list(model.keys())
    if start is None or start not in model:
        cur = random.choice(chars)
    else:
        cur = start

    out = cur
    for _ in range(length - 1):
        next_probs = model.get(cur, {})
        if not next_probs:
            cur = random.choice(chars)
        else:
            cur = random.choices(
                population=list(next_probs.keys()),
                weights=list(next_probs.values()),
                k=1,
            )[0]
        out += cur

    return out
