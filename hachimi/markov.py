"""马尔科夫链歌词模型（实现统一 LyricsModel 接口）。"""

from __future__ import annotations

import json
import logging
import random

from hachimi.base import LyricsModel, build_allowed_chars, load_songs

logger = logging.getLogger(__name__)


class MarkovModel(LyricsModel):
    """一阶字符马尔科夫链。

    训练得到 ``model[char][next_char] = probability`` 的转移概率矩阵。
    """

    def __init__(self) -> None:
        self._model: dict[str, dict[str, float]] = {}

    def train(self, allowed_classes: list[str]) -> "MarkovModel":
        allowed_chars = {c: True for c in build_allowed_chars(allowed_classes)}
        model: dict[str, dict[str, float]] = {c: {} for c in allowed_chars}

        for song in load_songs():
            text = song.get("lines", "")
            prev: str | None = None
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

        self._model = model
        logger.info("Markov model trained: %d states", len(model))
        return self

    @property
    def vocab(self) -> list[str]:
        return list(self._model.keys())

    def sample(self, context: str = "") -> str:
        chars = list(self._model.keys())
        if not chars:
            return ""

        # 空上下文 → 从先验中随机取一个起点字符
        cur = context[-1] if context else random.choice(chars)

        next_probs = self._model.get(cur, {})
        if not next_probs:
            return random.choice(chars)
        return random.choices(
            population=list(next_probs.keys()),
            weights=list(next_probs.values()),
            k=1,
        )[0]

    def generate(self, start: str | None = None, length: int = 50) -> str:
        out = "" if start is None else str(start)
        while len(out) < length:
            out += self.sample(out)
        return out[:length]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return self._model

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._model, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 兼容旧函数式接口（main.py 旧代码、外部脚本）
# ---------------------------------------------------------------------------

def train(allowed_classes: list[str]) -> dict[str, dict[str, float]]:
    """训练并返回转移概率矩阵（兼容旧接口）。"""
    return MarkovModel().train(allowed_classes).to_dict()


def generate(model: dict, start: str | None = None, length: int = 50) -> str:
    """从转移概率矩阵生成字符序列（兼容旧接口）。"""
    m = MarkovModel()
    m._model = model
    return m.generate(start, length)
