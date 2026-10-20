"""歌词生成模型统一接口与数据加载。

``MarkovModel`` 与 ``TransformerModel`` 都实现 ``LyricsModel`` 接口，
从而可以在歌词生成流程中无缝互换。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Iterator

from hachimi.config import ALPHABET, DATA_PATH


class LyricsModel(ABC):
    """歌词生成模型的统一接口。"""

    @abstractmethod
    def train(self, allowed_classes: list[str]) -> "LyricsModel":
        """用 ``allowed_classes`` 对应的歌词集训练，返回 self。"""

    @abstractmethod
    def sample(self, context: str = "") -> str:
        """给定上文 ``context``，采样并返回下一个字符。"""

    @abstractmethod
    def generate(self, start: str | None = None, length: int = 50) -> str:
        """自回归生成 ``length`` 个字符。"""

    @property
    @abstractmethod
    def vocab(self) -> list[str]:
        """可生成的字符集合。"""

    def save(self, path: str) -> None:  # pragma: no cover - 子类按需覆写
        raise NotImplementedError

    def __bool__(self) -> bool:
        return True


def load_songs() -> list[dict]:
    """加载语料库（``data/corpus.json``）。"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_allowed_chars(allowed_classes: list[str]) -> list[str]:
    """返回 ``allowed_classes`` 合并后的去重字符表（保持顺序）。"""
    chars: list[str] = []
    seen: set[str] = set()
    for cls in allowed_classes:
        for c in ALPHABET[cls]:
            if c not in seen:
                seen.add(c)
                chars.append(c)
    return chars


def iter_allowed_texts(allowed_classes: list[str]) -> Iterator[tuple[str, float]]:
    """产出 ``(text, weight)``，text 已过滤为 ``allowed_classes`` 内字符。

    过滤掉长度不足 2 的片段（无法构成前后文）。
    """
    allowed = set(build_allowed_chars(allowed_classes))
    for song in load_songs():
        text = "".join(c for c in song.get("lines", "") if c in allowed)
        if len(text) >= 2:
            yield text, float(song.get("weight", 1.0))
