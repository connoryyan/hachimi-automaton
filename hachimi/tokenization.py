"""分词：字符级 Tokenizer（供 transformer / markov 使用）+ BPE 工具。

本项目词表是一个封闭的小字符集（几十个汉字），默认使用字符级分词器，
每个字符即一个 token，无需 BPE 子词合并。BPE 作为可选工具保留，供实验用。
"""

from __future__ import annotations

import re
from collections import defaultdict

# 特殊 token 顺序固定，放在词表最前面
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Tokenizer:
    """字符级分词器。

    Args:
        alphabet: 合法字符集（字符串或字符列表）。重复字符会被去重，保持首次出现顺序。
    """

    def __init__(self, alphabet: str | list[str]):
        chars = list(dict.fromkeys(alphabet))
        self.itos: list[str] = [*SPECIAL_TOKENS, *chars]
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        """总词表大小（含特殊 token）。"""
        return len(self.itos)

    @property
    def vocab(self) -> list[str]:
        """去掉特殊 token 的普通字符表。"""
        return self.itos[len(SPECIAL_TOKENS):]

    def encode(self, text: str) -> list[int]:
        """字符串 → token id 列表（未知字符映射到 <unk>）。"""
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        """token id 列表 → 字符串（跳过特殊 token）。"""
        return "".join(self.itos[i] for i in ids if i >= len(SPECIAL_TOKENS))


# ---------------------------------------------------------------------------
# BPE 工具（可选，供实验；修正了原始实现的导入与空对问题）
# ---------------------------------------------------------------------------

def _bpe_stats(vocab: dict[str, int]) -> defaultdict:
    """统计相邻符号对的频次。"""
    pairs: defaultdict = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def _bpe_merge_vocab(pair: tuple[str, str], v_in: dict[str, int]) -> dict[str, int]:
    """把词表中所有出现的相邻 pair 合并为一个符号。"""
    v_out: dict[str, int] = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        v_out[p.sub("".join(pair), word)] = v_in[word]
    return v_out


def train_bpe(text: str, num_merges: int = 50) -> dict:
    """在文本上执行 ``num_merges`` 次 BPE 合并。

    中文无空格分词，因此将整段文本视为单个“词”，在字符间进行合并。

    Returns:
        {"vocab": 合并后的词表, "merges": 依次执行的合并对}。
    """
    vocab = {" ".join(list(text.strip())) + " </w>": 1}
    merges: list[tuple[str, str]] = []

    for _ in range(num_merges):
        stats = _bpe_stats(vocab)
        if not stats:
            break
        best = max(stats, key=stats.get)
        merges.append(best)
        vocab = _bpe_merge_vocab(best, vocab)

    return {"vocab": vocab, "merges": merges}


if __name__ == "__main__":
    demo = "哈基米摸南北绿豆阿西噶呀库那路"
    result = train_bpe(demo, num_merges=5)
    print("BPE 合并序列:", result["merges"])
    print("最终词表:", result["vocab"])

    tok = Tokenizer("哈基米摸南北绿豆阿西噶呀库那路")
    ids = tok.encode("哈基米")
    print("encode('哈基米') =", ids)
    print("decode:", tok.decode(ids), "| vocab_size =", tok.vocab_size)
