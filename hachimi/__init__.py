"""哈基米音乐自动生成工具 (Hachimi Automaton).

歌词生成模型统一入口：::

    from hachimi import create_model, MarkovModel, TransformerModel

    # 两种模型共用同一接口，可互换：
    model = create_model("transformer", allowed_classes=["哈基米"])
    model = create_model("markov", allowed_classes=["曼波only"])

    model.generate(length=50)   # 生成 50 个字符
    model.sample("哈基米")      # 采样下一个字符
"""

__version__ = "0.3.0"

from hachimi.tokenization import Tokenizer, train_bpe
from hachimi.base import LyricsModel, load_songs
from hachimi.markov import MarkovModel
from hachimi.transformer import TransformerModel
from hachimi.models import create_model

__all__ = [
    "Tokenizer",
    "train_bpe",
    "LyricsModel",
    "load_songs",
    "MarkovModel",
    "TransformerModel",
    "create_model",
]
