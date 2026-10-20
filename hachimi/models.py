"""模型工厂：统一创建 / 加载 Markov 或 Transformer 歌词模型。"""

from __future__ import annotations

from hachimi.base import LyricsModel


def create_model(kind: str, allowed_classes: list[str], **kwargs) -> LyricsModel:
    """按类型创建并训练一个歌词模型。

    Args:
        kind: ``"markov"`` 或 ``"transformer"``。
        allowed_classes: 允许的歌词集（如 ``["哈基米"]``）。
        **kwargs: 传给对应模型的超参数（transformer 的 block_size/n_layer 等）。

    Returns:
        已训练好的 ``LyricsModel`` 实例。
    """
    if kind == "markov":
        from hachimi.markov import MarkovModel
        return MarkovModel().train(allowed_classes)

    if kind == "transformer":
        from hachimi.transformer import TransformerModel
        return TransformerModel(**kwargs).train(allowed_classes)

    raise ValueError(f"未知模型类型: {kind!r}（可选 'markov' / 'transformer'）")


def load_model(kind: str, path: str) -> LyricsModel:
    """从磁盘加载已保存的歌词模型（不重新训练）。

    Args:
        kind: ``"markov"``（.json）或 ``"transformer"``（.npz）。
        path: 模型文件路径。

    Returns:
        加载好的 ``LyricsModel`` 实例，可直接 ``generate`` / ``sample``。
    """
    if kind == "markov":
        from hachimi.markov import MarkovModel
        return MarkovModel.load(path)

    if kind == "transformer":
        from hachimi.transformer import TransformerModel
        return TransformerModel.load(path)

    raise ValueError(f"未知模型类型: {kind!r}（可选 'markov' / 'transformer'）")


__all__ = ["create_model", "load_model", "LyricsModel"]
