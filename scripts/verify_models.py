"""模型与分词验证脚本（从项目根目录运行：python scripts/verify_models.py）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hachimi.tokenization import Tokenizer, train_bpe
from hachimi.base import build_allowed_chars, iter_allowed_texts
from hachimi.markov import MarkovModel
from hachimi.transformer import TransformerModel, GPT, _cross_entropy, grad_check
from hachimi.models import create_model


def test_tokenizer():
    tok = Tokenizer("哈基米摸南北绿豆阿西噶呀库那路")
    ids = tok.encode("哈基米")
    assert tok.decode(ids) == "哈基米", tok.decode(ids)
    assert tok.vocab_size == 4 + 15  # 4 特殊 + 15 字符
    print("[tokenizer] OK  encode('哈基米')=", ids, " vocab_size=", tok.vocab_size)

    res = train_bpe("哈基米摸南北绿豆阿西噶呀库那路", num_merges=5)
    print("[bpe] OK  merges=", res["merges"])


def test_grad_check():
    gpt = GPT(vocab_size=6, block_size=6, n_embd=8, n_head=2, n_layer=1, seed=0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 6, size=(1, 4))
    targets = rng.integers(0, 6, size=(1, 4))
    err = grad_check(gpt, idx, targets)
    print(f"[grad_check] max relative error = {err:.2e}")
    assert err < 1e-2, f"梯度校验失败: {err}"


def test_markov():
    m = MarkovModel().train(["哈基米"])
    assert set(m.vocab) == set("哈基米摸南北绿豆阿西噶呀库那路")
    out = m.generate(length=40)
    assert len(out) == 40 and set(out) <= set(m.vocab)
    print("[markov] OK  示例:", out)
    return m


def test_transformer():
    m = TransformerModel(block_size=24, n_embd=32, n_head=4, n_layer=2,
                         steps=800, batch_size=16, lr=2e-3, seed=0)
    m.train(["哈基米"])
    out = m.generate(length=40)
    assert len(out) == 40
    print("[transformer] OK  示例:", out)
    print("            vocab_size =", m.gpt.vocab_size, "params =", sum(p.size for p in m.gpt.flat_params()))
    return m


def test_factory():
    mk = create_model("markov", ["曼波only"])
    tf = create_model("transformer", ["哈基米only"], block_size=16, n_embd=16,
                      n_head=2, n_layer=1, steps=300, batch_size=16, lr=2e-3)
    print("[factory] OK  markov(曼波only):", mk.generate(length=20))
    print("                 transformer(哈基米only):", tf.generate(length=20))


if __name__ == "__main__":
    test_tokenizer()
    test_grad_check()
    test_markov()
    test_transformer()
    test_factory()
    print("\n全部验证通过 ✔")
