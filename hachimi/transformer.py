import logging
import random

import numpy as np

from hachimi.base import LyricsModel, build_allowed_chars, iter_allowed_texts
from hachimi.tokenization import SPECIAL_TOKENS, Tokenizer

logger = logging.getLogger(__name__)

_BLOCK_KEYS = [
    "ln1_g", "ln1_b", "c_attn_w", "c_attn_b", "c_proj_w", "c_proj_b",
    "ln2_g", "ln2_b", "mlp_w1", "mlp_b1", "mlp_w2", "mlp_b2",
]


# ---------------------------------------------------------------------------
# 数学工具
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _gelu_backward(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    u = x + 0.044715 * x ** 3
    t = np.tanh(c * u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3 * 0.044715 * x ** 2)


def _ln_forward(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    sigma = np.sqrt(var + eps)
    xhat = (x - mu) / sigma
    y = g * xhat + b
    return y, (xhat, sigma)


def _ln_backward(dy: np.ndarray, xhat: np.ndarray, sigma: np.ndarray, g: np.ndarray):
    dxhat = dy * g
    mean_dxhat = dxhat.mean(axis=-1, keepdims=True)
    mean_dxhat_xhat = (dxhat * xhat).mean(axis=-1, keepdims=True)
    dx = (1.0 / sigma) * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat)

    axes = tuple(range(dy.ndim - 1))
    dg = (dy * xhat).sum(axis=axes)
    db = dy.sum(axis=axes)
    return dx, dg, db


def _cross_entropy(logits: np.ndarray, targets: np.ndarray):
    """返回 (loss, dlogits)。logits (B,T,V)，targets (B,T)。"""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    t_flat = targets.reshape(-1)

    probs = _softmax(logits_flat, axis=-1)
    nll = -np.log(probs[np.arange(len(t_flat)), t_flat] + 1e-12)
    loss = float(nll.mean())

    d = probs.copy()
    d[np.arange(len(t_flat)), t_flat] -= 1.0
    d /= len(t_flat)
    return loss, d.reshape(B, T, V)


class GPT:
    """一个简单的GPT类Transformer。仅实现前向、反向和参数遍历，不含训练循环。"""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        seed: int = 1337,
    ):
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_layer = n_layer

        rng = np.random.default_rng(seed)
        self.params = self._init_params(rng)
        self.cache: dict | None = None

    # ---- 初始化 ----
    def _init_params(self, rng) -> dict:
        C, V, T, L = self.n_embd, self.vocab_size, self.block_size, self.n_layer

        def normal(shape, std: float = 0.02):
            return (rng.standard_normal(shape) * std).astype(np.float32)

        residual_std = 0.02 / np.sqrt(2 * L)

        params = {
            "wte": normal((V, C)),
            "wpe": normal((T, C)),
            "blocks": [],
            "lnf_g": np.ones(C, dtype=np.float32),
            "lnf_b": np.zeros(C, dtype=np.float32),
        }
        for _ in range(L):
            blk = {
                "ln1_g": np.ones(C, dtype=np.float32),
                "ln1_b": np.zeros(C, dtype=np.float32),
                "c_attn_w": normal((C, 3 * C)),
                "c_attn_b": np.zeros(3 * C, dtype=np.float32),
                "c_proj_w": normal((C, C), std=residual_std),
                "c_proj_b": np.zeros(C, dtype=np.float32),
                "ln2_g": np.ones(C, dtype=np.float32),
                "ln2_b": np.zeros(C, dtype=np.float32),
                "mlp_w1": normal((C, 4 * C)),
                "mlp_b1": np.zeros(4 * C, dtype=np.float32),
                "mlp_w2": normal((4 * C, C), std=residual_std),
                "mlp_b2": np.zeros(C, dtype=np.float32),
            }
            params["blocks"].append(blk)
        return params

    # ---- 前向 ----
    def forward(self, idx: np.ndarray) -> np.ndarray:
        B, T = idx.shape
        C, H, D = self.n_embd, self.n_head, self.head_dim
        p = self.params

        pos = np.arange(T, dtype=np.int64)
        x_emb = p["wte"][idx] + p["wpe"][pos]  # (B,T,C)
        x = x_emb

        cache_blocks = []
        for blk in p["blocks"]:
            x_in = x

            # --- 因果自注意力 ---
            h, (xhat1, sigma1) = _ln_forward(x, blk["ln1_g"], blk["ln1_b"])
            qkv = h @ blk["c_attn_w"] + blk["c_attn_b"]  # (B,T,3C)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B,H,T,D)
            k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D)  # (B,H,T,T)
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask[None, None], -1e9, scores)
            p_att = _softmax(scores, axis=-1)

            out = p_att @ v                                    # (B,H,T,D)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
            attn_out = out @ blk["c_proj_w"] + blk["c_proj_b"]
            x = x + attn_out
            x_mid = x

            # --- MLP ---
            h2, (xhat2, sigma2) = _ln_forward(x, blk["ln2_g"], blk["ln2_b"])
            mlp_pre = h2 @ blk["mlp_w1"] + blk["mlp_b1"]
            m = _gelu(mlp_pre)
            mlp_out = m @ blk["mlp_w2"] + blk["mlp_b2"]
            x = x + mlp_out

            cache_blocks.append({
                "x_in": x_in, "x_mid": x_mid,
                "xhat1": xhat1, "sigma1": sigma1, "h": h,
                "q": q, "k": k, "v": v, "p_att": p_att, "out_pre_proj": out,
                "xhat2": xhat2, "sigma2": sigma2, "h2": h2,
                "mlp_pre": mlp_pre, "m": m,
            })

        xf, (xhatf, sigmaf) = _ln_forward(x, p["lnf_g"], p["lnf_b"])
        logits = xf @ p["wte"].T  # tied head, (B,T,V)

        self.cache = {
            "idx": idx, "pos": pos, "x_emb": x_emb,
            "blocks": cache_blocks, "xf": xf, "xhatf": xhatf, "sigmaf": sigmaf,
        }
        return logits

    # ---- 反向 ----
    def backward(self, dlogits: np.ndarray) -> dict:
        """dlogits (B,T,V) → 各参数梯度（结构与 params 一致）。"""
        p = self.params
        c = self.cache
        B, T = c["idx"].shape
        C = self.n_embd

        grads = self._zero_grads()

        # 输出头（tied）：logits = xf @ wte.T
        dL = dlogits.reshape(-1, self.vocab_size)
        grads["wte"] = dL.T @ c["xf"].reshape(-1, C)
        dx = dlogits @ p["wte"]  # (B,T,C)

        # 最终 LayerNorm
        dx, dgf, dbf = _ln_backward(dx, c["xhatf"], c["sigmaf"], p["lnf_g"])
        grads["lnf_g"] = dgf
        grads["lnf_b"] = dbf

        # 逐层反向
        for i in reversed(range(self.n_layer)):
            dx = self._block_backward(dx, p["blocks"][i], c["blocks"][i], grads["blocks"][i])

        # 输入嵌入（tied）：x_emb = wte[idx] + wpe[pos]
        grads["wpe"][:T] += dx.sum(axis=0)
        np.add.at(grads["wte"], c["idx"].reshape(-1), dx.reshape(-1, C))

        self.cache = None
        return grads

    def _block_backward(self, dx: np.ndarray, blk: dict, cb: dict, gblk: dict) -> np.ndarray:
        B, T, C = dx.shape
        H, D = self.n_head, self.head_dim

        # --- 反向 MLP ---
        dmlp_out = dx
        gblk["mlp_w2"] = cb["m"].reshape(-1, 4 * C).T @ dmlp_out.reshape(-1, C)
        gblk["mlp_b2"] = dmlp_out.sum(axis=(0, 1))
        dm = dmlp_out @ blk["mlp_w2"].T
        dmlp_pre = dm * _gelu_backward(cb["mlp_pre"])
        gblk["mlp_w1"] = cb["h2"].reshape(-1, C).T @ dmlp_pre.reshape(-1, 4 * C)
        gblk["mlp_b1"] = dmlp_pre.sum(axis=(0, 1))
        dh2 = dmlp_pre @ blk["mlp_w1"].T

        # --- 反向 LN2（+ mlp 残差） ---
        dx_mid_body, gblk["ln2_g"], gblk["ln2_b"] = _ln_backward(
            dh2, cb["xhat2"], cb["sigma2"], blk["ln2_g"]
        )
        dx_mid = dx_mid_body + dmlp_out

        # --- 反向注意力 ---
        dattn_out = dx_mid
        dout_pre = dattn_out @ blk["c_proj_w"].T
        gblk["c_proj_w"] = cb["out_pre_proj"].reshape(-1, C).T @ dattn_out.reshape(-1, C)
        gblk["c_proj_b"] = dattn_out.sum(axis=(0, 1))

        dO = dout_pre.reshape(B, T, H, D).transpose(0, 2, 1, 3)   # (B,H,T,D)
        dp_att = dO @ cb["v"].transpose(0, 1, 3, 2)               # (B,H,T,T)
        dV = cb["p_att"].transpose(0, 1, 3, 2) @ dO               # (B,H,T,D)

        dscores = cb["p_att"] * (dp_att - (dp_att * cb["p_att"]).sum(-1, keepdims=True))
        dscores = dscores / np.sqrt(D)
        dQ = dscores @ cb["k"]
        dK = dscores.transpose(0, 1, 3, 2) @ cb["q"]

        dQf = dQ.transpose(0, 2, 1, 3).reshape(B, T, C)
        dKf = dK.transpose(0, 2, 1, 3).reshape(B, T, C)
        dVf = dV.transpose(0, 2, 1, 3).reshape(B, T, C)
        dqkv = np.concatenate([dQf, dKf, dVf], axis=-1)

        gblk["c_attn_w"] = cb["h"].reshape(-1, C).T @ dqkv.reshape(-1, 3 * C)
        gblk["c_attn_b"] = dqkv.sum(axis=(0, 1))
        dh = dqkv @ blk["c_attn_w"].T

        # --- 反向 LN1（+ attn 残差） ---
        dx_in_body, gblk["ln1_g"], gblk["ln1_b"] = _ln_backward(
            dh, cb["xhat1"], cb["sigma1"], blk["ln1_g"]
        )
        return dx_in_body + dx_mid

    # ---- 参数工具 ----
    def _zero_grads(self) -> dict:
        p = self.params
        grads = {
            "wte": np.zeros_like(p["wte"]),
            "wpe": np.zeros_like(p["wpe"]),
            "blocks": [{k: np.zeros_like(v) for k, v in blk.items()} for blk in p["blocks"]],
            "lnf_g": np.zeros_like(p["lnf_g"]),
            "lnf_b": np.zeros_like(p["lnf_b"]),
        }
        return grads

    def flat_params(self) -> list[np.ndarray]:
        order = [self.params["wte"], self.params["wpe"]]
        for blk in self.params["blocks"]:
            order += [blk[k] for k in _BLOCK_KEYS]
        order += [self.params["lnf_g"], self.params["lnf_b"]]
        return order

    def flat_grads(self, grads: dict) -> list[np.ndarray]:
        order = [grads["wte"], grads["wpe"]]
        for gblk in grads["blocks"]:
            order += [gblk[k] for k in _BLOCK_KEYS]
        order += [grads["lnf_g"], grads["lnf_b"]]
        return order


# ---------------------------------------------------------------------------
# Adam 优化器
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, betas=(0.9, 0.99), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Transformer 歌词模型（统一 LyricsModel 接口）
# ---------------------------------------------------------------------------

class TransformerModel(LyricsModel):
    """GPT-2 风格歌词模型。

    Args:
        block_size: 上下文窗口长度。
        n_embd: 嵌入维度。
        n_head: 注意力头数。
        n_layer: Transformer 层数。
        steps: 训练步数。
        batch_size: 批大小。
        lr: 学习率。
        temperature: 采样温度（>1 更随机，<1 更确定）。
        seed: 随机种子。
    """

    def __init__(
        self,
        block_size: int = 32,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 3,
        steps: int = 1500,
        batch_size: int = 32,
        lr: float = 1e-3,
        temperature: float = 1.0,
        seed: int = 1337,
    ):
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.seed = seed

        self.tok: Tokenizer | None = None
        self.gpt: GPT | None = None
        self._corpus: np.ndarray | None = None

    # ---- 训练 ----
    def train(self, allowed_classes: list[str]) -> "TransformerModel":
        alphabet = "".join(build_allowed_chars(allowed_classes))
        if not alphabet:
            raise ValueError(f"allowed_classes={allowed_classes} 没有可用字符")

        self.tok = Tokenizer(alphabet)
        corpus = self._build_corpus(allowed_classes)

        self.gpt = GPT(
            vocab_size=self.tok.vocab_size,
            block_size=self.block_size,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_layer=self.n_layer,
            seed=self.seed,
        )

        rng = np.random.default_rng(self.seed)
        opt = Adam(self.gpt.flat_params(), lr=self.lr)

        T = self.block_size + 1
        for step in range(self.steps):
            # 随机采样 batch 个 (block_size+1) 的窗口
            starts = rng.integers(0, len(corpus) - T, size=self.batch_size)
            xs = np.stack([corpus[s:s + T] for s in starts])
            x, y = xs[:, :-1], xs[:, 1:]

            logits = self.gpt.forward(x)
            loss, dlogits = _cross_entropy(logits, y)
            grads = self.gpt.backward(dlogits)
            opt.step(self.gpt.flat_grads(grads))

            if (step + 1) % max(1, self.steps // 10) == 0:
                logger.info("step %d/%d loss=%.4f", step + 1, self.steps, loss)

        self._corpus = corpus
        logger.info(
            "Transformer trained: vocab=%d blocks=%d params=%d",
            self.tok.vocab_size, self.n_layer, sum(p.size for p in self.gpt.flat_params()),
        )
        return self

    def _build_corpus(self, allowed_classes: list[str]) -> np.ndarray:
        ids: list[int] = []
        for text, weight in iter_allowed_texts(allowed_classes):
            reps = max(1, int(round(weight)))
            seg = self.tok.encode(text) + [self.tok.eos_id]
            ids.extend(seg * reps)

        corpus = np.array(ids, dtype=np.int64)
        # 语料太短时重复，保证至少有 block_size+1 长度
        need = self.block_size + 2
        while len(corpus) < need:
            corpus = np.concatenate([corpus, corpus])
        self._corpus = corpus
        return corpus

    # ---- 生成 ----
    @property
    def vocab(self) -> list[str]:
        return self.tok.vocab if self.tok else []

    def sample(self, context: str = "") -> str:
        if self.gpt is None or self.tok is None:
            return ""

        ids = self.tok.encode(context)[-self.block_size:]
        if not ids:
            ids = [self.tok.bos_id]
        x = np.array([ids], dtype=np.int64)

        logits = self.gpt.forward(x)[0, -1, :]
        logits = logits - logits.max()
        probs = np.exp(logits / self.temperature)
        probs = probs / probs.sum()

        # 屏蔽特殊 token，禁止生成 <pad>/<bos>/<eos>/<unk>
        probs[:len(SPECIAL_TOKENS)] = 0.0
        if probs.sum() <= 0:
            return random.choice(self.vocab)
        probs /= probs.sum()

        idx = int(np.random.choice(len(probs), p=probs))
        return self.tok.decode([idx])

    def generate(self, start: str | None = None, length: int = 50) -> str:
        out = "" if start is None else str(start)
        while len(out) < length:
            nxt = self.sample(out)
            if not nxt:
                break
            out += nxt
        return out[:length]

    # ---- 持久化 ----
    def save(self, path: str) -> None:
        if self.gpt is None:
            raise RuntimeError("尚未训练，无法保存")
        arrays = {}
        for i, arr in enumerate(self.gpt.flat_params()):
            arrays[f"p{i}"] = arr
        arrays["_vocab"] = np.array("".join(self.tok.vocab))
        np.savez_compressed(
            path,
            block_size=self.block_size, n_embd=self.n_embd, n_head=self.n_head,
            n_layer=self.n_layer, temperature=self.temperature,
            **arrays,
        )

    @classmethod
    def load(cls, path: str) -> "TransformerModel":
        with np.load(path, allow_pickle=False) as data:
            block_size = int(data["block_size"])
            n_embd = int(data["n_embd"])
            n_head = int(data["n_head"])
            n_layer = int(data["n_layer"])
            temperature = float(data["temperature"])
            vocab = str(data["_vocab"])

        model = cls(
            block_size=block_size, n_embd=n_embd, n_head=n_head,
            n_layer=n_layer, temperature=temperature,
        )
        model.tok = Tokenizer(vocab)
        model.gpt = GPT(
            vocab_size=model.tok.vocab_size, block_size=block_size,
            n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        )
        with np.load(path, allow_pickle=False) as data:
            for i, arr in enumerate(model.gpt.flat_params()):
                arr[...] = data[f"p{i}"]
        return model


# ---------------------------------------------------------------------------
# 梯度校验工具（测试用）
# ---------------------------------------------------------------------------

def grad_check(model: GPT, idx: np.ndarray, targets: np.ndarray, eps: float = 1e-3):
    """数值梯度 vs 解析梯度。返回最大相对误差。

    参数为 float32，因此 eps 取 1e-3（过小会被舍入误差淹没）。
    正确实现下最大相对误差应 < 1e-2。
    """
    logits = model.forward(idx)
    _, dlogits = _cross_entropy(logits, targets)
    grads = model.backward(dlogits)

    flat_params = model.flat_params()
    flat_grads = model.flat_grads(grads)

    max_rel = 0.0
    for p, g in zip(flat_params, flat_grads):
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            lp, _ = _cross_entropy(model.forward(idx), targets)
            p[i] = old - eps
            lm, _ = _cross_entropy(model.forward(idx), targets)
            p[i] = old
            num[i] = (lp - lm) / (2 * eps)
            it.iternext()

        denom = np.abs(num) + np.abs(g) + 1e-8
        rel = np.max(np.abs(num - g) / denom)
        max_rel = max(max_rel, float(rel))

    return max_rel
