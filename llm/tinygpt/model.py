"""A small GPT-style (decoder-only, causal) Transformer language model, written from
scratch in PyTorch. All weights start random - no pretrained model is used.

How a forward pass works, for a batch of token-ID sequences of shape (B, T):

  token IDs ──► token embedding  (each ID -> a learned vector of size n_embd)
            ──► + position embedding (each position 0..T-1 -> a learned vector)
            ──► n_layer x Transformer block:
                    x = x + Attention(LayerNorm(x))   # each token looks at earlier tokens
                    x = x + MLP(LayerNorm(x))         # each token is processed on its own
            ──► final LayerNorm
            ──► output layer -> one score ("logit") per vocabulary entry, per position

Training objective (causal language modelling): at every position, predict the NEXT
token. The causal mask guarantees position t can only use tokens 0..t, never the future.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position may only attend to itself and
    earlier positions."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.fused = cfg.fused_attention
        self.dropout = cfg.dropout
        # one Linear produces queries, keys and values for all heads at once
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # lower-triangular matrix of ones: row t has ones in columns 0..t (the allowed positions)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head works on its own slice of the vector
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))

        if self.fused:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
        else:
            # similarity of every query with every key, scaled so softmax stays well-behaved
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)       # (B, nh, T, T)
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))   # hide future positions
            att = F.softmax(att, dim=-1)                                      # weights over allowed positions sum to 1
            att = self.attn_drop(att)
            y = att @ v                                                       # weighted mix of values

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-join the heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: widen to 4x, apply GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One Transformer block (pre-norm): normalise, transform, add back (residual)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: the output layer reuses the input embedding matrix - fewer
        # parameters, and in practice it trains better for small models
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # residual output projections start smaller so the sum over many layers stays stable
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # random initialisation - this is what "from scratch" means
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:  # position embeddings are a lookup table, not "thinking" capacity
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token IDs. targets: (B, T) the next token at each position, or None.
        Returns (logits, loss); loss is None when no targets are given."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            # generating: only the last position's prediction is needed
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        # cross-entropy = how surprised the model is by the true next token, averaged
        # (reshape, not view: targets may be a slice of a larger tensor, e.g. seq[:, 1:])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 eos_id: int | None = None) -> torch.Tensor:
        """Extend idx (B, T) one token at a time.
        temperature: <1 = safer/more repetitive, >1 = more random, 0 = always the top token.
        top_k: sample only among the k most likely tokens.
        top_p: sample only among the smallest set of tokens whose probabilities sum to p."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size:]  # the model can only see block_size tokens
            logits, _ = self(context)
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    drop = cum - F.softmax(sorted_logits, dim=-1) > top_p  # keep the token that crosses p
                    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1,
                            betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
        """AdamW with weight decay on matrices only (not on biases, LayerNorm or embeddings'
        1-D parameters), the usual setup for Transformers."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused = all(p.is_cuda for p in params)
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
