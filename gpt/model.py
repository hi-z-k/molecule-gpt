"""GPT-2 style decoder-only Transformer, built from basic tensor ops.

Notation: B = batch, T = sequence length, C = n_embd, H = n_head, D = head_dim = C / H,
V = vocab_size.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm over the last dim: y = (x - mean) / sqrt(var + eps) * gamma + beta."""

    def __init__(self, dim, eps=1e-5, bias=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))                        # gamma (C,)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None       # beta (C,)

    def forward(self, x):                                                  # x: (B,T,C)
        mean = x.mean(dim=-1, keepdim=True)                                # (B,T,1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)                  # (B,T,1)
        x = (x - mean) / torch.sqrt(var + self.eps)                        # (B,T,C)
        x = x * self.weight                                                # (B,T,C)
        if self.bias is not None:
            x = x + self.bias                                              # (B,T,C)
        return x                                                           # (B,T,C)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V   (Vaswani et al. 2017, Eq. 1),
    with a causal mask so position t only attends to positions <= t.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # (C) -> (3C)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)     # (C) -> (C)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # lower-triangular matrix of ones: 1 = may attend, 0 = future position
        mask = torch.tril(torch.ones(config.block_size, config.block_size))       # (S,S)
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))  # (1,1,S,S)

    def forward(self, x):                                                  # x: (B,T,C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                              # 3 x (B,T,C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B,T,C) -> (B,H,T,D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B,T,C) -> (B,H,T,D)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B,T,C) -> (B,H,T,D)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)        # (B,H,T,D)@(B,H,D,T) -> (B,H,T,T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # (B,H,T,T) future -> -inf
        weights = F.softmax(scores, dim=-1)                                # (B,H,T,T) rows sum to 1
        weights = self.attn_drop(weights)                                  # (B,H,T,T)
        out = weights @ v                                                  # (B,H,T,T)@(B,H,T,D) -> (B,H,T,D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)               # (B,H,T,D) -> (B,T,H,D) -> (B,T,C)
        return self.resid_drop(self.proj(out))                             # (B,T,C) -> (B,T,C)


class MLP(nn.Module):
    """Position-wise feed-forward: Linear(C,4C) -> GELU -> Linear(4C,C)."""

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)    # (C) -> (4C)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # (4C) -> (C)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):                                                  # x: (B,T,C)
        x = self.fc(x)                                                     # (B,T,C) -> (B,T,4C)
        x = F.gelu(x)                                                      # exact (erf) GELU, (B,T,4C)
        x = self.proj(x)                                                   # (B,T,4C) -> (B,T,C)
        return self.drop(x)                                                # (B,T,C)


class Block(nn.Module):
    """Pre-LN Transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):                                                  # x: (B,T,C)
        x = x + self.attn(self.ln1(x))                                     # residual, (B,T,C)
        x = x + self.mlp(self.ln2(x))                                      # residual, (B,T,C)
        return x                                                           # (B,T,C)


class GPT(nn.Module):
    """Token + learned position embeddings -> N blocks -> final LN -> tied output head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)          # token emb (V,C)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)          # position emb (S,C)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # (C) -> (V)
        self.lm_head.weight = self.wte.weight                              # weight tying

        self.apply(self._init_weights)
        # scale residual projections by 1/sqrt(2*n_layer) (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        """Number of unique parameters (the tied head is counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None):
        """idx: (B,T) token ids. targets: (B,T) with -100 at positions to ignore.

        loss_mask (optional, (B,T) bool): extra positions to include; False -> ignored.
        Returns logits (B,T,V) and loss (scalar) or None.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)                           # (T,)
        x = self.wte(idx) + self.wpe(pos)                                  # (B,T,C) + (T,C) -> (B,T,C)
        x = self.drop(x)                                                   # (B,T,C)
        for block in self.blocks:
            x = block(x)                                                   # (B,T,C)
        x = self.ln_f(x)                                                   # (B,T,C)
        logits = self.lm_head(x)                                           # (B,T,C) -> (B,T,V)
        loss = None
        if targets is not None:
            if loss_mask is not None:
                targets = targets.masked_fill(~loss_mask.bool(), -100)     # (B,T)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),                          # (B*T,V)
                targets.reshape(-1),                                       # (B*T,)
                ignore_index=-100,
            )
        return logits, loss
