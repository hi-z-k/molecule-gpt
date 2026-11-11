import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.config import GPTConfig
from gpt.model import GPT
from gpt.tokenizer import CharTokenizer


def tiny(vocab=50, **kw):
    cfg = dict(vocab_size=vocab, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg))


def test_output_shapes():
    torch.manual_seed(0)
    m = tiny()
    idx = torch.randint(0, 50, (3, 10))
    logits, loss = m(idx)
    assert logits.shape == (3, 10, 50) and loss is None
    logits, loss = m(idx, idx)
    assert loss.dim() == 0


def test_initial_loss_close_to_ln_vocab():
    torch.manual_seed(0)
    m = tiny(vocab=100)
    idx = torch.randint(0, 100, (16, 16))
    targets = torch.randint(0, 100, (16, 16))  # independent of idx (tied weights favour y == x)
    _, loss = m(idx, targets)
    assert abs(loss.item() - math.log(100)) < 0.3


def test_causal_no_leak():
    torch.manual_seed(0)
    m = tiny().eval()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 50  # change only the future
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)


def test_tied_weights():
    m = tiny()
    assert m.lm_head.weight is m.wte.weight


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = tiny(vocab=20)
    x = torch.randint(0, 20, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    loss = None
    for _ in range(200):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_ignore_index():
    torch.manual_seed(0)
    m = tiny().eval()
    x = torch.randint(0, 50, (2, 8))
    y = x.clone()
    y[:, :4] = -100
    _, l1 = m(x, y)
    _, l2 = m(x, x.clone(), loss_mask=torch.arange(8).expand(2, 8) >= 4)
    assert torch.allclose(l1, l2)


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_texts(["hello world", "ሰላም"])
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.sep_id == 2 and tok.eos_id == 3
    ids = tok.encode("hello ሰላም")
    assert tok.decode(ids) == "hello ሰላም"
    assert tok.decode([tok.bos_id] + ids + [tok.eos_id]) == "hello ሰላም"
    assert tok.encode("hZ") == tok.encode("h")  # unknown dropped
    tok2 = CharTokenizer.from_dict(tok.to_dict())
    assert tok2.encode("hello") == tok.encode("hello")
