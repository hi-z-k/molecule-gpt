"""Sampling, beam search, checkpoint loading and a small CLI."""
import argparse

import torch
import torch.nn.functional as F

from gpt.config import GPTConfig
from gpt.model import GPT
from gpt.tokenizer import CharTokenizer


def _allowed_mask(allowed_ids, vocab_size, device):
    """Bool mask (V,) that is True for ids we may produce, or None for no restriction."""
    if allowed_ids is None:
        return None
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    mask[torch.as_tensor(list(allowed_ids) if not torch.is_tensor(allowed_ids) else allowed_ids,
                         dtype=torch.long, device=device)] = True
    return mask


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, allowed_ids=None, eos_id=None):
    """Autoregressive sampling. idx: (1,T) -> (1,T'). temperature 0 = greedy."""
    model.eval()
    block = model.config.block_size
    allowed = _allowed_mask(allowed_ids, model.config.vocab_size, idx.device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])                                 # (1,T,V)
        logits = logits[:, -1, :]                                          # (1,V) last position
        if allowed is not None:
            logits = logits.masked_fill(~allowed, float("-inf"))           # (1,V)
        if temperature == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)                      # (1,1)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k).values[:, -1:]                 # (1,1)
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)                              # (1,V)
            nxt = torch.multinomial(probs, num_samples=1)                  # (1,1)
        idx = torch.cat([idx, nxt], dim=1)                                 # (1,T+1)
        if eos_id is not None and nxt.item() == eos_id:
            break
    return idx


@torch.no_grad()
def beam_search(model, idx, max_new_tokens, beam=4, allowed_ids=None, eos_id=None, n_best=3):
    """Beam search from prompt idx (1,T). Returns [(new_ids_list, length-normalized logprob)].

    The returned ids are the generated tokens only (prompt excluded, eos included if produced).
    """
    model.eval()
    block = model.config.block_size
    allowed = _allowed_mask(allowed_ids, model.config.vocab_size, idx.device)
    beams = [([], 0.0)]            # (generated ids, sum logprob)
    finished = []
    for _ in range(max_new_tokens):
        candidates = []
        for toks, score in beams:
            seq = torch.cat([idx, torch.tensor([toks], dtype=torch.long, device=idx.device)], dim=1) if toks else idx
            logits, _ = model(seq[:, -block:])                             # (1,T,V)
            logits = logits[0, -1, :]                                      # (V,)
            if allowed is not None:
                logits = logits.masked_fill(~allowed, float("-inf"))
            logp = F.log_softmax(logits, dim=-1)                           # (V,)
            top = torch.topk(logp, min(beam, logp.size(-1)))
            for lp, i in zip(top.values.tolist(), top.indices.tolist()):
                if lp == float("-inf"):
                    continue
                candidates.append((toks + [i], score + lp))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for toks, score in candidates[:beam]:
            if eos_id is not None and toks[-1] == eos_id:
                finished.append((toks, score))
            else:
                beams.append((toks, score))
        if not beams or len(finished) >= beam:
            break
    pool = finished + beams if len(finished) < n_best else finished
    results = [(t, s / max(len(t), 1)) for t, s in pool]
    results.sort(key=lambda r: r[1], reverse=True)
    return results[:n_best]


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint -> (model in eval mode, tokenizer, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = CharTokenizer.from_dict(ckpt["vocab"])
    return model, tok, ckpt


def main():
    p = argparse.ArgumentParser(description="Sample text from a trained checkpoint")
    p.add_argument("--ckpt", default="checkpoints/smiles/best.pt")
    p.add_argument("--prompt", default="", help="SMILES prefix to continue; empty starts from <bos>")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    if a.seed is not None:
        torch.manual_seed(a.seed)
    model, tok, _ = load_checkpoint(a.ckpt, device)
    ids = tok.encode(a.prompt) or [tok.bos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, a.max_new_tokens, a.temperature, a.top_k or None)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
