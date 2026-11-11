"""Batched sampling of SMILES from a checkpoint: python -m mol.sample [--qed_bin 9 ...]."""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from gpt.generate import load_checkpoint
from mol import metrics
from mol.data import bin_of, load_csv


@torch.no_grad()
def sample_batch(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None,
                 eos_id=None, pad_id=0):
    """Sample a whole batch at once. prompt_ids: (B,T0) -> (B,T0+T') with pad after <eos>."""
    model.eval()
    block = model.config.block_size
    idx = prompt_ids                                                       # (B,T0)
    B = idx.size(0)
    done = torch.zeros(B, dtype=torch.bool, device=idx.device)             # (B,) rows that hit eos
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])                                 # (B,T,V)
        logits = logits[:, -1, :]                                          # (B,V) last position
        if temperature == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)                      # (B,1)
        else:
            logits = logits / temperature
            if top_k:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k).values[:, -1:]                 # (B,1)
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)                              # (B,V)
            nxt = torch.multinomial(probs, num_samples=1)                  # (B,1)
        nxt = nxt.masked_fill(done.unsqueeze(1), pad_id)                   # finished rows -> pad
        idx = torch.cat([idx, nxt], dim=1)                                 # (B,T+1)
        if eos_id is not None:
            done |= nxt.squeeze(1) == eos_id
            if bool(done.all()):
                break
    return idx


def prompt_ids(tok, qed_bin=None, logp_bin=None):
    """[<bos>, <qed_*>, <logp_*>]; None means the "?" (unspecified) variant."""
    q = "<qed?>" if qed_bin is None else f"<qed{qed_bin}>"
    p = "<logp?>" if logp_bin is None else f"<logp{logp_bin}>"
    return [tok.bos_id, tok.id_of(q), tok.id_of(p)]


def sample_smiles(model, tok, n, block_size, qed_bin=None, logp_bin=None, temperature=1.0,
                  top_k=None, batch_size=128, device="cpu"):
    """Generate n SMILES strings in batches of batch_size."""
    base = prompt_ids(tok, qed_bin, logp_bin)
    out = []
    while len(out) < n:
        b = min(batch_size, n - len(out))
        start = torch.tensor([base] * b, dtype=torch.long, device=device)  # (B,3)
        ids = sample_batch(model, start, block_size - len(base), temperature, top_k,
                           tok.eos_id, tok.pad_id)                        # (B,T)
        out.extend(tok.decode(row.tolist()) for row in ids)
    return out[:n]


def print_table(m):
    rows = [("n", m["n"]), ("validity", m["validity"]), ("uniqueness", m["uniqueness"]),
            ("novelty", m["novelty"]), ("mean qed", m["qed"]), ("mean logp", m["logp"]),
            ("mean mol weight", m["mol_weight"]), ("mean rings", m["rings"])]
    for name, v in rows:
        shown = "n/a" if v is None else (f"{v}" if isinstance(v, int) else f"{v:.4f}")
        print(f"  {name:<16} {shown}")


def main():
    p = argparse.ArgumentParser(description="Sample SMILES from a trained checkpoint")
    p.add_argument("--ckpt", default="checkpoints/smiles/best.pt")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--qed_bin", type=int, default=None)
    p.add_argument("--logp_bin", type=int, default=None)
    p.add_argument("--target_qed", type=float, default=None)
    p.add_argument("--target_logp", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out", default=None)
    p.add_argument("--grid", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.seed is not None:
        torch.manual_seed(a.seed)

    model, tok, ckpt = load_checkpoint(a.ckpt, device)
    edges = ckpt.get("edges") or {}
    qed_bin, logp_bin = a.qed_bin, a.logp_bin
    if a.target_qed is not None:
        qed_bin = bin_of(a.target_qed, edges["qed"])
    if a.target_logp is not None:
        logp_bin = bin_of(a.target_logp, edges["logp"])
    print(f"ckpt={a.ckpt} device={device} qed_bin={qed_bin} logp_bin={logp_bin} n={a.n}")

    gen = sample_smiles(model, tok, a.n, model.config.block_size, qed_bin, logp_bin,
                        a.temperature, a.top_k or None, a.batch_size, device)

    # training SMILES for novelty, if the data directory is still around
    data_dir = ckpt.get("train_config", {}).get("data_dir", "data/processed")
    train_csv = os.path.join(data_dir, "train.csv")
    train_smiles = None
    if os.path.exists(train_csv):
        train_smiles = {r[0] for r in load_csv(train_csv)}
    else:
        print(f"warning: {train_csv} not found, novelty will be reported as None")

    m = metrics.score(gen, train_smiles)
    print("metrics:")
    print_table(m)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        payload = {k: v for k, v in m.items() if k != "valid_canonical"}
        payload.update({"ckpt": a.ckpt, "temperature": a.temperature, "top_k": a.top_k,
                        "qed_bin": qed_bin, "logp_bin": logp_bin, "generated": gen})
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {a.out}")
    if a.grid:
        os.makedirs(os.path.dirname(a.grid) or ".", exist_ok=True)
        if metrics.draw_grid(m["valid_canonical"], a.grid):
            print(f"wrote {a.grid}")


if __name__ == "__main__":
    main()
