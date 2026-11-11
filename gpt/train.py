"""Training script: python -m gpt.train [--any_config_field value]."""
import argparse
import dataclasses
import json
import math
import os
import time

import torch

from gpt.config import GPTConfig, TrainConfig, add_config_args, config_from_args
from gpt.model import GPT


def get_lr(step, cfg):
    """Linear warmup, then cosine decay from lr to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def make_optimizer(model, cfg):
    """AdamW; weight decay only on 2D weights (matrices / embeddings), not on biases / LN."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def build_data(cfg, block_size):
    """Return (get_batch(split, device), tokenizer, meta, train_smiles)."""
    from mol.data import get_datasets
    train_ds, val_ds, tok, meta = get_datasets(cfg, block_size)
    sets = {"train": train_ds, "val": val_ds}

    def get_batch(split, device):
        ds = sets[split]
        ix = torch.randint(len(ds), (cfg.batch_size,)).tolist()
        items = [ds[i] for i in ix]
        x = torch.stack([it[0] for it in items])                           # (B,T)
        y = torch.stack([it[1] for it in items])                           # (B,T), -100 on prompt/pad
        return x.to(device), y.to(device)

    train_smiles = {r[0] for r in train_ds.rows}
    return get_batch, tok, meta, train_smiles


@torch.no_grad()
def estimate_loss(model, get_batch, cfg, device, amp):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(split, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def eval_samples(model, tok, cfg, device, train_smiles):
    """Generate cfg.eval_samples molecules unconditionally -> validity/uniqueness/novelty."""
    from mol import metrics
    from mol.sample import sample_smiles
    model.eval()
    gen = sample_smiles(model, tok, cfg.eval_samples, model.config.block_size,
                        batch_size=min(cfg.eval_samples, 256), device=device)
    m = metrics.score(gen, train_smiles)
    model.train()
    return {"validity": m["validity"], "uniqueness": m["uniqueness"], "novelty": m["novelty"]}


def save_plot(log, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if log["train"]:
        ax1.plot(*zip(*log["train"]), label="train")
    if log["val"]:
        ax1.plot(*zip(*log["val"]), label="val")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend()
    for key in ("validity", "uniqueness", "novelty"):
        pts = [(step, m[key]) for step, m in log["samples"] if m.get(key) is not None]
        if pts:
            ax2.plot(*zip(*pts), label=key)
    ax2.set_xlabel("step")
    ax2.set_ylabel("fraction")
    ax2.set_ylim(0, 1.02)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train(cfg, gcfg):
    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    amp = cfg.fp16 and device.startswith("cuda")
    get_batch, tok, meta, train_smiles = build_data(cfg, gcfg.block_size)
    gcfg.vocab_size = tok.vocab_size
    edges = {"qed": meta["qed_edges"], "logp": meta["logp_edges"]}

    model = GPT(gcfg).to(device)
    n_params = model.num_params()
    print(f"device={device} vocab={tok.vocab_size} params={n_params / 1e6:.2f}M "
          f"train={meta['n_train']} val={meta['n_val']}")
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    log = {"train": [], "val": [], "samples": [],
           "config": {"model": dataclasses.asdict(gcfg), "train": dataclasses.asdict(cfg)},
           "params": n_params, "minutes": 0.0}
    best_val = float("inf")

    def save(path, step, val_loss):
        torch.save({"model": model.state_dict(), "model_config": dataclasses.asdict(gcfg),
                    "train_config": dataclasses.asdict(cfg), "vocab": tok.to_dict(),
                    "edges": edges, "n_bins": meta["n_bins"],
                    "step": step, "val_loss": val_loss}, path)

    def write_log():
        log["minutes"] = (time.time() - t0) / 60
        with open(os.path.join(cfg.results_dir, f"{cfg.run_name}_log.json"), "w") as f:
            json.dump(log, f)
        save_plot(log, os.path.join(cfg.results_dir, f"{cfg.run_name}_loss.png"))

    def full_eval(step):
        """Loss on both splits plus the generation metrics; saves checkpoints and the log."""
        nonlocal best_val, val_loss
        losses = estimate_loss(model, get_batch, cfg, device, amp)
        val_loss = losses["val"]
        log["val"].append([step, val_loss])
        line = f"step {step}: train {losses['train']:.4f} val {val_loss:.4f}"
        if cfg.eval_samples > 0:
            s = eval_samples(model, tok, cfg, device, train_smiles)
            log["samples"].append([step, s])
            nov = "n/a" if s["novelty"] is None else f"{s['novelty']:.3f}"
            line += f" | valid {s['validity']:.3f} uniq {s['uniqueness']:.3f} novel {nov}"
        print(f"{line} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            save(os.path.join(ckpt_dir, "best.pt"), step, val_loss)
        save(os.path.join(ckpt_dir, "last.pt"), step, val_loss)
        write_log()

    t0 = time.time()
    model.train()
    step = 0
    val_loss = float("nan")
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            full_eval(step)
        if step == cfg.max_steps or (time.time() - t0) / 60 >= cfg.max_minutes:
            break

        for g in opt.param_groups:
            g["lr"] = get_lr(step, cfg)
        x, y = get_batch("train", device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        log["train"].append([step, loss.item()])
        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.4f}", flush=True)

    # final eval if we stopped on the time limit
    if step < cfg.max_steps and (not log["val"] or log["val"][-1][0] != step):
        full_eval(step)
    write_log()
    print(f"done in {(time.time() - t0) / 60:.1f} min; checkpoints in {ckpt_dir}")


def main():
    p = argparse.ArgumentParser(description="Train the GPT model")
    add_config_args(p, TrainConfig, GPTConfig)
    args = p.parse_args()
    cfg = config_from_args(args, TrainConfig)
    gcfg = config_from_args(args, GPTConfig)
    train(cfg, gcfg)


if __name__ == "__main__":
    main()
