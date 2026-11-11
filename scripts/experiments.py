"""Experiments for the write-up: python scripts/experiments.py {temperature,context,ablation,all}.

1. temperature: validity / uniqueness / novelty of unconditional samples vs temperature
2. context:     per-position validation loss, and generation quality vs real-prefix length
3. ablation:    train small models while varying n_layer / n_head / n_embd
Each writes <name>.json and <name>.png into --out_dir; summary.md is rebuilt from the JSONs.
"""
import argparse
import copy
import dataclasses
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.config import GPTConfig, TrainConfig
from gpt.generate import load_checkpoint
from mol import metrics
from mol.data import get_datasets, load_csv
from mol.sample import prompt_ids, sample_batch, sample_smiles

AXIS_DEFAULTS = {"n_layer": [2, 4, 8], "n_head": [2, 4, 8], "n_embd": [64, 128, 256]}


def floats(s):
    return [float(v) for v in s.split(",") if v.strip()]


def ints(s):
    return [int(v) for v in s.split(",") if v.strip()]


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def save_json(out_dir, name, obj):
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=2)


def load_ckpt_or_exit(args):
    if not os.path.exists(args.ckpt):
        print(f"error: checkpoint {args.ckpt} not found. Train first (python -m gpt.train), "
              f"or pass --ckpt. The `ablation` experiment works without a checkpoint.")
        sys.exit(1)
    return load_checkpoint(args.ckpt, args.device)


def load_train_set(ckpt):
    """Training SMILES set for novelty, or None (with a warning) if the data is gone."""
    path = os.path.join(ckpt["train_config"]["data_dir"], "train.csv")
    if not os.path.exists(path):
        print(f"warning: {path} not found, novelty will be reported as null")
        return None
    return {r[0] for r in load_csv(path)}


def mean_len(m):
    c = m["valid_canonical"]
    return sum(map(len, c)) / len(c) if c else None


# ---------------------------------------------------------------- 1. temperature

def run_temperature(args):
    model, tok, ckpt = load_ckpt_or_exit(args)
    train_set = load_train_set(ckpt)
    block = model.config.block_size
    results = []
    print(f"\n== temperature (n={args.n}) ==")
    print(f"{'T':>5} {'valid':>7} {'uniq':>7} {'novel':>7} {'qed':>7} {'logp':>7} {'mw':>8} {'len':>6}")
    for t in floats(args.temperatures):
        gen = sample_smiles(model, tok, args.n, block, temperature=t,
                            batch_size=args.batch_size, device=args.device)
        m = metrics.score(gen, train_set)
        r = {"temperature": t, "validity": m["validity"], "uniqueness": m["uniqueness"],
             "novelty": m["novelty"], "qed": m["qed"], "logp": m["logp"],
             "mol_weight": m["mol_weight"], "mean_len": mean_len(m), "examples": gen[:3]}
        results.append(r)
        print(f"{t:>5} {fmt(r['validity']):>7} {fmt(r['uniqueness']):>7} {fmt(r['novelty']):>7} "
              f"{fmt(r['qed']):>7} {fmt(r['logp']):>7} {fmt(r['mol_weight'], 1):>8} {fmt(r['mean_len'], 1):>6}")
    save_json(args.out_dir, "temperature.json", {"n": args.n, "ckpt": args.ckpt, "results": results})

    plt.figure(figsize=(6, 4))
    ts = [r["temperature"] for r in results]
    for key in ("validity", "uniqueness", "novelty"):
        pts = [(t, r[key]) for t, r in zip(ts, results) if r[key] is not None]
        if pts:
            plt.plot(*zip(*pts), marker="o", label=key)
    plt.xlabel("temperature (0 = greedy)")
    plt.ylabel("fraction")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "temperature.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------- 2. context

@torch.no_grad()
def position_loss(model, val_ds, n, batch_size, device):
    """Mean cross-entropy at each position index over n val examples (-100 targets ignored)."""
    model.eval()
    n = min(n, len(val_ds))
    T = model.config.block_size
    tot = torch.zeros(T)
    cnt = torch.zeros(T)
    for i in range(0, n, batch_size):
        items = [val_ds[j] for j in range(i, min(i + batch_size, n))]
        x = torch.stack([it[0] for it in items]).to(device)                # (B,T)
        y = torch.stack([it[1] for it in items]).to(device)                # (B,T)
        logits, _ = model(x)                                               # (B,T,V)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100, reduction="none").view_as(y)   # (B,T)
        mask = (y != -100).float().cpu()
        tot += (loss.cpu() * mask).sum(0)
        cnt += mask.sum(0)
    return [(tot[t] / cnt[t]).item() if cnt[t] > 0 else None for t in range(T)]


def run_context(args):
    model, tok, ckpt = load_ckpt_or_exit(args)
    train_set = load_train_set(ckpt)
    block = model.config.block_size
    tc = ckpt["train_config"]
    cfg = TrainConfig(data_dir=tc["data_dir"], seed=args.seed)
    for k in ("n_bins", "cond_dropout", "max_rows"):
        if k in tc:
            setattr(cfg, k, tc[k])
    _, val_ds, _, _ = get_datasets(cfg, block)
    print(f"\n== context (n={args.n}) ==")
    ppl = position_loss(model, val_ds, args.n, args.batch_size, args.device)
    valid_pos = [(t, v) for t, v in enumerate(ppl) if v is not None]
    print("per-position val loss (every 10th position):")
    for t, v in valid_pos[::10]:
        print(f"  pos {t:>3}: {v:.4f}")

    val_path = os.path.join(tc["data_dir"], "val.csv")
    val_smiles = [r[0] for r in load_csv(val_path)]
    base = prompt_ids(tok)
    prefix_res = []
    print(f"{'k':>4} {'n':>5} {'valid':>7} {'uniq':>7} {'novel':>7} {'len':>6}")
    for k in ints(args.prefix_lens):
        mols = [s for s in val_smiles if len(s) > k][:args.n]               # distinct, longer than prefix
        if not mols:
            print(f"{k:>4} no validation molecule longer than {k}, skipped")
            continue
        outs = []
        for i in range(0, len(mols), args.batch_size):
            chunk = mols[i:i + args.batch_size]
            start = torch.tensor([base + tok.encode(s[:k]) for s in chunk],
                                 dtype=torch.long, device=args.device)     # (B,T0)
            ids = sample_batch(model, start, block - start.size(1), 1.0, None,
                               tok.eos_id, tok.pad_id)                     # (B,T)
            outs.extend(tok.decode(row.tolist()) for row in ids)
        m = metrics.score(outs, train_set)
        r = {"prefix_len": k, "n": len(outs), "validity": m["validity"], "uniqueness": m["uniqueness"],
             "novelty": m["novelty"], "mean_len": mean_len(m),
             "examples": [{"prefix": s[:k], "completed": o} for s, o in zip(mols[:3], outs[:3])]}
        prefix_res.append(r)
        print(f"{k:>4} {r['n']:>5} {fmt(r['validity']):>7} {fmt(r['uniqueness']):>7} "
              f"{fmt(r['novelty']):>7} {fmt(r['mean_len'], 1):>6}")
    save_json(args.out_dir, "context.json",
              {"n": args.n, "ckpt": args.ckpt, "position_loss": ppl, "prefix": prefix_res})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(*zip(*valid_pos))
    ax1.set_xlabel("position index")
    ax1.set_ylabel("cross-entropy")
    ax1.set_title("loss by position")
    if prefix_res:
        ax2.plot([r["prefix_len"] for r in prefix_res], [r["validity"] for r in prefix_res], marker="o")
    ax2.set_xlabel("prefix length (characters)")
    ax2.set_ylabel("validity")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("validity vs prefix length")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "context.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 3. ablation

def run_axis(args, axis):
    from gpt.train import train
    values = ints(args.values) if args.values else AXIS_DEFAULTS[axis]
    runs_dir = os.path.join(args.out_dir, "ablation_runs")
    base_g = GPTConfig(block_size=args.ab_block_size, n_layer=args.ab_base_layer,
                       n_head=args.ab_base_head, n_embd=args.ab_base_embd, dropout=0.1)
    base_t = TrainConfig(batch_size=args.ab_batch_size, max_steps=args.ab_steps,
                         eval_interval=args.ab_eval_interval, eval_iters=20,
                         eval_samples=args.ab_eval_samples, max_rows=args.ab_rows, fp16=False,
                         max_minutes=args.ab_max_minutes, warmup_steps=50)
    print(f"\n== ablation: {axis} in {values} ==")
    results = []
    for v in values:
        gcfg, cfg = copy.deepcopy(base_g), copy.deepcopy(base_t)
        setattr(gcfg, axis, v)
        if gcfg.n_embd % gcfg.n_head != 0:
            print(f"skipping {axis}={v}: n_embd {gcfg.n_embd} not divisible by n_head {gcfg.n_head}")
            continue
        cfg.run_name = f"ab_{axis}{v}"
        cfg.results_dir = runs_dir
        cfg.out_dir = os.path.join(args.out_dir, "ablation_ckpts")
        cfg.device = args.device
        cfg.seed = args.seed
        train(cfg, gcfg)
        with open(os.path.join(runs_dir, f"{cfg.run_name}_log.json")) as f:
            log = json.load(f)
        vals = [x[1] for x in log["val"]]
        results.append({"axis": axis, "value": v, "params": log["params"],
                        "final_val_loss": vals[-1] if vals else None,
                        "best_val_loss": min(vals) if vals else None,
                        "minutes": log["minutes"],
                        "samples": log["samples"][-1][1] if log["samples"] else None,
                        "val_curve": log["val"]})
    print(f"{'value':>6} {'params':>9} {'final':>8} {'best':>8} {'min':>6} {'valid':>7} {'uniq':>7} {'novel':>7}")
    for r in results:
        s = r["samples"] or {}
        print(f"{r['value']:>6} {r['params']:>9} {fmt(r['final_val_loss'], 4):>8} {fmt(r['best_val_loss'], 4):>8} "
              f"{r['minutes']:>6.1f} {fmt(s.get('validity')):>7} {fmt(s.get('uniqueness')):>7} "
              f"{fmt(s.get('novelty')):>7}")
    save_json(args.out_dir, f"ablation_{axis}.json",
              {"axis": axis, "base": {"model": dataclasses.asdict(base_g), "train": dataclasses.asdict(base_t)},
               "results": results})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        if r["val_curve"]:
            ax1.plot(*zip(*r["val_curve"]), marker="o",
                     label=f"{axis}={r['value']} ({r['params'] / 1e6:.1f}M params)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("val loss")
    ax1.legend()
    pts = [(r["value"], r["final_val_loss"]) for r in results if r["final_val_loss"] is not None]
    if pts:
        ax2.plot(*zip(*pts), marker="o")
    ax2.set_xlabel(axis)
    ax2.set_ylabel("final val loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"ablation_{axis}.png"), dpi=120)
    plt.close(fig)


def run_ablation(args):
    axes = list(AXIS_DEFAULTS) if args.axis == "all" else [args.axis]
    for axis in axes:
        run_axis(args, axis)


# ---------------------------------------------------------------- summary

def _load(out_dir, name):
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_summary(out_dir):
    """Rebuild summary.md from whichever experiment JSONs exist."""
    L = ["# Experiments summary", ""]
    d = _load(out_dir, "temperature.json")
    if d:
        L += ["## 1. Temperature", "", f"{d['n']} unconditional samples per temperature "
              f"(checkpoint `{d['ckpt']}`; T=0 is greedy, so every sample is identical).", "",
              "| T | validity | uniqueness | novelty | qed | logp | mol weight | mean length |",
              "|---|---|---|---|---|---|---|---|"]
        for r in d["results"]:
            L.append(f"| {r['temperature']} | {fmt(r['validity'])} | {fmt(r['uniqueness'])} | {fmt(r['novelty'])} | "
                     f"{fmt(r['qed'])} | {fmt(r['logp'])} | {fmt(r['mol_weight'], 1)} | {fmt(r['mean_len'], 1)} |")
        L += ["", "Examples:", ""]
        for r in d["results"]:
            L.append(f"- T={r['temperature']}: " + ", ".join(f"`{s}`" for s in r["examples"]))
        L += ["", "![temperature](temperature.png)", ""]
    d = _load(out_dir, "context.json")
    if d:
        pl = [v for v in d["position_loss"] if v is not None]
        L += ["## 2. Prompt length and context", "",
              f"Per-position validation loss ({d['n']} examples max): first scored position "
              f"{fmt(pl[0], 3) if pl else 'n/a'}, last {fmt(pl[-1], 3) if pl else 'n/a'}, "
              f"mean {fmt(sum(pl) / len(pl), 3) if pl else 'n/a'}.", "",
              "Completions of real validation prefixes at temperature 1.0:", "",
              "| prefix length | n | validity | uniqueness | novelty | mean length |", "|---|---|---|---|---|---|"]
        for r in d["prefix"]:
            L.append(f"| {r['prefix_len']} | {r['n']} | {fmt(r['validity'])} | {fmt(r['uniqueness'])} | "
                     f"{fmt(r['novelty'])} | {fmt(r['mean_len'], 1)} |")
        L += ["", "Examples (prefix -> completed):", ""]
        for r in d["prefix"]:
            for e in r["examples"]:
                L.append(f"- k={r['prefix_len']}: `{e['prefix']}` -> `{e['completed']}`")
        L += ["", "![context](context.png)", ""]
    axes = [a for a in AXIS_DEFAULTS if _load(out_dir, f"ablation_{a}.json")]
    if axes:
        L += ["## 3. Architecture ablation", ""]
    for a in axes:
        d = _load(out_dir, f"ablation_{a}.json")
        b = d["base"]
        L += [f"### {a}", "", f"Baseline: {b['model']['n_layer']} layers, {b['model']['n_head']} heads, "
              f"n_embd {b['model']['n_embd']}, block {b['model']['block_size']}; {b['train']['max_steps']} steps, "
              f"batch {b['train']['batch_size']}, {b['train']['max_rows']} rows.", "",
              f"| {a} | params | final val loss | best val loss | minutes | validity | uniqueness | novelty |",
              "|---|---|---|---|---|---|---|---|"]
        for r in d["results"]:
            s = r["samples"] or {}
            L.append(f"| {r['value']} | {r['params']} | {fmt(r['final_val_loss'], 4)} | {fmt(r['best_val_loss'], 4)} | "
                     f"{r['minutes']:.1f} | {fmt(s.get('validity'))} | {fmt(s.get('uniqueness'))} | {fmt(s.get('novelty'))} |")
        L += ["", f"![ablation {a}](ablation_{a}.png)", ""]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print(f"wrote {os.path.join(out_dir, 'summary.md')}")


def main():
    p = argparse.ArgumentParser(description="Run the temperature / context / ablation experiments")
    p.add_argument("experiment", choices=["temperature", "context", "ablation", "all"])
    p.add_argument("--ckpt", default="checkpoints/smiles/best.pt")
    p.add_argument("--out_dir", default="results/experiments")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n", type=int, default=500, help="samples per setting")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--temperatures", default="0,0.5,0.7,0.8,1.0,1.2,1.5")
    p.add_argument("--prefix_lens", default="0,5,10,20,40")
    p.add_argument("--axis", default="n_layer", choices=["n_layer", "n_head", "n_embd", "all"])
    p.add_argument("--values", default=None, help="comma-separated ints overriding the axis defaults")
    p.add_argument("--ab_steps", type=int, default=300)
    p.add_argument("--ab_batch_size", type=int, default=32)
    p.add_argument("--ab_rows", type=int, default=15000)
    p.add_argument("--ab_block_size", type=int, default=96)
    p.add_argument("--ab_eval_interval", type=int, default=100)
    p.add_argument("--ab_eval_samples", type=int, default=100)
    p.add_argument("--ab_base_layer", type=int, default=4)
    p.add_argument("--ab_base_head", type=int, default=4)
    p.add_argument("--ab_base_embd", type=int, default=128)
    p.add_argument("--ab_max_minutes", type=float, default=20.0)
    args = p.parse_args()
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    if args.experiment in ("temperature", "all"):
        run_temperature(args)
    if args.experiment in ("context", "all"):
        run_context(args)
    if args.experiment in ("ablation", "all"):
        run_ablation(args)
    write_summary(args.out_dir)


if __name__ == "__main__":
    main()
