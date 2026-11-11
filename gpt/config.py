"""Config dataclasses and CLI helpers (every field is overridable as --field value)."""
import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 0      # set from the tokenizer
    block_size: int = 128    # max context length T
    n_layer: int = 8         # number of Transformer blocks
    n_head: int = 8          # attention heads
    n_embd: int = 256        # embedding width C
    dropout: float = 0.1
    bias: bool = True        # bias in Linear / LayerNorm


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints"
    results_dir: str = "results"
    batch_size: int = 256
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 20000
    max_minutes: float = 150.0
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    seed: int = 1337
    device: str = "auto"
    fp16: bool = True
    run_name: str = "smiles"
    data_dir: str = "data/processed"
    cond_dropout: float = 0.1   # chance each property bin token becomes its "?" variant
    n_bins: int = 10            # quantile bins per property
    eval_samples: int = 500     # molecules generated at every full eval
    smoke: bool = False
    max_rows: int = 0           # cap train/val rows (0 = use all); for fast ablation runs


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def add_config_args(parser, *classes):
    """Add one --field option per dataclass field, defaulting to None (= not given)."""
    for cls in classes:
        for f in dataclasses.fields(cls):
            kind = _str2bool if f.type in (bool, "bool") else {"int": int, "float": float, "str": str}.get(f.type, f.type)
            if kind is _str2bool:
                parser.add_argument(f"--{f.name}", type=_str2bool, nargs="?", const=True, default=None)
            else:
                parser.add_argument(f"--{f.name}", type=kind, default=None)


def config_from_args(args, cls, base=None):
    """Build a `cls` instance: defaults (or `base`), overridden by any args that were given."""
    cfg = base if base is not None else cls()
    for f in dataclasses.fields(cls):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg
