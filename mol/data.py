"""SMILES data: download, canonicalize, property bins, and the training Dataset.

Sequence format, padded to block_size:
    <bos> <qed_k> <logp_k> c h a r s ... <eos> <pad> <pad> ...
Loss is taken only where the target is a SMILES character or <eos>.
"""
import csv
import json
import os
import random

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED

RDLogger.DisableLog("rdApp.*")

DEFAULT_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"


def download(url, dest):
    """Stream `url` into `dest`; skip if the file is already there and non-empty."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"{dest} already present, skipping download")
        return dest
    import requests
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def canonical(smiles):
    """Canonical SMILES string, or None if RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def props(mol):
    """(qed, logp) of an RDKit mol."""
    return QED.qed(mol), Crippen.MolLogP(mol)


def property_specials(n_bins):
    """Special token strings for the two property bins, plus their "unspecified" variants."""
    qed = [f"<qed{i}>" for i in range(n_bins)] + ["<qed?>"]
    logp = [f"<logp{i}>" for i in range(n_bins)] + ["<logp?>"]
    return qed + logp


def bin_of(value, edges):
    """Index of the quantile bucket `value` falls into, clamped to [0, len(edges)]."""
    i = int(np.searchsorted(np.asarray(edges), value, side="right"))
    return max(0, min(i, len(edges)))


def _quantile_edges(values, n_bins):
    """The n_bins-1 interior quantile edges that cut `values` into n_bins equal buckets."""
    qs = [i / n_bins for i in range(1, n_bins)]
    return [float(v) for v in np.quantile(np.asarray(values, dtype=np.float64), qs)]


def _read_source(path, limit=None):
    """Yield (smiles, split) from a .csv with a SMILES column or a .txt of one per line."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".csv"):
            reader = csv.DictReader(f)
            names = {k.lower(): k for k in (reader.fieldnames or [])}
            smi_key = names.get("smiles") or (reader.fieldnames or [None])[0]
            split_key = names.get("split")
            for n, row in enumerate(reader):
                if limit is not None and n >= limit:
                    break
                yield row[smi_key], (row[split_key] if split_key else None)
        else:
            for n, line in enumerate(f):
                if limit is not None and n >= limit:
                    break
                line = line.strip().split()[0] if line.strip() else ""
                if line:
                    yield line, None


def prepare(source, out_dir, max_len=100, n_bins=10, limit=None, val_frac=0.05, seed=1337):
    """Canonicalize `source`, drop/dedup, compute properties and bins, write out_dir/*.csv + meta.json."""
    rng = random.Random(seed)
    rows = {"train": [], "val": []}      # split -> [(smiles, qed, logp)]
    seen = set()
    n_in = n_bad = n_long = n_dup = 0
    lengths = []
    for smiles, split in _read_source(source, limit):
        n_in += 1
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_bad += 1
            continue
        smi = Chem.MolToSmiles(mol)
        if len(smi) > max_len:
            n_long += 1
            continue
        if smi in seen:
            n_dup += 1
            continue
        seen.add(smi)
        lengths.append(len(smi))
        qed, logp = props(mol)
        if split is None:
            which = "val" if rng.random() < val_frac else "train"
        else:
            which = "train" if split.strip().lower() == "train" else "val"
        rows[which].append((smi, qed, logp))

    if not rows["train"]:
        raise ValueError(f"no usable training molecules in {source}")
    qed_edges = _quantile_edges([r[1] for r in rows["train"]], n_bins)
    logp_edges = _quantile_edges([r[2] for r in rows["train"]], n_bins)

    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val"):
        with open(os.path.join(out_dir, f"{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["smiles", "qed", "logp"])
            for smi, qed, logp in rows[split]:
                w.writerow([smi, f"{qed:.6f}", f"{logp:.6f}"])
    meta = {"n_bins": n_bins, "max_len": max_len, "source": str(source),
            "qed_edges": qed_edges, "logp_edges": logp_edges,
            "n_train": len(rows["train"]), "n_val": len(rows["val"])}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pcts = np.percentile(lengths, [50, 90, 99]) if lengths else [0, 0, 0]
    print(f"read {n_in} rows: kept {len(lengths)} "
          f"(dropped {n_bad} unparseable, {n_long} longer than {max_len}, {n_dup} duplicates)")
    print(f"train {len(rows['train'])} / val {len(rows['val'])}; "
          f"length p50 {pcts[0]:.0f} p90 {pcts[1]:.0f} p99 {pcts[2]:.0f} max {max(lengths or [0])}")
    print(f"wrote {out_dir}/train.csv, val.csv, meta.json")
    return meta


def load_csv(path):
    """Read a prepared csv -> [(smiles, qed, logp)]."""
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((row["smiles"], float(row["qed"]), float(row["logp"])))
    return out


class SmilesDataset(torch.utils.data.Dataset):
    """Tokenized (x, y) pairs of length block_size, with conditioning dropout when train=True."""

    def __init__(self, rows, tok, meta, block_size, train=True, cond_dropout=0.1, seed=1337):
        self.rows = rows
        self.tok = tok
        self.block_size = block_size
        self.train = train
        self.cond_dropout = cond_dropout
        self.qed_edges = meta["qed_edges"]
        self.logp_edges = meta["logp_edges"]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def prompt_ids(self, qed_bin=None, logp_bin=None):
        """[<bos>, <qed_*>, <logp_*>]; None means the "?" (unspecified) variant."""
        q = "<qed?>" if qed_bin is None else f"<qed{qed_bin}>"
        p = "<logp?>" if logp_bin is None else f"<logp{logp_bin}>"
        return [self.tok.bos_id, self.tok.id_of(q), self.tok.id_of(p)]

    def __getitem__(self, i):
        smi, qed, logp = self.rows[i]
        qb = bin_of(qed, self.qed_edges)
        pb = bin_of(logp, self.logp_edges)
        if self.train:
            if self.rng.random() < self.cond_dropout:
                qb = None
            if self.rng.random() < self.cond_dropout:
                pb = None
        ids = self.prompt_ids(qb, pb) + self.tok.encode(smi) + [self.tok.eos_id]
        ids = ids[:self.block_size + 1]
        n = len(ids)
        ids = ids + [self.tok.pad_id] * (self.block_size + 1 - n)
        seq = torch.tensor(ids, dtype=torch.long)                          # (T+1,)
        x = seq[:self.block_size].clone()                                  # (T,)
        y = seq[1:self.block_size + 1].clone()                             # (T,) shifted by one
        # only predict SMILES characters and <eos>: the two bin tokens and the padding are masked
        y[:2] = -100
        y[n - 1:] = -100
        return x, y


def get_datasets(cfg, block_size):
    """Read cfg.data_dir -> (train_ds, val_ds, tokenizer, meta)."""
    meta_path = os.path.join(cfg.data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found; run scripts/prepare_data.py --out_dir {cfg.data_dir} first")
    with open(meta_path) as f:
        meta = json.load(f)
    train_rows = load_csv(os.path.join(cfg.data_dir, "train.csv"))
    val_rows = load_csv(os.path.join(cfg.data_dir, "val.csv"))
    if getattr(cfg, "smoke", False):
        train_rows, val_rows = train_rows[:2000], val_rows[:200]
    # cap dataset size for quick ablation runs
    max_rows = int(getattr(cfg, "max_rows", 0) or 0)
    if max_rows > 0:
        train_rows = train_rows[:max_rows]
        val_rows = val_rows[:max(1, max_rows // 20)]
    chars = set()
    for smi, _, _ in train_rows:
        chars.update(smi)
    from gpt.tokenizer import CharTokenizer
    tok = CharTokenizer(sorted(chars), property_specials(meta["n_bins"]))
    cond_dropout = getattr(cfg, "cond_dropout", 0.1)
    train_ds = SmilesDataset(train_rows, tok, meta, block_size, True, cond_dropout, cfg.seed)
    val_ds = SmilesDataset(val_rows, tok, meta, block_size, False, cond_dropout, cfg.seed)
    return train_ds, val_ds, tok, meta
