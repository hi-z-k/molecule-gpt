import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.tokenizer import CharTokenizer
from mol import metrics
from mol.data import (SmilesDataset, bin_of, canonical, load_csv, prepare, property_specials)

# 31 real molecules (solvents, simple rings, a few drugs) - enough to form quantile bins
FIXTURE = [
    "CCO", "CCC", "CCCC", "c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CCN(CC)CC", "c1ccncc1", "OCC1OC(O)C(O)C(O)C1O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(N)C(=O)O", "NCCc1ccc(O)c(O)c1", "c1ccc2ccccc2c1",
    "CCOC(=O)C", "CC(C)O", "ClCCl", "CCBr", "c1ccc(cc1)O", "Nc1ccccc1",
    "CC(=O)Nc1ccc(O)cc1", "CCCCCCCC", "O=C(O)c1ccccc1O",
    "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21", "CC1=CC(=O)CC(C)(C)C1", "CCOCC", "C1CCCCC1",
    "N#Cc1ccccc1", "OC(=O)CCCCC", "CSC", "C1COCCO1",
]


def write_fixture(tmp_path):
    path = tmp_path / "smiles.txt"
    path.write_text("\n".join(FIXTURE) + "\n")
    return str(path)


def test_tokenizer_extra_specials_roundtrip():
    extras = property_specials(3)
    tok = CharTokenizer("CON=#()1", extras)
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.sep_id == 2 and tok.eos_id == 3
    assert tok.specials[:4] == CharTokenizer.SPECIALS and tok.specials[4:] == extras
    assert tok.id_of("<qed0>") == 4 and tok.id_of("<logp?>") == 4 + len(extras) - 1
    assert tok.special_ids == set(range(4 + len(extras)))
    ids = tok.encode("CC(=O)O")
    assert tok.decode([tok.bos_id, tok.id_of("<qed1>"), tok.id_of("<logp?>")] + ids
                      + [tok.eos_id, tok.pad_id]) == "CC(=O)O"
    tok2 = CharTokenizer.from_dict(tok.to_dict())
    assert tok2.itos == tok.itos
    assert CharTokenizer.from_dict({"chars": ["a", "b"]}).vocab_size == 6  # old checkpoints


def test_canonical():
    assert canonical("C(C)O") == "CCO"
    assert canonical("c1ccccc1") == canonical("C1=CC=CC=C1")
    assert canonical("not a smiles") is None
    assert canonical("C(C(C") is None


def test_bin_of_clamps():
    edges = [0.0, 1.0, 2.0]
    assert bin_of(-5.0, edges) == 0
    assert bin_of(0.5, edges) == 1
    assert bin_of(99.0, edges) == 3


def test_prepare_end_to_end(tmp_path):
    out = tmp_path / "processed"
    meta = prepare(write_fixture(tmp_path), str(out), max_len=100, n_bins=4, val_frac=0.2, seed=7)
    assert os.path.exists(out / "train.csv") and os.path.exists(out / "val.csv")
    assert json.load(open(out / "meta.json")) == meta
    assert len(meta["qed_edges"]) == len(meta["logp_edges"]) == 3   # n_bins - 1 interior edges
    assert meta["n_bins"] == 4 and meta["max_len"] == 100
    assert meta["n_train"] + meta["n_val"] == len(FIXTURE)
    with open(out / "train.csv") as f:
        assert next(csv.reader(f)) == ["smiles", "qed", "logp"]
    rows = load_csv(str(out / "train.csv"))
    assert rows and all(0.0 <= r[1] <= 1.0 for r in rows)
    assert all(canonical(r[0]) == r[0] for r in rows)               # stored canonical


def test_dataset_shapes_and_loss_mask(tmp_path):
    out = tmp_path / "processed"
    meta = prepare(write_fixture(tmp_path), str(out), n_bins=4, val_frac=0.2, seed=7)
    rows = load_csv(str(out / "train.csv"))
    tok = CharTokenizer(sorted({c for r in rows for c in r[0]}), property_specials(meta["n_bins"]))
    block = 64
    ds = SmilesDataset(rows, tok, meta, block, train=False)
    assert len(ds) == len(rows)
    x, y = ds[0]
    assert x.shape == y.shape == (block,) and x.dtype == y.dtype == torch.int64
    n = 3 + len(rows[0][0]) + 1                                    # bos + 2 bins + chars + eos
    assert x[0] == tok.bos_id and int(x[1]) in tok.special_ids and int(x[2]) in tok.special_ids
    assert (y[:2] == -100).all()                                   # the two bin tokens
    assert (y[2:n - 1] != -100).all()                              # chars and eos are predicted
    assert y[n - 2] == tok.eos_id
    assert (y[n - 1:] == -100).all()                               # padding
    assert (y[x == tok.pad_id] == -100).all()

    # conditioning dropout only happens when train=True
    train_ds = SmilesDataset(rows, tok, meta, block, train=True, cond_dropout=1.0)
    xt, _ = train_ds[0]
    assert int(xt[1]) == tok.id_of("<qed?>") and int(xt[2]) == tok.id_of("<logp?>")


def test_metrics_score():
    gen = ["CCO", "CCO", "not a smiles", "c1ccccc1"]
    m = metrics.score(gen, {"CCO"})
    assert m["n"] == 4
    assert m["validity"] == 3 / 4
    assert m["uniqueness"] == 2 / 3                                # {CCO, c1ccccc1} of 3 valid
    assert m["novelty"] == 1 / 3                                   # only benzene is unseen
    assert m["valid_canonical"] == ["CCO", "CCO", "c1ccccc1"]
    assert m["rings"] == 1 / 3 and m["mol_weight"] > 0
    assert metrics.score(gen)["novelty"] is None
    assert metrics.score([])["validity"] == 0.0
