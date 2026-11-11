# molecule-gpt

A character-level GPT, written from scratch in PyTorch, that generates molecules.

Molecules can be written as text. SMILES does exactly that — aspirin is
`CC(=O)OC1=CC=CC=C1C(=O)O` — so a language model can learn chemistry as a
string-prediction problem. This repo trains one on ~1.6M drug-like molecules
and then checks, with RDKit, whether what it invents is real chemistry.

The point is not scale. It is that a small transformer trained only to predict
the next character has to learn a strict grammar to score at all: brackets must
close, ring-opening digits must pair up (`C1...C1`), and every atom has to end
up with a legal valence. RDKit either parses the output or it does not, which
makes the result a number rather than an opinion.

## Results

| metric | meaning | score |
| --- | --- | --- |
| validity | generated strings RDKit can parse as a molecule | _see `results/`_ |
| uniqueness | distinct molecules among the valid ones | _see `results/`_ |
| novelty | valid molecules not present in the training set | _see `results/`_ |

Property steering (section 4 of the notebook) asks the model for a given
drug-likeness and measures whether the molecules it returns actually move in
that direction.

## Quickstart

```bash
pip install -r requirements.txt

# download MOSES, canonicalize, compute QED/logP, write train/val splits
python scripts/prepare_data.py --out_dir data/processed

# train (time-boxed; stops on --max_steps or --max_minutes, whichever comes first)
python -m gpt.train --max_minutes 100

# generate 5000 molecules and score them
python -m mol.sample --n 5000 --out results/uncond.json --grid results/grid.png
```

Any config field is a flag: `--n_layer 12 --block_size 160 --batch_size 128`.
Add `--limit 20000` to `prepare_data.py` for a fast end-to-end check.

## How it works

Each molecule is trained as one sequence:

```
<bos> <qed_k> <logp_k> C C ( = O ) O c 1 c c c c c 1 ... <eos>
```

`<qed_k>` and `<logp_k>` are quantile-bin tokens for drug-likeness (QED) and
lipophilicity (logP), computed by RDKit and binned on the training split. Loss
is applied only to the SMILES characters and the final `<eos>`, so the
conditioning tokens are a prompt, not a prediction target. During training each
bin token is independently dropped to `<qed?>` / `<logp?>` with probability
`--cond_dropout`, which leaves the same model able to generate unconditionally
and to be steered:

```bash
python -m mol.sample --target_qed 0.9 --n 1000   # ask for drug-like
python -m mol.sample --qed_bin 0 --n 1000        # ask for the opposite end
```

| file | what it holds |
| --- | --- |
| `gpt/model.py` | the transformer: causal self-attention, MLP blocks, weight tying |
| `gpt/train.py` | training loop, cosine schedule with warmup, AMP, time limit |
| `gpt/generate.py` | sampling, beam search, checkpoint loading |
| `gpt/tokenizer.py` | character vocabulary plus the property special tokens |
| `mol/data.py` | MOSES download, canonicalization, property binning, dataset |
| `mol/metrics.py` | validity / uniqueness / novelty, QED, logP, molecular weight |
| `mol/sample.py` | batched generation CLI and scoring report |
| `notebooks/train_kaggle.ipynb` | the full run on a Kaggle T4 |

Validity is also measured on fresh samples at every eval step, so the training
log shows the grammar being acquired rather than just the loss going down.

## Why this is worth doing

There are roughly 10^60 plausible drug-like molecules — far more than anyone can
synthesize or test. A generative model trained on known drug-like chemistry
proposes new structures that look like they belong, so chemists can test a short
list instead of an infinite one. This is a small, honest version of what
MolGPT and AstraZeneca's REINVENT do in real hit-finding pipelines.

## The honest limit

Everything this model produces is a *candidate*. Valid SMILES that resembles the
drug-like distribution is not a drug: nothing here shows any molecule binds a
target, is synthesizable, or is safe. That takes docking simulations and lab
work. What is demonstrated is narrower and real — the model learned SMILES
grammar and the shape of drug-like chemical space from next-character
prediction alone, and can be aimed at a region of it.

## Data

[MOSES](https://github.com/molecularsets/moses) (Polykovskiy et al.), a curated
drug-like subset of ZINC, with the authors' official train/test split. Downloaded
by `scripts/prepare_data.py`; nothing large is committed to this repo.

## License

MIT — see `LICENSE`.
