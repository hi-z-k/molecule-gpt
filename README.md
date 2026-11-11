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

## Model configuration

The trained model uses the defaults in `gpt/config.py`.

| setting | value |
| --- | --- |
| `block_size` | 128 |
| `n_layer` | 8 |
| `n_head` | 8 |
| `n_embd` | 256 |
| head dimension (`n_embd / n_head`) | 32 |
| `dropout` | 0.1 |
| vocab size | set from the data: 4 base specials + 22 property-bin specials + the distinct SMILES characters in the training split |

Training settings:

| setting | value |
| --- | --- |
| `batch_size` | 256 |
| optimizer | AdamW, betas 0.9 / 0.95 |
| `lr` / `min_lr` | 3e-4 / 3e-5 (cosine decay) |
| `weight_decay` | 0.1 |
| `warmup_steps` | 200 |
| `grad_clip` | 1.0 |
| `max_minutes` | 150 (training is time-boxed; stops here or at `max_steps`, whichever comes first) |

The parameter count is printed at the start of training and recorded in
`results/smiles_log.json`. Any field above can be overridden as a `--flag`.

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
| `scripts/experiments.py` | the three experiments: temperature, context, architecture ablation |
| `notebooks/train_kaggle.ipynb` | the full run on a Kaggle T4 |
| `notebooks/experiments_kaggle.ipynb` | runs those experiments on a Kaggle T4, reusing the trained checkpoint |

Validity is also measured on fresh samples at every eval step, so the training
log shows the grammar being acquired rather than just the loss going down.

## Experiments

`scripts/experiments.py` runs three experiments on the model:

1. `temperature` — validity, uniqueness and novelty of unconditional samples
   across sampling temperatures. Temperature 0 is greedy decoding, so every
   sample is identical.
2. `context` — two measurements. The first is mean cross-entropy at each
   position index over the validation set. The second is the validity of
   completions of real validation-molecule prefixes of increasing length
   (0, 5, 10, 20 and 40 characters).
3. `ablation` — trains small models from scratch, varying one architectural
   parameter at a time (`--axis n_layer|n_head|n_embd|all`), with everything
   else and the data subset held fixed.

```bash
python scripts/experiments.py temperature --n 2000
python scripts/experiments.py context --n 1000
python scripts/experiments.py ablation --axis all
```

`temperature` and `context` need a trained checkpoint (`--ckpt`, default
`checkpoints/smiles/best.pt`); `ablation` does not. Each experiment writes a
JSON of raw numbers plus a PNG into `--out_dir` (default `results/experiments`),
and `summary.md` there collects every table in markdown.

The ablation runs are short and deliberately small. They are controlled
comparisons between settings, not converged models, so they support claims
about relative trends only. The `--ab_*` flags (`--ab_steps`, `--ab_rows`,
`--ab_batch_size`, `--ab_block_size`, `--ab_max_minutes`, and the
`--ab_base_*` baseline flags) exist to rerun them larger on a GPU.

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

## Technical report

`REPORT.md` explains the Transformer architecture from "Attention Is All You
Need" and how this implementation relates to it.
