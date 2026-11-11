# Technical Report — A GPT-2-style Transformer From Scratch

> **Status.** Section C (implementation) is written from the code in this repo.
> Sections A, B and D are scaffolded with the questions they must answer, plus
> pointers to the evidence in this repo. Numbers marked `TBD` are filled from
> `results/experiments/summary.md` and `results/smiles_log.json` once the runs finish.

**Model.** A decoder-only Transformer trained on SMILES strings — text representations of
molecules, e.g. aspirin is `CC(=O)OC1=CC=CC=C1C(=O)O`. The task is next-character prediction;
the interesting part is that the model must learn a strict grammar (brackets close, ring-opening
digits pair, valences stay legal) to score at all, and RDKit can then check mechanically whether
each generated string is real chemistry.

---

## A. Understanding *Attention Is All You Need*

### A.1 What problem was the Transformer designed to address?

*To write.* Points to cover: sequential computation in RNNs prevents parallelism across
positions within a training example; path length between distant tokens grows with distance,
so gradients must travel far; the paper's claim is that attention alone gives constant path
length between any two positions and full parallelism over the sequence.

### A.2 How does the Transformer differ from RNNs and LSTMs?

*To write.* Contrast: recurrent hidden state vs. direct all-pairs interaction; O(n) sequential
steps vs. O(1); fixed-size bottleneck state vs. a representation per position; why LSTM gating
mitigates but does not remove the vanishing-gradient/path-length problem.

### A.3 How does scaled dot-product attention work?

*To write*, grounded in the equation implemented at `gpt/model.py:52-65`:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

Cover: queries as "what this position is looking for", keys as "what each position offers",
values as "what gets passed on if matched"; the dot product as unnormalized similarity; why the
$1/\sqrt{d_k}$ scaling is needed (for large $d_k$ the dot products grow in variance, pushing
softmax into saturated regions with vanishing gradients); softmax turning scores into a convex
combination over positions.

### A.4 Why is attention divided into multiple heads?

*To write.* A single softmax produces one averaged mixture, which blurs distinct relations.
$h$ heads each attend in a $d_k = d_{\text{model}}/h$ subspace, so different heads can track
different relations at the same total cost. **This repo has direct evidence:** the `n_head`
ablation sweeps 2/4/8 heads at fixed `n_embd`, so it varies only how attention is partitioned,
not capacity — see section D.

### A.5 What are positional encodings and why are they necessary?

*To write.* Attention is permutation-equivariant: without position information, `C1CCCCC1` and
a shuffling of those characters would receive identical representations, yet ring-closure digits
only make sense positionally. Contrast the paper's fixed sinusoidal encodings with the **learned**
positional embeddings this repo uses (`gpt/model.py:106`, following GPT-2) — see section B.5.

### A.6 What roles do feed-forward networks, layer norm and residual connections play?

*To write.* FFN: attention mixes information across positions but is linear in the values, so the
position-wise MLP supplies per-position non-linear transformation (`gpt/model.py:67-81`).
LayerNorm: stabilizes the scale of activations per token (`gpt/model.py:13-30`). Residuals: give
an identity path so gradients reach early layers and each block learns a *refinement* of the
residual stream rather than a replacement (`gpt/model.py:94-95`).

---

## B. From the Original Transformer to GPT-2

### B.1 The encoder-decoder architecture

*To write.* The 2017 paper targets translation: an encoder builds a bidirectional
representation of the source, a decoder generates the target while attending both to its own
prefix (masked self-attention) and to the encoder output (cross-attention).

### B.2 Why GPT-2 is decoder-only

*To write.* Language modelling has no separate source sequence — the context *is* the prefix.
With nothing to encode, the encoder stack and the cross-attention sublayer have no input, so
only the masked-self-attention decoder remains.

### B.3 Masked vs. unmasked self-attention vs. cross-attention

*To write.* Suggested table: what attends to what, which mask applies, and where each is used.

| variant | Q from | K,V from | mask | used in |
| --- | --- | --- | --- | --- |
| masked self-attention | the sequence itself | the sequence itself | causal (lower-triangular) | GPT-2; this repo |
| unmasked self-attention | the sequence itself | the sequence itself | none (full bidirectional) | original encoder; BERT |
| cross-attention | decoder | encoder output | none over the source | original decoder; absent here |

### B.4 How causal masking enables autoregressive generation

*To write*, grounded in `gpt/model.py:49-50` and `:59`. The key point: the mask makes position
$t$'s prediction depend only on positions $\le t$, so a single forward pass over a length-$T$
sequence yields $T$ next-token predictions that are all valid simultaneously — the model trains
on every position at once, yet at inference the same weights generate strictly left-to-right.
Without the mask the model could read the answer from the future and the loss would be trivially
minimized while generation stayed impossible.

### B.5 Main architectural differences: original Transformer vs. GPT-2

*To write*, using this table (all "this repo" entries verified against the code):

| aspect | original Transformer (2017) | GPT-2 / this repo |
| --- | --- | --- |
| structure | encoder + decoder | decoder only (`gpt/model.py:99`) |
| positional info | fixed sinusoidal | **learned** embedding `wpe` (`gpt/model.py:106`) |
| norm placement | **post**-LN: `x + Sublayer(x)` then norm | **pre**-LN: `x + Sublayer(LN(x))` (`gpt/model.py:94-95`) |
| final norm | — | extra `ln_f` before the output head (`gpt/model.py:109`) |
| activation | ReLU | GELU (`gpt/model.py:78`) |
| output head | separate projection | **tied** to the token embedding (`gpt/model.py:111`) |
| residual init | — | projections scaled by $1/\sqrt{2N}$ (`gpt/model.py:115-117`) |

Why pre-LN matters is worth a sentence: with post-LN the residual path passes through a
normalization every block, so deep stacks need careful warmup; pre-LN keeps an unnormalized
identity path from input to output, which trains more stably.

---

## C. Implementation Explanation

### C.1 What one training sequence looks like

Each molecule becomes one fixed-length sequence (`mol/data.py`, `SmilesDataset.__getitem__`):

```
<bos> <qed_k> <logp_k>  C  C  (  =  O  )  O  c  1  c  c  c  c  c  1  …  <eos> <pad> <pad> …
└──────── prompt ─────┘ └──────────────── SMILES characters ──────────────┘  └── padding ──┘
```

`<qed_k>` and `<logp_k>` are quantile-bin tokens for drug-likeness (QED) and lipophilicity
(logP), computed per molecule by RDKit and binned on the training split. Targets are the inputs
shifted left by one. Loss is masked to `-100` on the two property tokens and on padding, so the
conditioning tokens act as a **prompt, not a prediction target**. During training each bin token
is independently replaced by its `<qed?>` / `<logp?>` variant with probability `cond_dropout`,
which is what leaves one model able both to generate unconditionally and to be steered.

**Tokenizer** (`gpt/tokenizer.py`): character-level. The vocabulary is 4 base specials
(`<pad> <bos> <sep> <eos>`), then the property-bin specials, then the sorted set of characters
observed in the training SMILES, so `V` is data-derived rather than fixed:
`V = 4 + 22 + (distinct characters)`. On the 16.5k-molecule subset 25 characters occur, giving
`V = 51`; confirm against the `vocab=` figure printed at the start of the main run. `encode` maps a
string to ids and drops unknown characters; `decode` inverts it, skipping specials. Character
level is the honest choice here: SMILES has no word structure, and a subword vocabulary would
let the model memorize fragments instead of learning the grammar character by character.

### C.2 Data flow, with tensor shapes

Notation: `B` batch, `T` sequence length, `C = n_embd`, `H = n_head`, `D = C/H` head dimension,
`V` vocabulary size. Current configuration: `B=256, T=128, C=256, H=8, D=32, n_layer=8`, with `V` as in C.1.

```
                 idx (B,T) int64  token ids
                   │
      ┌────────────┴────────────┐
      │  wte(idx)      (B,T,C)  │  token embedding      (V,C) lookup
      │  wpe(pos)        (T,C)  │  learned position embedding, broadcast over B
      └────────────┬────────────┘
                   │  x = wte + wpe   (B,T,C)      ← the "residual stream"
                   │  dropout
    ╔══════════════╪══════════════════════════════════════════════╗
    ║  Block × n_layer                                            ║
    ║              │                                              ║
    ║      ┌───────┴────────┐                                     ║
    ║      │ LayerNorm      │ (B,T,C)                             ║
    ║      │ CausalSelfAttn │ (B,T,C)  ── see C.3                 ║
    ║      └───────┬────────┘                                     ║
    ║        x = x + ─┘            residual   (B,T,C)             ║
    ║              │                                              ║
    ║      ┌───────┴────────┐                                     ║
    ║      │ LayerNorm      │ (B,T,C)                             ║
    ║      │ MLP: C→4C→C    │ (B,T,4C) → GELU → (B,T,C)           ║
    ║      └───────┬────────┘                                     ║
    ║        x = x + ─┘            residual   (B,T,C)             ║
    ╚══════════════╪══════════════════════════════════════════════╝
                   │
            ln_f   │ (B,T,C)      final LayerNorm
            lm_head│ (B,T,V)      tied to wte.weight
                   ▼
              logits (B,T,V)   logits[b,t,:] = scores for the token after position t
```

### C.3 Inside causal self-attention

From `gpt/model.py:52-65`, for one block:

| step | operation | shape |
| --- | --- | --- |
| input | `x` | `(B,T,C)` |
| project | `qkv(x)` | `(B,T,3C)` |
| split | `q, k, v` | `3 × (B,T,C)` |
| reshape to heads | `view(B,T,H,D).transpose(1,2)` | `3 × (B,H,T,D)` |
| scores | `q @ kᵀ / √D` | `(B,H,T,T)` |
| causal mask | `masked_fill(tril == 0, -inf)` | `(B,H,T,T)` |
| normalize | `softmax(dim=-1)` | `(B,H,T,T)`, rows sum to 1 |
| gather values | `weights @ v` | `(B,H,T,D)` |
| merge heads | `transpose(1,2).view(B,T,C)` | `(B,T,C)` |
| output projection | `proj(...)` | `(B,T,C)` |

Two implementation notes worth making in the report:

1. **Multi-head is a reshape, not a loop.** All `H` heads are computed in one batched matmul by
   folding the head axis next to the batch axis. This is exactly the paper's
   $\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O$, with the concatenation realized
   as the inverse reshape and $W^O$ as `proj`.
2. **The mask is applied before the softmax, with `-inf`.** Masking after the softmax would
   leave the rows not summing to 1. Setting pre-softmax scores to $-\infty$ makes
   $e^{-\infty}=0$, so future positions receive exactly zero weight and the remaining weights
   still form a proper distribution.

### C.4 Loss

`F.cross_entropy` over `logits.view(-1,V)` against `targets.reshape(-1)` with
`ignore_index=-100` (`gpt/model.py:150`). Because of the causal mask, one forward pass
scores every position at once: the loss is the mean negative log-likelihood of the true next
character over all unmasked positions, i.e. exactly the autoregressive factorization
$p(x) = \prod_t p(x_t \mid x_{<t})$.

### C.5 Generation

`gpt/generate.py` / `mol/sample.py`: feed the prompt, take `logits[:, -1, :]`, optionally divide
by temperature and restrict to top-k, softmax, sample (or `argmax` when `temperature == 0`),
append, repeat until `<eos>` or `block_size`. The context is cropped to the last `block_size`
tokens, which is what makes the learned `wpe` table of fixed size sufficient. `mol/sample.py`
does this batched, tracking which rows have emitted `<eos>` and padding them thereafter.

---

## D. Experiments and Findings

Produced by `scripts/experiments.py`; every table below is in
`results/experiments/summary.md` ready to paste.

### D.1 Training and validation loss

Main run: `TBD` steps, final train `TBD`, final val `TBD`, `TBD` M parameters,
`TBD` minutes on a Kaggle T4. Curve: `results/smiles_loss.png`.

Validity is also measured on freshly generated samples at every eval step, so the log shows the
*grammar being acquired* rather than only the loss falling — worth discussing as the more
meaningful signal.

| metric | score |
| --- | --- |
| validity | `TBD` |
| uniqueness | `TBD` |
| novelty | `TBD` |

### D.2 Effect of temperature

`TBD` — table from `summary.md`. Expected shape of the result, to confirm or refute against the
numbers: temperature rescales logits before the softmax, so `T < 1` sharpens toward the argmax
and `T > 1` flattens the distribution. Predict validity high and uniqueness low at low `T`, and
the reverse at high `T`; `T = 0` is greedy, so every sample from the same prompt is byte-identical
and uniqueness collapses to a single molecule. Discuss where the useful trade-off sits.

### D.3 Effect of prompt length and context

Two measurements:

1. **Per-position loss** — mean cross-entropy at each position index over the validation set.
   Tests whether the next character genuinely becomes easier to predict as more of the molecule
   is visible. `TBD`
2. **Prefix priming** — completions of real validation-molecule prefixes of length 0/5/10/20/40,
   scored for validity. Tests whether a longer committed prefix constrains the model into
   finishing a legal molecule, or instead traps it (a prefix with two open rings must be closed,
   and the model may not have the budget). `TBD`

### D.4 Effect of architectural parameters

Three one-at-a-time sweeps, all other hyperparameters and the data subset fixed:

| axis | values | result |
| --- | --- | --- |
| `n_layer` | 2, 4, 8 | `TBD` |
| `n_head` | 2, 4, 8 (fixed `n_embd`) | `TBD` |
| `n_embd` | 64, 128, 256 | `TBD` |

**Caveat to state plainly:** these are 1200-step runs on a 200k-molecule subset, capped at 15
minutes each. They are a *controlled comparison between settings*, not converged models, so they
support claims about relative trends and not about best achievable quality.

Note for the `n_head` sweep: `n_embd` is held fixed, so more heads means smaller $d_k$. Any
non-monotonic result is therefore informative — it suggests a trade-off between having more
attention subspaces and having enough dimensions in each.

### D.5 Limitations, challenges and lessons learned

*To write.* Candidate material:

- **Everything generated is a candidate, not a drug.** Valid SMILES resembling the drug-like
  distribution says nothing about whether a molecule binds a target, is synthesizable, or is
  safe. That needs docking and lab work.
- Character-level modelling means no guarantee of validity — the grammar is learned, not
  enforced — which is precisely why validity is a *measurement* here rather than an assumption.
- Compute discipline: the ablation was first attempted on CPU and abandoned, because at 300 steps
  validity was still 0.000 and a comparison between such models would have measured noise.
- Any implementation bugs found and how (e.g. the pre-cap row count that was being logged while
  training on a capped subset).
