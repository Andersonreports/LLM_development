# tinygpt - a language model built from scratch

Roadmap track A of the ADVAT LLM project: a small GPT-style (decoder-only, causal)
Transformer written in PyTorch, starting from **random weights**. No pretrained model is used.

New to these ideas? Read the **[Learning Guide](LEARNING_GUIDE.md)**: every concept used
here, explained in plain language with numbers from our own runs.

## Status

| Roadmap step | State |
|---|---|
| Phase 1 - environment (Python, PyTorch + CUDA, RTX 3050 verified) | done |
| Phase 2 - tokenizer + Transformer architecture | done |
| Phase 3 - training pipeline (corpus, training loop, checkpoints) | done - first run below |
| Phase 4 - text generation + evaluation | `generate.py` done; systematic evaluation next |

## Train and generate

```powershell
.venv\Scripts\python prepare_data.py                   # download Tiny Shakespeare, tokenizer, train/val split
.venv\Scripts\python train.py                          # small model, 5000 steps (~15 min on the RTX 3050)
.venv\Scripts\python train.py --resume                 # continue an interrupted run
.venv\Scripts\python generate.py --prompt "ROMEO:" --tokens 300 --temperature 0.8
```

Checkpoints and `log.csv` (loss per evaluation) go to `checkpoints/<name>/`; `best.pt` is
the lowest-validation-loss state and is what `generate.py` loads by default.

## First run: Tiny Shakespeare, character-level (2026-10-23)

Data: 1,115,394 characters, 65-character vocabulary, last 10% held out for validation
(`data/shakespeare/meta.json` records source, licence and checksum).
Model: `small` preset, 10.7M parameters, context 256, dropout 0.2, batch 64, AdamW,
lr 1e-3 cosine to 1e-4, bf16. 5,000 steps in 15.4 min, 1.74 GB GPU memory.

| Step | Train loss | Val loss | Sample |
|---|---|---|---|
| 0 | 4.30 | 4.30 | random characters (theory: ln 65 = 4.17) |
| 250 | 1.99 | 2.10 | "I lon bead get thy lave diver bith;" - word shapes, play layout |
| 1,000 | 1.26 | 1.53 | "Your sword, and he desire her best." - real words, character names |
| **1,750** | **1.09** | **1.47** | **best validation loss -> `best.pt`** |
| 5,000 | 0.60 | 1.72 | fluent, but partly memorised training text |

**Overfitting from step ~1,750**: training loss kept falling while validation loss rose,
so the model started memorising the training text instead of generalising. `best.pt`
keeps the step-1,750 weights. Remedies to try: stop earlier, more dropout, a smaller
model, or more data (1 MB is very little for 10.7M parameters).

Generated from `best.pt` with the prompt `ROMEO:`:

```
ROMEO:
If the king my chief stand hath need to my husband,
So in the advantage of his story
Would I live out of a grave-market mourn.
```

Correct spelling, metre-like lines and play formatting - and no meaning. This is the
roadmap's key lesson in practice: fluent output is not evidence of understanding.

## Setup (Windows, once)

```powershell
cd llm
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q tests        # 14 tests, CPU only, a few seconds
```

## Code map

| File | What it does |
|---|---|
| `tinygpt/config.py` | `GPTConfig` (vocabulary, context window, layers, heads, width, dropout) and size presets |
| `tinygpt/tokenizer.py` | `CharTokenizer` (one token per character) and `BPETokenizer` (byte-level Byte Pair Encoding, trained from our own text) |
| `tinygpt/model.py` | The Transformer: embeddings, causal self-attention, MLP, blocks, loss, `generate()` (temperature / top-k / top-p) |
| `tests/` | Shape, causal-mask, initial-loss, generation, checkpoint and "can it learn" tests |

## How the model works

```
token IDs (B, T)
  -> token embedding + position embedding         each token becomes a vector of size n_embd
  -> n_layer x [ x + Attention(LayerNorm(x))      tokens gather information from EARLIER tokens only
                 x + MLP(LayerNorm(x)) ]          each token is processed on its own
  -> LayerNorm -> output layer                    one score per vocabulary token, per position
  -> cross-entropy against the next token          training signal
```

- **Causal mask**: position *t* can only attend to positions 0..*t*. `tests/test_model.py`
  checks this directly: changing a later token never changes earlier predictions.
- **Initial loss** is about ln(vocab_size), i.e. an untrained model is equally unsure about
  every token. Training should push it down; validation loss tells us whether it generalises.
- **Weight tying**: the output layer shares weights with the token embedding.
- **Two attention paths** (`fused_attention`): PyTorch's fast fused kernel, or the
  step-by-step version (scores -> mask -> softmax -> weighted sum) for learning. A test
  checks they give identical results.

## Sizes on the RTX 3050 (6 GB), measured

| Preset | Parameters* | Context | Batch | Peak GPU memory | Throughput (bf16) |
|---|---|---|---|---|---|
| tiny | ~1M | 128 | - | - | for CPU tests |
| small | 12.2M | 256 | 32 | 1.35 GB | ~96k tokens/s |
| medium | 27.3M | 512 | 16 | 2.20 GB | ~47k tokens/s |

\* with a 4,096-token vocabulary, excluding position embeddings.

## Limitations to keep in mind

A model of this size learns spelling, common phrases and local grammar of its training
text. It does not acquire reliable knowledge or reasoning, and fluent output is not
evidence of correctness. It must never classify variants or produce clinical statements on
its own (roadmap section 9).
