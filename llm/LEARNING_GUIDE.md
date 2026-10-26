# LLM Learning Guide

A plain-language guide to every concept used while building **tinygpt**, our language
model built from scratch. Each section explains the idea, shows where it lives in our
code, and uses real numbers from our own runs.

This guide is updated after every completed step of the
[ADVAT LLM roadmap](../ADVAT_LLM_Project_Roadmap.docx.pdf). See the
[progress log](#progress-log) at the end for what changed and when.

**Contents**

1. [The big picture](#1-the-big-picture)
2. [Text to numbers: tokens and tokenizers](#2-text-to-numbers-tokens-and-tokenizers)
3. [Inside the model: the Transformer](#3-inside-the-model-the-transformer)
4. [How the model predicts: logits, probabilities, loss](#4-how-the-model-predicts-logits-probabilities-loss)
5. [How the model learns: training](#5-how-the-model-learns-training)
6. [Checking that it learns: validation and overfitting](#6-checking-that-it-learns-validation-and-overfitting)
7. [Saving progress: checkpoints](#7-saving-progress-checkpoints)
8. [Hardware: GPU, memory, precision](#8-hardware-gpu-memory-precision)
9. [Writing text: generation and sampling](#9-writing-text-generation-and-sampling)
10. [Data: where training text comes from](#10-data-where-training-text-comes-from)
11. [Limits: what a small model can and cannot do](#11-limits-what-a-small-model-can-and-cannot-do)
12. [Glossary](#12-glossary)
13. [Progress log](#progress-log)

---

## 1. The big picture

**What is a language model?**
A program that, given some text, predicts what comes next. That is all it does. Writing a
paragraph is just predicting the next piece, adding it, and predicting again.

```
"To be or not to b"  ──►  model  ──►  next is probably "e"
```

**What does "from scratch" mean?**
The model starts with **random numbers** inside it. It knows nothing - not English, not
spelling, not Shakespeare. Everything it ends up "knowing" comes from our training. We do
not download or reuse anyone else's trained model.

**The four stages we follow**

| Stage | What happens | Our file |
|---|---|---|
| 1. Tokenize | Turn text into numbers the model can process | `tinygpt/tokenizer.py` |
| 2. Build | Define the neural network (the Transformer) | `tinygpt/model.py` |
| 3. Train | Show it lots of text so it learns to predict the next token | `train.py` |
| 4. Generate | Ask it to continue a prompt | `generate.py` |

**Important distinction (from the roadmap):** *training a language model* and *building an
application that uses one* are separate jobs. A language model does not replace variant
calling, annotation, ACMG criteria or expert review in ADVAT.

---

## 2. Text to numbers: tokens and tokenizers

Neural networks only work with numbers, so text must first be converted.

**Token** - one unit of text the model sees. Depending on the tokenizer it can be a
character, part of a word, or a whole word.

**Token ID** - the number assigned to a token. `"a"` might be 39, `"b"` 40, and so on.

**Vocabulary** - the full list of tokens the tokenizer knows. **Vocabulary size** is how
many there are.

**Tokenizer** - the tool that converts text to token IDs (*encode*) and back (*decode*).

### Character tokenizer (what we used first)

One token per character. Our Shakespeare text has exactly **65 distinct characters**
(letters, punctuation, space, newline), so the vocabulary size is 65.

```
"ROMEO:"  ──encode──►  [30, 27, 25, 17, 27, 10]  ──decode──►  "ROMEO:"
```

- Simple and never meets an unknown word.
- But sequences are long (every letter is a step), so the model has to work harder to
  learn words.

### BPE tokenizer (Byte Pair Encoding - what real LLMs use)

Starts with single bytes and repeatedly **merges the most common pair** into a new token:

```
start:      t h e   t h e n   t h e r e
merge "th": th e    th e n    th e r e
merge "the":the     the n     the r e
```

After a few thousand merges, common words and word-pieces (`" the"`, `"ing"`,
`"variant"`) become single tokens. Text gets shorter, and the model can learn faster.
Because it falls back to raw bytes, BPE can encode *any* text, even symbols it never saw
(`"🧬"`, `"c.664del"`).

**Special token** - a reserved token with a fixed meaning, such as `<|endoftext|>`
marking where one document ends.

*Our code:* `CharTokenizer` and `BPETokenizer` in `tinygpt/tokenizer.py`; both written
from scratch.

---

## 3. Inside the model: the Transformer

Our model is a **GPT-style, decoder-only Transformer** - the same family of design used by
large modern LLMs, just much smaller.

```
token IDs
  └► token embedding + position embedding     each token becomes a list of numbers
  └► Transformer block  x 6                   tokens exchange information, then get processed
  └► final LayerNorm
  └► output layer                             a score for every possible next token
```

### Embeddings

An **embedding** turns a token ID into a **vector** - a list of numbers (384 numbers in
our `small` model). At the start these numbers are random; during training, tokens that
behave similarly end up with similar vectors.

**Position embedding** - a second vector added for the token's *position* (1st, 2nd,
3rd...). Without it, the model could not tell "dog bites man" from "man bites dog".

### Self-attention

The key idea of the Transformer. For each token, attention asks: *which earlier tokens
are relevant to me, and how much?* It then mixes in information from those tokens.

Example: predicting the character after `"ROMEO:\nI will go to th"`, attention lets the
model look back at `"th"` and `"go to"` to decide `"e"` is likely.

Mechanically, each token produces three vectors:
- **Query** - "what am I looking for?"
- **Key** - "what do I contain?"
- **Value** - "what information do I pass on?"

Each query is compared with every key (a similarity score), the scores are turned into
weights with softmax, and the values are mixed using those weights.

**Multi-head attention** - several attentions run side by side (6 "heads" in our model),
each free to focus on a different kind of relationship.

### Causal mask

A language model must predict the *next* token, so it may only look at tokens **before**
the current position - never the future. The **causal mask** blocks attention to later
positions.

```
            can look at →   pos 0  pos 1  pos 2  pos 3
position 0 (T)                ✔      ✘      ✘      ✘
position 1 (o)                ✔      ✔      ✘      ✘
position 2 (space)            ✔      ✔      ✔      ✘
position 3 (b)                ✔      ✔      ✔      ✔
```

Our test `test_causal_mask_hides_the_future` proves it: changing a later character never
changes the predictions for earlier positions.

### The other parts of a block

- **MLP / feed-forward layer** - after attention, each token is processed on its own by a
  small two-layer network (widen to 4x, apply GELU, shrink back).
- **Residual connection** - each step *adds* its result to the input (`x = x + step(x)`)
  instead of replacing it. This keeps information flowing through many layers and makes
  training stable.
- **LayerNorm** - rescales numbers to a consistent range before each step, so values don't
  grow or shrink out of control.
- **Dropout** - during training, randomly switches off a fraction of values (20% in our
  run). This stops the model relying too heavily on any single path, which reduces
  memorisation. It is turned off when generating.

### Parameters (weights)

The **parameters** are all the numbers inside the model that training adjusts. Our
`small` model has **10.7 million** of them (with the 65-character vocabulary). Large
commercial LLMs have hundreds of billions.

**Weight tying** - our output layer reuses the token-embedding numbers instead of having
its own, saving parameters.

**Random initialisation** - all parameters start as small random numbers (around ±0.02).
This is literally what "from scratch" means.

### Model size settings (`tinygpt/config.py`)

| Setting | Meaning | `small` preset |
|---|---|---|
| `vocab_size` | number of possible tokens | 65 (characters) |
| `block_size` | **context window**: how many previous tokens it can see | 256 |
| `n_layer` | number of Transformer blocks | 6 |
| `n_head` | attention heads per block | 6 |
| `n_embd` | vector size per token ("width") | 384 |
| `dropout` | fraction switched off during training | 0.2 |

---

## 4. How the model predicts: logits, probabilities, loss

**Forward pass** - running input through the model to get its predictions.

**Logits** - the raw scores the model outputs, one per vocabulary token. Higher means "more
likely next". They can be any number, positive or negative.

**Softmax** - converts logits into **probabilities** that are all between 0 and 1 and add
up to 100%.

```
logits:        ' ' 5.1    'e' 5.1    'm' 2.6    'r' 0.9   (approximate)
softmax  ──►   ' ' 47%    'e' 47%    'm' 3.9%   'r' 0.7%
```

(The probabilities are real output of our trained model for the prompt `"I will go to the"`.)

**Loss** - one number measuring how wrong the predictions were. Lower is better.

**Cross-entropy loss** - the standard loss for language models. It looks at the
probability the model gave to the *correct* next token:

| Probability given to the right answer | Loss |
|---|---|
| 100% | 0.00 (perfect) |
| 50% | 0.69 |
| 10% | 2.30 |
| 1.5% (random guess among 65) | 4.17 |

**Why our untrained model started at loss 4.30:** with random weights it is equally unsure
about all 65 characters, so it gives each about 1/65 = 1.5%. The theoretical loss for that
is ln(65) = **4.17**. Our measured 4.30 was right next to it - a good sign the model was
built correctly.

---

## 5. How the model learns: training

**Training** = repeatedly showing the model text, measuring how wrong its predictions are,
and nudging all parameters to be a little less wrong.

One **training step** (also called an **iteration**):

```
1. take a batch of text chunks from the training data
2. forward pass: predict the next token at every position
3. compute the loss
4. backpropagation: work out how each parameter contributed to the error
5. optimizer step: adjust every parameter slightly to reduce the loss
```

Our first run did **5,000** of these steps in 15.4 minutes.

**Batch** - several text chunks processed together (we used **64 chunks of 256
characters**). Larger batches give steadier learning but need more GPU memory.

**Epoch** - one full pass through all the training data. (We sample random chunks instead
of counting epochs; 5,000 steps x 64 x 256 ≈ 82 million characters seen, roughly 80 passes
over our 1 million training characters.)

**Gradient** - for each parameter, the direction and amount to change it to reduce the
loss.

**Backpropagation** - the algorithm that calculates all gradients efficiently, working
backwards from the loss through every layer. PyTorch does this automatically
(`loss.backward()`; called **autograd**).

**Optimizer** - the rule that turns gradients into parameter updates. We use **AdamW**,
the standard choice for Transformers; it adapts the step size for each parameter.

**Learning rate** - how big each update step is. Too big: training becomes unstable. Too
small: learning is very slow. We used a peak of **0.001**.

**Learning-rate schedule** - changing the learning rate during training:
- **Warm-up** - start tiny and increase over the first 100 steps, so early random updates
  don't wreck the model.
- **Cosine decay** - then slowly lower it to 0.0001 by the end, for fine adjustments.

**Weight decay** - gently pulls parameters towards zero, discouraging extreme values (a
form of regularisation). We used 0.1.

**Gradient clipping** - caps unusually large gradients (at 1.0) so one bad batch cannot
cause a huge, damaging update.

**Regularisation** - any technique that reduces memorisation (dropout, weight decay,
early stopping, more data).

---

## 6. Checking that it learns: validation and overfitting

**Training set** - the text the model learns from (the first 90%, 1,003,856 characters).

**Validation set** - text held back that the model **never trains on** (the next 5%,
55,769 characters). Measuring loss here tells us whether the model learned general
patterns or just memorised. We also use it to **choose**: which checkpoint to keep, which
settings work best.

**Test set** - the last 5% (55,769 characters), locked away and used **only once at the
end** for a final, honest score. Why not just use validation? Because every time we pick
the setting with the best validation loss, we slightly "tune to" the validation text, so
its score becomes a bit optimistic. The test set has never influenced any decision, so its
score is unbiased.

```
[ train (90%) .................................. | val (5%) | test (5%) ]
   learn from this                                  choose with   final score only
```

(Our first run used a 90/10 train/validation split; we added the test set in Phase 4.)

**Why we split by position, not randomly:** neighbouring lines of a play are similar. If
validation lines were scattered among training lines, the model would effectively have
seen them - called **data leakage** - and validation loss would look better than reality.

### Overfitting - seen in our own run

| Step | Training loss | Validation loss | What is happening |
|---|---|---|---|
| 0 | 4.30 | 4.30 | knows nothing |
| 1,000 | 1.26 | 1.53 | both falling: learning general patterns |
| **1,750** | **1.09** | **1.47** | **best validation loss** |
| 3,000 | 0.85 | 1.57 | training still falling, validation **rising** |
| 5,000 | 0.60 | 1.72 | memorising the training text |

**Overfitting** - the model fits the training data better and better while getting
*worse* on new data. It has started memorising instead of generalising.

**Generalisation** - performing well on data the model has not seen. This is what we
actually want.

**Early stopping** - keeping the weights from the point of best validation loss (step
1,750 here) instead of the last step. Our `best.pt` does this automatically.

**Why it overfitted:** 10.7 million parameters is a lot of memory for only 1 million
characters of text. Remedies: more data, a smaller model, more dropout, or stopping
earlier.

### Running fair experiments (what we did in Phase 4)

**Hyperparameter** - a setting *we* choose before training (dropout, model size, number of
steps, learning rate), as opposed to the parameters the model learns itself.

**Baseline** - the reference result every new idea is compared against. Our baseline is
the first run (run A).

**Controlled experiment / ablation** - change **one thing at a time**, keep everything
else identical (same data, same seed, same other settings). Then any difference in results
must come from that one change. If we changed dropout *and* model size together and the
result improved, we would not know which one helped.

**Model capacity** - how much a model *can* memorise, roughly its number of parameters. Too
much capacity for too little data leads to overfitting.

Our three experiments against the baseline, all aimed at reducing overfitting:

| Run | One change | Parameters | Validation loss | Time | Result |
|---|---|---|---|---|---|
| A (baseline) | - | 10.7M | 1.390 | 15.4 min | overfits strongly after step 1,750 |
| B | more dropout (0.2 → 0.3) | 10.7M | 1.382 | 15.2 min | slower learning, less memorising |
| **C** | **smaller model** (4 layers, width 256) | **3.2M** | **1.369** | **6.0 min** | **best, and fastest** |
| D | fewer steps (5,000 → 2,500) | 10.7M | 1.374 | 7.7 min | stops before memorising sets in |

How strongly each run overfitted, shown by the training/validation gap at the end:

```
A  train 0.60  val 1.72   ← big gap: memorised the training text
B  train 0.89  val 1.42
D  train 0.98  val 1.37
C  train 1.08  val 1.37   ← small gap: learned general patterns
```

**Lesson: bigger is not automatically better.** With only 1 MB of text, the 3.2M-parameter
model beat the 10.7M one - and trained 2.5x faster using half the GPU memory. The bigger
model spent its extra capacity memorising. More capacity only pays off with more data.

**Final test score** (run once, for the baseline and the winner only):
C **1.525** vs baseline 1.563. Both are higher than their validation scores because the
test passage (the end of the file) happens to be harder text - the important thing is that
the ranking stayed the same.

**Sampled vs exact evaluation** - during training, `train.py` *estimates* loss from 100
random batches (fast, slightly noisy). `evaluate.py` computes the *exact* loss over every
token of a split. Use exact numbers when comparing experiments.

**Noise / run-to-run variance** - training involves randomness (initial weights, which
batches are drawn, dropout). Two runs with different seeds give slightly different
results. Our differences (0.01-0.02) are small, so to be sure a winner is real you would
repeat each run with 2-3 seeds and compare the averages.

**Perplexity** - another way of expressing loss (perplexity = e^loss). Our best validation
loss 1.47 means perplexity ≈ 4.3: on average the model is as unsure as if choosing between
about 4 characters. It measures language-modelling quality only - **not** correctness or
clinical validity.

---

## 7. Saving progress: checkpoints

**Checkpoint** - a file saving the model's parameters (and training state) at a point in
time, so we can reload it later or continue training.

Our training saves two:
- `checkpoints/shakespeare_char/best.pt` - the lowest-validation-loss weights (used for
  generating)
- `checkpoints/shakespeare_char/last.pt` - the latest state, including the optimizer, so
  training can be **resumed** with `python train.py --resume` after an interruption

Also written: `log.csv` (loss at every evaluation) and `summary.json`.

**Inference** - using a trained model to produce outputs (as opposed to training it).

Checkpoints are **not** stored in git: they are large and can be regenerated by
re-running training.

---

## 8. Hardware: GPU, memory, precision

**GPU** - a graphics card. It performs thousands of calculations in parallel, which is
exactly what neural networks need. Ours: **NVIDIA RTX 3050 Laptop, 6 GB**.

**CUDA** - NVIDIA's software that lets PyTorch run calculations on the GPU. We installed
PyTorch's CUDA build; `torch.cuda.is_available()` confirmed the GPU is detected.

**VRAM** - the GPU's own memory (6 GB). The model, the batch and all intermediate values
must fit in it. Our run peaked at **1.74 GB**.

What uses VRAM: model size, batch size, context length, and the optimizer's extra
memory. If you run out, reduce batch size or context first.

**Throughput** - how fast training goes, measured in tokens per second (~96,000 for the
`small` model on our GPU).

**Precision** - how many bits store each number:
- **fp32** - 32-bit, standard, most memory
- **bf16** (bfloat16) - 16-bit, half the memory traffic, nearly the same accuracy. We
  train in bf16 using **autocast** (`torch.autocast`), which uses bf16 where safe and
  fp32 where precision matters. Our GPU supports bf16.

**Device** - where a calculation runs: `"cuda"` (GPU) or `"cpu"`. Our tests run on CPU so
they work on any machine; real training runs on the GPU.

---

## 9. Writing text: generation and sampling

**Generation (autoregressive)** - the model writes one token at a time, each time feeding
everything so far back in:

```
"ROMEO:"        → predicts "\n"  → "ROMEO:\n"
"ROMEO:\n"      → predicts "I"   → "ROMEO:\nI"
"ROMEO:\nI"     → predicts " "   → ...
```

**Prompt** - the starting text you give (`--prompt "JULIET:"`).

**Max tokens** (`--tokens`) - how many tokens to write. With our character model, 1 token
= 1 character, so `--tokens 300` stops after exactly 300 characters, even mid-word (that
is why output can end like `"QUEEN ELIZAB"`).

**Context window during generation** - the model only ever sees the last 256 tokens
(`block_size`); older text drops out of view.

### Sampling: how the next token is chosen

The model gives a probability for every token, then **picks one at random according to
those probabilities** - like a weighted dice roll. That is why every run gives different
text.

**Greedy decoding** (temperature 0) - always take the single most likely token. Same output
every time, but tends to get repetitive.

### Temperature

Adjusts the odds *before* the dice is rolled. Real numbers from our model, prompt
`"I will go to the"`:

| Temperature | `' '` | `'e'` | `'m'` | `'r'` | Effect |
|---|---|---|---|---|---|
| 0.5 (low) | 49.9% | 49.8% | 0.3% | 0.0% | unlikely options squeezed out → safe, repetitive |
| 1.0 (normal) | 46.9% | 46.9% | 3.9% | 0.7% | the model's own probabilities |
| 1.5 (high) | 40.6% | 40.6% | 7.7% | 2.6% | unlikely options boosted → creative, then nonsense |

What it looked like (same seed):
- **0.2:** *"The senators of the common of the world. / Servant: / The news is a thing..."* - real
  words, very repetitive
- **0.8:** *"O, after one of your lands to the friar?"* - balanced (our default)
- **1.5:** *"Mustice is ever boy, and murdep with a gial"* - invents words

Mathematically: logits are divided by the temperature before softmax. Dividing by a small
number exaggerates differences; dividing by a large number flattens them.

**For clinical use** (report drafts, explanations) use **low temperature or 0**: the same
input should give the same, most-likely wording. Randomness is unwanted in a report.

### Top-k and top-p

Extra filters applied before the dice roll:
- **Top-k** (`--top-k 40`) - only the 40 most likely tokens are allowed; everything else is
  removed. Prevents rare garbage choices.
- **Top-p / nucleus sampling** (`--top-p 0.9`) - keep the smallest set of tokens whose
  probabilities add up to 90%. Adapts automatically: few options when the model is sure,
  more when it is unsure.

### Seed

Computers generate "random" numbers from a starting number called the **seed**. The same
seed always produces the same sequence of dice rolls, so the **same text**.

```
--seed 42  → identical output every run
--seed 7   → different, but also repeatable
no seed    → different every run
```

42 is not special - any number works. Use a seed when **comparing settings** (change one
thing, keep luck fixed), **sharing a result**, or when output must be **reproducible**.
`train.py` always uses a seed (1337) so training runs are comparable.

---

## 10. Data: where training text comes from

**Corpus** - the collection of text used for training. Our first: **Tiny Shakespeare**,
1,115,394 characters of Shakespeare's plays.

**Licence / legally usable** - we may only train on text we have the right to use.
Shakespeare is **public domain**. For biomedical text later (roadmap phase 6), each source
must be checked: for example ClinVar and HPO have usable terms, while OMIM and
GeneReviews restrict reuse.

**Provenance** - recording where data came from. `prepare_data.py` writes
`data/shakespeare/meta.json` with the source URL, licence, download date, a SHA-256
**checksum** (a fingerprint proving the file has not changed) and split sizes.

**Deduplication** - removing repeated text, so the model doesn't over-learn duplicates and
duplicates don't leak between training and validation.

**A model only knows its training data.** Give our model a medical prompt and it still
answers in Shakespeare style. An ADVAT model would need large amounts of relevant,
licence-clean genomics text.

**Patient data** is never used for training at this stage (roadmap section 8).

---

## 11. Limits: what a small model can and cannot do

What our 10.7M-parameter model learned from 1 MB of text:
- spelling of real words
- line rhythm and play formatting (`NAME:` then dialogue)
- local grammar patterns

What it did **not** learn:
- meaning (*"A drunken very fatal tinger"*)
- stories or characters - Juliet talks about Bolingbroke, who is from a different play
- any facts

**Key lesson (roadmap section 9): fluent output is not proof of correctness.** A model can
sound confident and be completely wrong. In ADVAT, a language model must never classify a
variant or make clinical statements on its own; validated tools and qualified review
remain responsible.

**Why we can't just make it bigger on this laptop:** useful general-purpose LLMs need
billions of parameters and hundreds of billions of tokens of text - far beyond 6 GB of
VRAM. Scaling is not just changing one setting.

---

## 11b. Using someone else's model: open weights, quantisation, serving

Everything above was about building a model from scratch. The ADVAT tool takes the other
route: start from a model someone else already pretrained. The concepts carry over, with a
few new ones.

**Open-weight model** - a model whose trained parameters are published for anyone to
download and run. Examples: Qwen, Gemma, Llama, Phi, Mistral. "Open weights" is not the
same as "open source": the weights are given, the training data usually is not.

**Licence** matters for a commercial lab. Apache 2.0 and MIT allow commercial use freely.
Others (Meta's Llama community licence, Google's health-model terms) add conditions that
need reading before clinical use.

**Pretrained vs fine-tuned.** Pretraining is the expensive stage someone else already paid
for (trillions of tokens, large GPU clusters). **Fine-tuning** continues training on your
own examples to teach style and task-specific behaviour. Fine-tuning teaches *how to
answer*, not *new facts* - a crucial limit.

**Quantisation** - storing each parameter in fewer bits so the model fits in less memory.

| Precision | Qwen3.5 4B size | Fits our 6 GB GPU? |
|---|---|---|
| bf16 (16-bit) | ~8 GB | no |
| 8-bit | ~4.5 GB | tight |
| **4-bit** | **~3 GB** | yes |

Quality drops slightly with heavier quantisation; 4-bit is the usual sweet spot for
running locally.

**LoRA / QLoRA** - ways to fine-tune cheaply. LoRA freezes the original weights and trains
small add-on layers (about 1% of the parameters). QLoRA also keeps the frozen model in
4-bit. Together they make fine-tuning possible on a laptop GPU.

**Serving / inference server** - a program that loads the model and answers requests over
a local API. We use **Ollama** (`http://127.0.0.1:11434`). Others: llama.cpp, vLLM.

**Local vs API.** A local model runs on your own hardware, so no data leaves the network
and there is no per-question cost. An API model is stronger but sends data to a third
party. For patient data that difference is the whole decision.

**Structured output / constrained generation** - forcing the model to emit JSON matching a
fixed schema. Ollama takes the schema in its `format` field and restricts generation to
valid output, so the answer can be parsed reliably instead of hoping the model formats it
correctly.

**Thinking mode** - the model writes a private reasoning chain before its answer. In our
tests this was the single change that stopped the model deleting evidence it should have
kept, at roughly 5x the time.

**Context window in practice.** Our WES VCF has 480,290 variants (~20 million tokens);
Qwen3.5 4B holds 262,144. Raw data cannot go into a model - the pipeline filters to ~40
candidates first, which is a few thousand tokens. This is why the architecture is *tools
first, model last*.

## 11c. Benchmarking a model's clinical output

The same idea as a **test set**, applied to the finished answer rather than to language
quality. The lab analogy is **proficiency testing**: run samples whose answers you already
know, and check the assay gets them right.

Our benchmark (`AI-Analyser/bench/`): four synthetic cases built from textbook genetics
(so they contain no patient data and can live in git), each scored on five checks -
correct result flag, correct variants in the primary section, no ACMG criteria deleted,
stated classification consistent with its own criteria, and no fabricated claims.

**Harness** - the machinery that runs the test, kept separate from what is tested. Ours
calls `ai.interpret()`, so swapping Qwen for Gemma or Claude is one environment variable
and the scores stay comparable.

**What we learned running it (2026-10-26, qwen3.5:4b, 19/20):**

- The model reliably got the **decisions** right: correct flag and correct reported
  variants in all four cases.
- It was **confidently wrong about a textbook fact** in one case, calling Dravet syndrome
  autosomal recessive when the supplied data said dominant. The correct answer was in the
  prompt and the model overrode it. This is a knowledge limit of a 4-billion-parameter
  model, and no prompt fixed it.
- **Guardrails beat instructions.** Asking the model not to delete criteria did not work;
  blocking deletion in code did.

**Two mistakes I made in the benchmark itself, both instructive:**

1. The cases hard-coded ACMG point totals, and one was wrong. Fixed by computing them with
   the real scoring engine, so a case can never disagree with itself.
2. The "no fabricated claims" check matched plain keywords, so it failed the model for
   writing *"precludes a definitive diagnosis"* - the correct thing to say about a VUS.
   Fixed with a **negation guard**, exactly as `wes/phenotype.py` already does for clinical
   text. Real score: 19/20, not 18/20.

Lesson: **a benchmark needs the same testing as the product code.** An unchecked benchmark
produces confident, wrong verdicts - the same failure it is meant to detect.

## 12. Glossary

| Term | Meaning |
|---|---|
| AdamW | The optimizer we use to update parameters |
| Attention | Mechanism letting each token use information from relevant earlier tokens |
| Autocast | Automatic use of lower precision (bf16) where safe |
| Autograd | PyTorch's automatic calculation of gradients |
| Ablation / controlled experiment | Changing one setting at a time to see its effect |
| Backpropagation | Computing how each parameter contributed to the error |
| Baseline | Reference result that new ideas are compared against |
| Benchmark | Fixed cases with known answers, used to measure a model objectively |
| Constrained generation | Forcing output to match a schema (e.g. valid JSON) |
| Batch | Group of text chunks processed together in one step |
| bf16 | 16-bit number format that saves memory with little accuracy loss |
| BPE | Byte Pair Encoding: tokenizer that merges frequent pairs into sub-word tokens |
| Causal mask | Prevents a position from seeing future tokens |
| Checkpoint | Saved model (and training state) file |
| Context window | How many previous tokens the model can see (`block_size`) |
| Corpus | Collection of training text |
| Cross-entropy | Standard loss: penalises low probability on the correct next token |
| CUDA | NVIDIA software for running calculations on the GPU |
| Data leakage | Validation/test data accidentally influencing training |
| Dropout | Randomly disabling values during training to reduce memorisation |
| Early stopping | Keeping the weights from the best validation point |
| Embedding | Learned vector representing a token (or a position) |
| Epoch | One full pass through the training data |
| Forward pass | Running input through the model to get predictions |
| Generalisation | Performing well on unseen data |
| Gradient | Direction and amount to change a parameter to reduce loss |
| Gradient clipping | Capping very large gradients for stability |
| Greedy decoding | Always choosing the most likely next token (temperature 0) |
| Hyperparameter | A setting chosen before training (dropout, size, steps, learning rate) |
| Inference | Using a trained model to produce output |
| Iteration / step | One training update |
| LayerNorm | Rescales values to a stable range |
| Learning rate | Size of each parameter update |
| Logits | Raw scores for each possible next token |
| Loss | Number measuring prediction error (lower is better) |
| MLP | Small feed-forward network inside each block |
| Harness | The machinery that runs a test, independent of what is tested |
| LoRA / QLoRA | Cheap fine-tuning: train small add-on layers, optionally over a 4-bit model |
| Model capacity | How much a model can memorise; roughly its parameter count |
| Open-weight model | A model whose trained parameters are published for anyone to run |
| Noise (run-to-run variance) | Differences between runs caused only by randomness |
| Overfitting | Better on training data, worse on new data: memorising |
| Parameters / weights | The learned numbers inside the model |
| Perplexity | e^loss; a language-modelling quality measure, not a correctness measure |
| Position embedding | Vector telling the model where a token sits in the sequence |
| Prompt | Starting text given to the model |
| Provenance | Record of where data came from and how it was processed |
| Quantisation | Storing parameters in fewer bits so a model fits in less memory |
| Residual connection | Adding a layer's output to its input (`x + f(x)`) |
| Seed | Starting number that makes "random" choices repeatable |
| Softmax | Turns logits into probabilities that sum to 100% |
| Temperature | Makes sampling safer (low) or more random (high) |
| Token / token ID | Unit of text / the number representing it |
| Tokenizer | Converts text to token IDs and back |
| Top-k / top-p | Sampling filters limiting which tokens may be picked |
| Training / validation / test set | Data to learn from / to check generalisation / for the final score |
| Transformer | Neural-network design built from attention and feed-forward layers |
| Validation loss | Loss on held-out text; tells us whether the model generalises |
| Vocabulary | All tokens the tokenizer can produce |
| VRAM | The GPU's memory |
| Weight decay | Gently shrinking parameters to discourage extreme values |
| Weight tying | Output layer sharing weights with the token embedding |

---

## Progress log

| Date | Roadmap step | What we did | Concepts added |
|---|---|---|---|
| 2026-10-23 | Phase 1 - environment | PyTorch 2.11 + CUDA installed in `llm/.venv`; RTX 3050 detected with bf16 support | GPU, CUDA, VRAM, precision, device |
| 2026-10-23 | Phase 2 - architecture | Character and BPE tokenizers; GPT-style Transformer with random initialisation; 14 tests | Tokens, tokenizers, embeddings, attention, causal mask, blocks, parameters, logits, loss |
| 2026-10-23 | Phase 3 - training | Tiny Shakespeare prepared (65-char vocab, 90/10 split); `small` model trained 5,000 steps in 15.4 min; best val loss 1.47 at step 1,750, then overfitting | Training loop, batch, gradient, backprop, AdamW, learning-rate schedule, validation, overfitting, early stopping, checkpoints, perplexity, provenance |
| 2026-10-23 | Phase 4 - generation (started) | `generate.py`; temperature, top-k, top-p, seed demonstrated on the trained model | Autoregressive generation, sampling, temperature, top-k, top-p, seed, greedy decoding |
| 2026-10-26 | Direction change - open-weight model instead of from-scratch | Installed Ollama + qwen3.5:4b; added a backend adapter to AI-Analyser (local by default, Claude opt-in); built a 4-case clinical benchmark. Score 19/20; one persistent factual error on a textbook gene | Open weights, licences, quantisation, LoRA/QLoRA, local serving, constrained generation, thinking mode, clinical benchmarking, harness, guardrails vs instructions |
| 2026-10-23 | Phase 4 - evaluation & overfitting | Added locked test set (90/5/5 split) and exact `evaluate.py`; ran 3 controlled experiments. Smaller model (3.2M) won: val 1.369, test 1.525 vs baseline 1.390 / 1.563, in 6 min | Test set, hyperparameter, baseline, controlled experiment/ablation, model capacity, sampled vs exact evaluation, run-to-run noise |
