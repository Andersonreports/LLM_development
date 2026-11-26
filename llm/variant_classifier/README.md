# variant_classifier &middot; ordinal pathogenicity model

An **ordinal 5-tier** classifier for variants: `B < LB < VUS < LP < P`. Tabular machine
learning (gradient-boosted trees) — **separate** from the tinygpt language model and the
chat UI. The ordinal target means a big mistake (calling a Pathogenic variant Benign) is
penalised far more than a small one (LP vs VUS), which is what you want clinically.

> **Scope, honestly.** With public (ClinVar) labels this largely re-expresses what REVEL/CADD
> and an ACMG engine already encode, so `train.py` always prints a **single-score (REVEL)
> baseline** — if the model doesn't beat it, use REVEL. The version that beats off-the-shelf
> tools is one trained on **your own signed-off classifications**. Nothing here is validated
> for clinical use.

## Run the demo (no real data needed)

```bash
cd llm
.venv/bin/python -m variant_classifier.demo
```

It trains on synthetic variants twice and shows the single most important methodology point:

```
HONEST  gene-grouped split : QWK 0.541
LEAKY   random split       : QWK 0.605   <- inflated, do not trust
```

A **random** train/test split lets the model memorise gene-level signal that appears in both
folds, inflating the score. Splitting by **gene** (so test genes are unseen) gives the honest
estimate. The pipeline always uses the gene-grouped split; `--split random` exists only to
demonstrate the trap.

## Train on your own data

Your VCFs are raw, so the order is **annotate → label → join → train**:

1. **Annotate** your VCFs (VEP or dbNSFP) so each variant carries `gnomAD_AF`, `REVEL_score`,
   `CADD_phred`, `SpliceAI`, conservation and gene constraint. AI-Analyser's `wes/annotate.py`
   runs the VEP step.
2. **Build the table** (ClinVar labels + annotated features):
   ```python
   from variant_classifier.data import clinvar_labels, features_from_vcf, build_dataset
   labels   = clinvar_labels("clinvar.vcf.gz")            # -> variant, gene, tier
   features = features_from_vcf("annotated.vcf.gz")        # adjust field names if needed
   df = build_dataset(labels, features)
   df.to_csv("variants.csv", index=False)
   ```
   `features_from_vcf` takes a `field_map` — match it to your annotation's INFO keys
   (defaults are common VEP/dbNSFP names, not guaranteed).
3. **Train + evaluate** (out-of-fold, gene-grouped):
   ```bash
   .venv/bin/python -m variant_classifier.train --table variants.csv --importance --save model.pkl
   ```

## What to read

- `schema.py` — the 5 tiers, the ClinVar→tier map, the feature list.
- `ordinal.py` — Frank & Hall ordinal model (K−1 cumulative GBMs) on `HistGradientBoosting`
  (handles missing scores natively).
- `metrics.py` — quadratic-weighted kappa (headline), MAE-in-tiers, Spearman, and a count of
  **dangerous** B↔P flips.
- `train.py` / `data.py` / `synth.py` — cross-validation, loaders, synthetic generator.

## Pitfalls this code is built around

- **Gene-grouped splitting** (above) — the default.
- **No label leakage** — ClinVar status is used only as the *label*, never as a feature.
- **Imbalance** — benign dominates; we report AUPRC-style per-tier behaviour and the confusion
  matrix, not just accuracy.
- **Baseline comparison** — every run is measured against REVEL alone.

Dependencies: `scikit-learn` (in `llm/.venv`); `numpy`/`pandas`/`scipy` from the system Python.
