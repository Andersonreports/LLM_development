"""variant_classifier: an ordinal 5-tier variant pathogenicity model (B < LB < VUS < LP < P).

This is tabular machine learning, deliberately separate from the from-scratch tinygpt
language model and the chat UI. The target is the ACMG ladder treated as an *ordered*
scale, so the model is penalised more for large mistakes (calling a Pathogenic variant
Benign) than small ones (LP vs VUS).

Pipeline:
  schema.py   the 5 tiers, the ClinVar -> tier mapping, and the feature list
  data.py     load a labelled feature table (synthetic demo, or real ClinVar + VCF)
  ordinal.py  a Frank & Hall ordinal classifier built on gradient-boosted trees
  metrics.py  ordinal metrics (quadratic-weighted kappa, MAE-in-tiers, dangerous errors)
  train.py    gene-grouped cross-validation, baseline comparison, final model
  demo.py     runs the whole thing end to end on synthetic data

Honest scope: with public (ClinVar) labels this mostly re-expresses what REVEL/CADD and an
ACMG engine already encode - train.py always reports a single-score baseline so you can see
whether the model earns its place. The version that beats off-the-shelf tools is one trained
on your own signed-off classifications.
"""
