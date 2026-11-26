"""A synthetic variant table so the whole pipeline runs today, before real annotation is
wired in. It is built to behave like the real problem in the ways that matter:

  * features are noisy and partly missing, and only jointly predict the label;
  * there is strong GENE-LEVEL signal (some genes are disease genes) - so a random
    train/test split leaks and inflates scores, while a gene-grouped split does not.
    train.py --split random vs gene demonstrates exactly this.

It is NOT real biology and must never be used for anything clinical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURES


def make(n: int = 8000, n_genes: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Each gene carries an identity effect on the LABEL that the per-variant scores capture
    # only weakly. The gene-level features (loeuf, missense_z) are near-unique per gene, so a
    # model can recover that effect ONLY by recognising a gene it has seen before - which
    # happens under a random split (gene leaks into both folds) but not a gene-grouped one.
    gene_offset = rng.normal(0, 1.3, n_genes)                       # label effect, hard to generalise
    gene_loeuf = np.clip(rng.uniform(0.05, 1.8, n_genes), 0.03, 2.0)  # near-unique gene fingerprint
    gene_mis_z = rng.normal(0, 1.5, n_genes)

    gene = rng.integers(0, n_genes, n)
    variant_effect = rng.normal(0, 1.0, n)
    # the per-variant scores (revel, cadd, ...) see the variant effect and only a little of
    # the gene offset, so they generalise to unseen genes
    z = variant_effect + 0.3 * gene_offset[gene]

    def noisy(signal, scale, lo=None, hi=None):
        v = signal + rng.normal(0, scale, n)
        if lo is not None:
            v = np.clip(v, lo, hi)
        return v

    # feature generation (directions match schema: higher score -> more pathogenic,
    # except gnomad_af and loeuf where LOWER is more pathogenic)
    revel = 1 / (1 + np.exp(-(1.3 * z + rng.normal(0, 0.8, n))))           # 0..1
    cadd = np.clip(15 + 6 * z + rng.normal(0, 5, n), 0, 50)                # phred
    spliceai = np.clip(1 / (1 + np.exp(-(0.9 * z - 1.5))) + rng.normal(0, 0.1, n), 0, 1)
    gerp = noisy(2.5 * np.tanh(z), 1.5, -12, 6)
    phylop = noisy(1.8 * np.tanh(z), 1.2, -20, 10)
    cons = np.clip(1 / (1 + np.exp(-(1.1 * z + rng.normal(0, 1.0, n)))), 0, 1)  # severity 0..1
    # pathogenic variants are rarer in the population
    log_af = np.clip(-2.0 - 1.2 * z + rng.normal(0, 1.0, n), -7, -0.3)
    gnomad_af = 10.0 ** log_af

    df = pd.DataFrame({
        "gene": gene,
        "gnomad_af": gnomad_af,
        "revel": revel,
        "cadd_phred": cadd,
        "spliceai": spliceai,
        "gerp": gerp,
        "phylop": phylop,
        "consequence_severity": cons,
        "loeuf": gene_loeuf[gene] + rng.normal(0, 0.05, n),
        "missense_z": gene_mis_z[gene] + rng.normal(0, 0.3, n),
    })

    # the LABEL carries the full gene offset (1.2x), most of which is NOT in the scores -
    # so recovering it requires recognising the gene (via its loeuf/missense_z fingerprint)
    lin = variant_effect + 1.2 * gene_offset[gene] + rng.normal(0, 0.6, n)
    cuts = np.quantile(lin, [0.55, 0.78, 0.93, 0.985])   # B, LB, VUS, LP, P proportions
    df["tier"] = np.searchsorted(cuts, lin).astype(int)

    # realistic missingness: some scores are simply absent for some variants
    for col, p in [("revel", 0.25), ("spliceai", 0.30), ("cadd_phred", 0.05),
                   ("gerp", 0.15), ("phylop", 0.15)]:
        df.loc[rng.random(n) < p, col] = np.nan

    return df[["gene", "tier", *FEATURES]]
