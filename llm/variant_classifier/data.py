"""Loaders. The model consumes one tidy table: columns `gene`, `tier` (0..4) and the
schema FEATURES. You get that table either from synth.make() for the demo, or by joining
real ClinVar labels to real annotated-VCF features with the helpers below.

The VCF helpers parse text (plain or .gz) so there is no pysam/cyvcf2 dependency. INFO
field names differ between annotation setups, so features_from_vcf takes a field map you
adjust to your VEP/dbNSFP output - the defaults are common names, not guaranteed."""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import (CLINVAR_REVSTAT_OK, CLINVAR_TO_TIER, FEATURES, TIER_INDEX,
                     consequence_severity)


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def _info(field: str) -> dict:
    out = {}
    for kv in field.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
        else:
            out[kv] = True
    return out


def _records(vcf: Path) -> Iterator[tuple[str, str, str, str, dict]]:
    with _open(vcf) as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 8:
                continue
            yield c[0].replace("chr", ""), c[1], c[3], c[4], _info(c[7])


def _key(chrom, pos, ref, alt) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


# --------------------------------------------------------------------------- ClinVar labels
def clinvar_labels(vcf: str | Path, *, require_review: bool = True) -> pd.DataFrame:
    """ClinVar VCF -> DataFrame[variant, gene, tier]. Keeps only clear B/LB/VUS/LP/P with an
    acceptable review status; drops 'conflicting' and no-assertion records."""
    rows = []
    for chrom, pos, ref, alt, info in _records(Path(vcf)):
        sig = (info.get("CLNSIG") or "").lower().replace("|", "/").strip("_")
        tier = CLINVAR_TO_TIER.get(sig)
        if tier is None:
            continue
        if require_review and (info.get("CLNREVSTAT", "") not in CLINVAR_REVSTAT_OK):
            continue
        gene = (info.get("GENEINFO", "") or "").split(":")[0] or None
        rows.append({"variant": _key(chrom, pos, ref, alt), "gene": gene,
                     "tier": TIER_INDEX[tier]})
    df = pd.DataFrame(rows).drop_duplicates("variant")
    return df


# --------------------------------------------------------------------------- VCF features
DEFAULT_FIELD_MAP = {
    "gnomad_af": "gnomAD_AF", "revel": "REVEL_score", "cadd_phred": "CADD_phred",
    "spliceai": "SpliceAI_pred_DS_AG", "gerp": "GERP++_RS", "phylop": "phyloP100way_vertebrate",
    "loeuf": "LOEUF", "missense_z": "mis_z", "consequence": "Consequence", "gene": "SYMBOL",
}


def _num(v):
    try:
        return float(str(v).split(",")[0].split("&")[0])
    except (TypeError, ValueError):
        return float("nan")


def features_from_vcf(vcf: str | Path, field_map: dict | None = None) -> pd.DataFrame:
    """Annotated VCF -> DataFrame[variant, gene, FEATURES]. Pulls plain INFO keys; if your
    annotation packs everything into a VEP 'CSQ' string, split that upstream first."""
    fm = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    rows = []
    for chrom, pos, ref, alt, info in _records(Path(vcf)):
        row = {"variant": _key(chrom, pos, ref, alt),
               "gene": info.get(fm["gene"]) or None,
               "consequence_severity": consequence_severity(info.get(fm["consequence"]))}
        for feat in FEATURES:
            if feat == "consequence_severity":
                continue
            row[feat] = _num(info.get(fm.get(feat, feat)))
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("variant")


def build_dataset(label_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Join labels to features on the variant key; prefer the label's gene, fall back to the
    feature gene. Returns the tidy table train.py expects."""
    df = label_df.merge(feature_df, on="variant", how="inner", suffixes=("", "_feat"))
    df["gene"] = df["gene"].fillna(df.get("gene_feat"))
    keep = ["gene", "tier", *FEATURES]
    for col in keep:
        if col not in df:
            df[col] = float("nan")
    return df[keep]


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a prepared table (.csv or .parquet) with columns gene, tier, FEATURES."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = {"gene", "tier", *FEATURES} - set(df.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    return df
