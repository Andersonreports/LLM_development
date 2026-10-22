"""Mitochondrial genome variant analysis.

Mitochondrial variants are not classified with the nuclear ACMG/AMP framework in
wes/acmg.py - that framework assumes diploid Mendelian inheritance (PVS1's
loss-of-function mechanism, PM2's population-frequency ceiling tuned for dominant/
recessive disease) which does not transfer to mtDNA. Standard practice for mtDNA
variant interpretation instead rests on three things this module implements:

  1. Match against well-established, confirmed-pathogenic mtDNA point mutations
     (the approach MITOMAP itself uses for its "confirmed pathogenic" list).
  2. Heteroplasmy level (what fraction of the many mitochondrial genome copies in
     the sample carry the variant) - most mtDNA disease is heteroplasmic, and the
     fraction present is itself part of the evidence.
  3. Whether the variant is a common, population-wide homoplasmic polymorphism
     (haplogroup marker) rather than a disease mutation.

Limitation, stated plainly: KNOWN_PATHOGENIC is a curated set of the most
well-established, textbook-confirmed pathogenic mtDNA point mutations - not the full
MITOMAP database. A variant absent from this list is classified VUS, not benign -
confirm against MITOMAP (mitomap.org) directly for anything not listed here before
treating it as excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

from .vcf import Genotype, VcfRecord

# The 37 canonical human mitochondrial genes (13 protein-coding, 2 rRNA, 22 tRNA).
MITO_GENES = {
    "MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3", "MT-ND3",
    "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB",
    "MT-RNR1", "MT-RNR2",
    "MT-TF", "MT-TV", "MT-TL1", "MT-TI", "MT-TQ", "MT-TM", "MT-TW", "MT-TA", "MT-TN",
    "MT-TC", "MT-TY", "MT-TS1", "MT-TD", "MT-TK", "MT-TG", "MT-TR", "MT-TH", "MT-TS2",
    "MT-TL2", "MT-TE", "MT-TT", "MT-TP",
}

# Curated, non-exhaustive: well-established, confirmed-pathogenic mtDNA point mutations
# (position is 1-based on the rCRS/GRCh38 chrM sequence). min_het is the heteroplasmy
# fraction (0-1) generally required for the phenotype; None means typically homoplasmic.
@dataclass(frozen=True)
class KnownMtMutation:
    gene: str
    disease: str
    min_het: float | None = None


KNOWN_PATHOGENIC: dict[tuple[int, str, str], KnownMtMutation] = {
    (3243, "A", "G"): KnownMtMutation("MT-TL1", "MELAS / MIDD / CPEO", 0.10),
    (8344, "A", "G"): KnownMtMutation("MT-TK", "MERRF", 0.10),
    (8993, "T", "G"): KnownMtMutation("MT-ATP6", "NARP / Leigh syndrome", 0.60),
    (8993, "T", "C"): KnownMtMutation("MT-ATP6", "NARP / Leigh syndrome", 0.60),
    (11778, "G", "A"): KnownMtMutation("MT-ND4", "Leber hereditary optic neuropathy (LHON)"),
    (3460, "G", "A"): KnownMtMutation("MT-ND1", "Leber hereditary optic neuropathy (LHON)"),
    (14484, "T", "C"): KnownMtMutation("MT-ND6", "Leber hereditary optic neuropathy (LHON)"),
    (1555, "A", "G"): KnownMtMutation("MT-RNR1", "Aminoglycoside-induced / nonsyndromic hearing loss"),
    (7445, "A", "G"): KnownMtMutation("MT-TS1", "Nonsyndromic hearing loss", 0.10),
    (3271, "T", "C"): KnownMtMutation("MT-TL1", "MELAS", 0.10),
    (3252, "A", "G"): KnownMtMutation("MT-TL1", "MELAS / mitochondrial myopathy", 0.10),
    (8356, "T", "C"): KnownMtMutation("MT-TK", "MERRF", 0.10),
    (10191, "T", "C"): KnownMtMutation("MT-ND3", "Leigh syndrome / Leigh-like disease", 0.10),
    (10158, "T", "C"): KnownMtMutation("MT-ND3", "Leigh syndrome", 0.10),
    (9176, "T", "C"): KnownMtMutation("MT-ATP6", "Leigh syndrome / NARP-like", 0.60),
    (9185, "T", "C"): KnownMtMutation("MT-ATP6", "Leigh syndrome / NARP-like", 0.60),
    (1494, "C", "T"): KnownMtMutation("MT-RNR1", "Aminoglycoside-induced / nonsyndromic hearing loss"),
    (4300, "A", "G"): KnownMtMutation("MT-TI", "Hypertrophic cardiomyopathy", 0.10),
    (3303, "C", "T"): KnownMtMutation("MT-TL1", "Hypertrophic cardiomyopathy", 0.10),
    (5728, "T", "C"): KnownMtMutation("MT-TN", "CPEO / mitochondrial myopathy", 0.10),
    (5703, "G", "A"): KnownMtMutation("MT-TN", "CPEO / mitochondrial myopathy", 0.10),
    (5447, "C", "T"): KnownMtMutation("MT-TW", "Encephalomyopathy", 0.10),
    (8363, "G", "A"): KnownMtMutation("MT-TK", "Hypertrophic cardiomyopathy / MERRF", 0.10),
    (1606, "G", "A"): KnownMtMutation("MT-TV", "Leigh syndrome / fatal infantile mitochondrial disease", 0.10),
    (1644, "G", "A"): KnownMtMutation("MT-TV", "Mitochondrial encephalomyopathy", 0.10),
    (15990, "C", "T"): KnownMtMutation("MT-TP", "Myopathy", 0.10),
    (7512, "T", "C"): KnownMtMutation("MT-TS1", "Myoclonic epilepsy / cardiomyopathy", 0.10),
    (3860, "G", "A"): KnownMtMutation("MT-ND1", "LHON-associated"),
    (4171, "C", "A"): KnownMtMutation("MT-ND1", "LHON-associated"),
    (14459, "G", "A"): KnownMtMutation("MT-ND6", "LHON / Leigh syndrome (dystonia)"),
    (13513, "G", "A"): KnownMtMutation("MT-ND5", "MELAS / Leigh syndrome overlap", 0.10),
    (13514, "A", "G"): KnownMtMutation("MT-ND5", "MELAS / Leigh syndrome overlap", 0.10),
}


def heteroplasmy_fraction(g: Genotype) -> float | None:
    """Fraction of reads supporting the alt allele - the practical heteroplasmy
    estimate available from a standard (non-mito-specific) variant caller's AD field."""
    if g.ad_ref is None or g.ad_alt is None:
        return None
    total = g.ad_ref + g.ad_alt
    return g.ad_alt / total if total else None


@dataclass
class MitoFinding:
    pos: int
    ref: str
    alt: str
    gene: str | None
    hgvsc: str | None
    heteroplasmy: float | None
    classification: str   # Pathogenic | Likely Pathogenic | Uncertain Significance | Likely Benign
    disease: str | None
    note: str


def classify(rec: VcfRecord, gt: Genotype, gene: str | None, hgvsc: str | None, consequence: str | None) -> MitoFinding:
    het = heteroplasmy_fraction(gt)
    known = KNOWN_PATHOGENIC.get((rec.pos, rec.ref, rec.alt))
    if known:
        if known.min_het is not None and het is not None and het < known.min_het:
            return MitoFinding(rec.pos, rec.ref, rec.alt, gene or known.gene, hgvsc, het,
                               "Uncertain Significance", known.disease,
                               f"Matches a confirmed pathogenic mutation ({known.disease}), but heteroplasmy "
                               f"({het:.0%}) is below the level generally associated with this phenotype "
                               f"(usually ≥{known.min_het:.0%}) - clinical significance at this level is uncertain")
        level = "homoplasmic/high heteroplasmy" if het is None or het >= 0.9 else f"{het:.0%} heteroplasmy"
        return MitoFinding(rec.pos, rec.ref, rec.alt, gene or known.gene, hgvsc, het, "Pathogenic", known.disease,
                           f"Confirmed pathogenic mtDNA mutation (m.{rec.pos}{rec.ref}>{rec.alt}), {level}")
    if consequence in ("synonymous_variant", "stop_retained_variant"):
        return MitoFinding(rec.pos, rec.ref, rec.alt, gene, hgvsc, het, "Likely Benign", None,
                           "Synonymous - not in the curated pathogenic mutation list")
    if het is not None and het >= 0.97 and gene not in (None,):
        # near-universally homoplasmic and not a known pathogenic mutation - most such
        # variants are ancestral/haplogroup-defining polymorphisms, not disease mutations,
        # but this is a soft heuristic, not a population-frequency database lookup
        return MitoFinding(rec.pos, rec.ref, rec.alt, gene, hgvsc, het, "Likely Benign", None,
                           "Homoplasmic and not in the curated pathogenic mutation list - most likely a "
                           "common/ancestral (haplogroup) variant, but this was not checked against a "
                           "population-frequency database")
    return MitoFinding(rec.pos, rec.ref, rec.alt, gene, hgvsc, het, "Uncertain Significance", None,
                       "Not found in the curated list of confirmed pathogenic mtDNA mutations - cross-check "
                       "against MITOMAP (mitomap.org) directly; this list is not exhaustive")


@dataclass
class MitoResult:
    data_available: bool
    findings: list[MitoFinding]
    flag: str  # POSITIVE | INCONCLUSIVE | NEGATIVE | NOT_PERFORMED


def analyse(mt_variants: list[dict]) -> MitoResult:
    """mt_variants: flat variant dicts as produced by variants.record_to_variants,
    already filtered to chrom == 'MT' and gt_ok for the proband."""
    if not mt_variants:
        return MitoResult(data_available=False, findings=[], flag="NOT_PERFORMED")
    findings = []
    for v in mt_variants:
        g = v.get("genotypes", {}).get("proband")
        if not g or not v.get("gt_ok", {}).get("proband"):
            continue
        ad = g.get("ad") or [None, None]
        gt = Genotype(gt=g.get("gt", "./."), ad_ref=ad[0], ad_alt=ad[1])
        f = classify_from_fields(v["pos"], v["ref"], v["alt"], gt, v.get("gene"), v.get("hgvsc"),
                                 (v.get("consequence") or [None])[0])
        if f.classification != "Likely Benign":
            findings.append(f)
    findings.sort(key=lambda f: {"Pathogenic": 0, "Uncertain Significance": 1}.get(f.classification, 2))
    if any(f.classification == "Pathogenic" for f in findings):
        flag = "POSITIVE"
    elif any(f.classification == "Uncertain Significance" for f in findings):
        flag = "INCONCLUSIVE"
    else:
        flag = "NEGATIVE"
    return MitoResult(data_available=True, findings=findings, flag=flag)


def classify_from_fields(pos: int, ref: str, alt: str, gt: Genotype, gene, hgvsc, consequence) -> MitoFinding:
    class _Rec:
        pass
    r = _Rec()
    r.pos, r.ref, r.alt = pos, ref, alt
    return classify(r, gt, gene, hgvsc, consequence)
