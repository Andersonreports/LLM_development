"""Sample-level QC computed from the (alt-site) VCF.

Metrics: variant counts, Ti/Tv, het/hom ratio, genetic sex from chrX
heterozygosity, runs of homozygosity (consanguinity indicator), Mendelian
error rate for trios and a shared-rare-variant relatedness indicator for couples.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .vcf import VcfRecord
from .variants import genotype_ok, in_par

TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


@dataclass
class _SampleAcc:
    total: int = 0
    passed: int = 0
    snv: int = 0
    indel: int = 0
    ti: int = 0
    tv: int = 0
    het: int = 0
    hom: int = 0
    x_het: int = 0
    x_hom: int = 0
    y_calls: int = 0
    low_quality: int = 0
    # autosomal genotype track for ROH: chrom -> list[(pos, is_het)]
    track: dict = field(default_factory=lambda: defaultdict(list))


def _transmissible(g) -> set[int]:
    """Number of ALT alleles a parent can pass on."""
    if g.is_hom_alt:
        return {1}
    if g.is_het:
        return {0, 1}
    return {0}


def _roh(track: dict, min_len: int = 3_000_000, min_sites: int = 15) -> list[dict]:
    """Stretches of consecutive homozygous-alt calls uninterrupted by any het call."""
    segs = []
    for chrom, sites in track.items():
        sites.sort()
        start = end = None
        n = 0
        for pos, is_het in sites:
            if is_het:
                if start is not None and end - start >= min_len and n >= min_sites:
                    segs.append({"chrom": chrom, "start": start, "end": end, "sites": n})
                start, end, n = None, None, 0
            else:
                if start is None:
                    start = pos
                end = pos
                n += 1
        if start is not None and end - start >= min_len and n >= min_sites:
            segs.append({"chrom": chrom, "start": start, "end": end, "sites": n})
    return segs


def compute_qc(records: list[VcfRecord], roles: list[str], declared_sex: dict[str, str | None],
               case_type: str) -> dict:
    acc = {r: _SampleAcc() for r in roles}
    mendel_checked = mendel_errors = 0
    couple_union = couple_shared = 0

    for rec in records:
        for role in roles:
            g = rec.genotypes.get(role)
            if g is None or not g.has_alt:
                continue
            a = acc[role]
            a.total += 1
            if not rec.passed:
                continue
            a.passed += 1
            if not genotype_ok(g):
                a.low_quality += 1
                continue
            if rec.is_snv:
                a.snv += 1
                if (rec.ref.upper(), rec.alt.upper()) in TRANSITIONS:
                    a.ti += 1
                else:
                    a.tv += 1
            else:
                a.indel += 1
            if rec.chrom == "X" and not in_par(rec.chrom, rec.pos):
                if g.is_het:
                    a.x_het += 1
                elif g.is_hom_alt:
                    a.x_hom += 1
            elif rec.chrom == "Y":
                a.y_calls += 1
            elif rec.chrom not in ("MT",):
                if g.is_het:
                    a.het += 1
                elif g.is_hom_alt:
                    a.hom += 1
                if rec.is_snv and (rec.qual or 0) >= 30:
                    a.track[rec.chrom].append((rec.pos, g.is_het))

        if case_type == "trio" and rec.passed and rec.chrom not in ("X", "Y", "MT"):
            p, f, m = (rec.genotypes.get(r) for r in ("proband", "father", "mother"))
            if p and f and m and all(x.called and genotype_ok(x) for x in (p, f, m)) and p.has_alt:
                mendel_checked += 1
                if not any(x + y == p.alt_count for x in _transmissible(f) for y in _transmissible(m)):
                    mendel_errors += 1

        if case_type == "couple" and rec.passed:
            a1, a2 = rec.genotypes.get("partner1"), rec.genotypes.get("partner2")
            if a1 and a2 and rec.chrom not in ("X", "Y", "MT"):
                h1, h2 = a1.has_alt and genotype_ok(a1), a2.has_alt and genotype_ok(a2)
                if h1 or h2:
                    couple_union += 1
                    if h1 and h2:
                        couple_shared += 1

    out: dict = {"samples": {}, "warnings": []}
    for role in roles:
        a = acc[role]
        x_ratio = a.x_het / (a.x_het + a.x_hom) if (a.x_het + a.x_hom) >= 20 else None
        inferred = None
        if x_ratio is not None:
            inferred = "male" if x_ratio < 0.15 else ("female" if x_ratio > 0.3 else "ambiguous")
        roh = _roh(a.track)
        roh_mb = sum(s["end"] - s["start"] for s in roh) / 1e6
        s = {
            "variants_total": a.total,
            "variants_pass": a.passed,
            "low_quality_genotypes": a.low_quality,
            "snv": a.snv, "indel": a.indel,
            "ti_tv": round(a.ti / a.tv, 2) if a.tv else None,
            "het_hom_ratio": round(a.het / a.hom, 2) if a.hom else None,
            "x_het_ratio": round(x_ratio, 3) if x_ratio is not None else None,
            "y_calls": a.y_calls,
            "inferred_sex": inferred,
            "declared_sex": declared_sex.get(role),
            "roh_total_mb": round(roh_mb, 1),
            "roh_segments": sorted(roh, key=lambda r: r["start"] - r["end"])[:30],
        }
        dec = declared_sex.get(role)
        if inferred in ("male", "female") and dec in ("male", "female") and inferred != dec:
            out["warnings"].append(f"{role}: declared sex '{dec}' but genetic sex appears '{inferred}' - check sample identity")
        if s["ti_tv"] is not None and a.snv >= 1000 and not (2.0 <= s["ti_tv"] <= 3.5):
            out["warnings"].append(f"{role}: Ti/Tv {s['ti_tv']} outside expected exome range (2.0-3.5)")
        if roh_mb >= 70:
            out["warnings"].append(
                f"{role}: {roh_mb:.0f} Mb in runs of homozygosity - suggests parental consanguinity; prioritise recessive models")
        out["samples"][role] = s

    if case_type == "trio":
        rate = mendel_errors / mendel_checked if mendel_checked else None
        out["mendelian"] = {"checked": mendel_checked, "errors": mendel_errors,
                            "error_rate": round(rate, 4) if rate is not None else None}
        if rate is not None and rate > 0.05:
            out["warnings"].append(
                f"Mendelian error rate {rate:.1%} - possible sample swap or non-biological parentage; de novo calls unreliable")
        out["parentage_confirmed"] = rate is not None and rate <= 0.05
    if case_type == "couple":
        frac = couple_shared / couple_union if couple_union else None
        out["couple_sharing"] = {"union": couple_union, "shared": couple_shared,
                                 "shared_fraction": round(frac, 3) if frac is not None else None}
    return out
