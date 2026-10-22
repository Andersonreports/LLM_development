"""Turns VEP-annotated VCF records into flat, JSON-serialisable variant dicts."""
from __future__ import annotations

import re
from typing import Iterable

from .config import settings
from .vcf import Genotype, VcfRecord

LOF_TERMS = {
    "stop_gained", "frameshift_variant", "splice_acceptor_variant", "splice_donor_variant",
    "start_lost", "transcript_ablation",
}
MISSENSE_TERMS = {"missense_variant"}
INFRAME_TERMS = {"inframe_insertion", "inframe_deletion", "protein_altering_variant", "stop_lost"}
SPLICE_REGION_TERMS = {
    "splice_region_variant", "splice_donor_5th_base_variant", "splice_donor_region_variant",
    "splice_polypyrimidine_tract_variant",
}
SYNONYMOUS_TERMS = {"synonymous_variant", "stop_retained_variant", "start_retained_variant"}

IMPACT_RANK = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0}

# GRCh38 pseudo-autosomal regions on X
PAR_X = [(10001, 2781479), (155701383, 156030895)]

CLINVAR_STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
    "criteria_provided,_single_submitter": 1,
}


def in_par(chrom: str, pos: int) -> bool:
    return chrom == "X" and any(a <= pos <= b for a, b in PAR_X)


def _float(v) -> float | None:
    if v in (None, "", "."):
        return None
    try:
        # VEP may join multiple values with '&'; take the max
        return max(float(x) for x in str(v).split("&") if x not in ("", "."))
    except ValueError:
        return None


def _clinvar(csq: dict) -> dict:
    sig_raw = (csq.get("ClinVar_CLNSIG") or "").replace("&", ",")
    rev = (csq.get("ClinVar_CLNREVSTAT") or "").replace("&", ",")
    stars = CLINVAR_STARS.get(rev.strip(","), 0)
    primary = re.split(r"[,&|]", sig_raw.lower())[0] if sig_raw else ""
    cat = {
        "pathogenic": "P",
        "likely_pathogenic": "LP",
        "pathogenic/likely_pathogenic": "P/LP",
        "uncertain_significance": "VUS",
        "likely_benign": "LB",
        "benign": "B",
        "benign/likely_benign": "B",
    }.get(primary)
    if "conflicting" in primary:
        cat = "conflicting"
    elif primary.startswith("pathogenic") and "low_penetrance" in sig_raw.lower():
        cat = "P"  # low-penetrance pathogenic; flagged via clinvar_sig text
    return {
        "clinvar_sig": sig_raw.replace("_", " ") or None,
        "clinvar_category": cat,
        "clinvar_stars": stars if cat else None,
        "clinvar_review": rev.replace("_", " ") or None,
        "clinvar_disease": (csq.get("ClinVar_CLNDN") or "").replace("_", " ").replace("&", "; ") or None,
        "clinvar_id": csq.get("ClinVar") or csq.get("ClinVar_ALLELEID") or None,
    }


# gnomAD genetic-ancestry groups used for filtering. Bottlenecked (ASJ, FIN, AMI),
# small (MID) and mixed (REMAINING) groups are excluded, as recommended by ClinGen
GRPMAX_POPS = ("AFR", "AMR", "EAS", "NFE", "SAS")


def _grpmax(csq: dict) -> tuple[float | None, str | None]:
    best, best_pop, seen = None, None, False
    for src in ("gnomADe", "gnomADg"):
        for pop in GRPMAX_POPS:
            v = _float(csq.get(f"{src}_{pop}_AF"))
            if v is None:
                continue
            seen = True
            if best is None or v > best:
                best, best_pop = v, f"{src}_{pop}"
    if not seen:
        return None, None
    return best, best_pop


def established_pathogenic(v: dict) -> bool:
    """ClinVar P/LP with >=2 stars (multiple submitters/expert panel) - e.g. common founder
    variants such as CFTR p.Phe508del or HFE p.Cys282Tyr that frequency filters must not drop."""
    return v.get("clinvar_category") in ("P", "LP", "P/LP") and (v.get("clinvar_stars") or 0) >= 2


def clinvar_pathogenic(v: dict, min_stars: int = 1) -> bool:
    return v.get("clinvar_category") in ("P", "LP", "P/LP") and (v.get("clinvar_stars") or 0) >= min_stars


def _spliceai_max(csq: dict) -> float | None:
    vals = [_float(csq.get(k)) for k in csq if k.startswith("SpliceAI_pred_DS_")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _rsid(csq: dict, rec: VcfRecord) -> str | None:
    ev = csq.get("Existing_variation") or ""
    for x in ev.split("&"):
        if x.startswith("rs"):
            return x
    return rec.id if rec.id.startswith("rs") else None


def _strip_tx(h: str | None) -> str | None:
    if not h:
        return None
    h = h.replace("%3D", "=")
    return h.split(":", 1)[1] if ":" in h else h


def _consequences(csq: dict) -> list[str]:
    return [c for c in (csq.get("Consequence") or "").split("&") if c]


def variant_type(cons: list[str]) -> str:
    cs = set(cons)
    if cs & LOF_TERMS:
        return "lof"
    if cs & MISSENSE_TERMS:
        return "missense"
    if cs & INFRAME_TERMS:
        return "inframe"
    if cs & SPLICE_REGION_TERMS:
        return "splice_region"
    if cs & SYNONYMOUS_TERMS:
        return "synonymous"
    return "other"


def genotype_ok(g: Genotype) -> bool:
    if not g.called:
        return False
    if g.dp is not None and g.dp < settings.min_dp:
        return False
    if g.gq is not None and g.gq < settings.min_gq:
        return False
    if g.is_het and g.allele_balance is not None and g.allele_balance < settings.min_ab_het:
        return False
    return True


def zygosity(g: Genotype, chrom: str, pos: int, sex: str | None) -> str:
    if not g.called:
        return "no_call"
    if g.is_ref:
        return "ref"
    hemi = sex == "male" and (chrom == "Y" or (chrom == "X" and not in_par(chrom, pos)))
    if hemi:
        return "hemizygous"
    if chrom == "MT":
        ab = g.allele_balance
        return "homoplasmic" if ab is None or ab >= 0.9 else "heteroplasmic"
    return "hom" if g.is_hom_alt else "het"


def record_to_variants(rec: VcfRecord, sexes: dict[str, str | None]) -> list[dict]:
    """One dict per (variant, gene). Variants without a gene-level consequence
    are dropped (intergenic)."""
    by_gene: dict[str, dict] = {}
    for c in rec.csq:
        sym = c.get("SYMBOL") or ""
        if not sym:
            continue
        if c.get("Allele") and c.get("Allele") not in (rec.alt, "-") and len(rec.ref) == len(rec.alt) == 1:
            continue
        prev = by_gene.get(sym)
        rank = (
            1 if c.get("MANE_SELECT") else 0,
            1 if c.get("BIOTYPE") == "protein_coding" else 0,
            1 if c.get("CANONICAL") == "YES" else 0,
            IMPACT_RANK.get(c.get("IMPACT", "MODIFIER"), 0),
        )
        if prev is None or rank > prev["_rank"]:
            by_gene[sym] = {"_rank": rank, "csq": c}

    # a variant inside one gene's coding sequence is not also a candidate for the
    # non-coding genes / UTR-intron overlaps around it
    if any(i["csq"].get("BIOTYPE") == "protein_coding" for i in by_gene.values()):
        by_gene = {k: i for k, i in by_gene.items() if i["csq"].get("BIOTYPE") == "protein_coding"}
    if any(i["csq"].get("IMPACT") in ("HIGH", "MODERATE", "LOW") for i in by_gene.values()):
        by_gene = {k: i for k, i in by_gene.items() if i["csq"].get("IMPACT") != "MODIFIER"}
    out = []
    for sym, item in by_gene.items():
        c = item["csq"]
        cons = _consequences(c)
        gnomade = _float(c.get("gnomADe_AF"))
        gnomadg = _float(c.get("gnomADg_AF"))
        max_af = _float(c.get("MAX_AF"))
        pop_af, pop_af_pop = _grpmax(c)
        if pop_af is None:  # no per-population fields: fall back to overall / VEP MAX_AF
            pop_af = max([x for x in (gnomade, gnomadg, max_af) if x is not None], default=None)
        sas = max([x for x in (_float(c.get("gnomADe_SAS_AF")), _float(c.get("gnomADg_SAS_AF"))) if x is not None],
                  default=None)
        v = {
            "key": f"{rec.key}:{sym}",
            "variant_id": rec.key,
            "chrom": rec.chrom, "pos": rec.pos, "ref": rec.ref, "alt": rec.alt,
            "qual": rec.qual, "filter": rec.filter,
            "gene": sym,
            "gene_id": c.get("Gene"),
            "transcript": c.get("MANE_SELECT") or c.get("Feature"),
            "ensembl_transcript": c.get("Feature"),
            "mane": bool(c.get("MANE_SELECT")),
            "hgvsc": _strip_tx(c.get("HGVSc")),
            "hgvsp": _strip_tx(c.get("HGVSp")),
            "consequence": cons,
            "consequence_str": ", ".join(x.replace("_", " ") for x in cons),
            "impact": c.get("IMPACT"),
            "vtype": variant_type(cons),
            "exon": c.get("EXON") or None,
            "intron": c.get("INTRON") or None,
            "biotype": c.get("BIOTYPE"),
            "variant_class": c.get("VARIANT_CLASS"),
            "rsid": _rsid(c, rec),
            "gnomad_exome_af": gnomade,
            "gnomad_genome_af": gnomadg,
            "max_af": max_af,
            "max_af_pops": c.get("MAX_AF_POPS") or None,
            "pop_af": pop_af,
            "pop_af_pop": pop_af_pop,
            "sas_af": sas,
            "sift": c.get("SIFT") or None,
            "polyphen": c.get("PolyPhen") or None,
            "am_score": _float(c.get("am_pathogenicity")),
            "am_class": c.get("am_class") or None,
            "revel": _float(c.get("REVEL")),
            "spliceai": _spliceai_max(c),
            "cadd": _float(c.get("CADD_PHRED")),
            **_clinvar(c),
            "genotypes": {},
            "zygosity": {},
            "gt_ok": {},
        }
        for role, g in rec.genotypes.items():
            v["genotypes"][role] = g.to_dict()
            v["zygosity"][role] = zygosity(g, rec.chrom, rec.pos, sexes.get(role))
            v["gt_ok"][role] = genotype_ok(g)
        out.append(v)
    return out


def exon_position(v: dict) -> tuple[int, int] | None:
    m = re.match(r"(\d+)(?:-\d+)?/(\d+)", v.get("exon") or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def worth_keeping(v: dict) -> bool:
    """Cheap pre-filter applied while streaming: drops common and non-coding
    variants that no downstream analysis (candidates, SF, carrier) can use."""
    if established_pathogenic(v):
        return True
    if (v.get("pop_af") or 0) > settings.af_known_pathogenic:
        return False
    if v.get("clinvar_category") in ("P", "LP", "P/LP", "conflicting", "VUS"):
        return True
    if v["vtype"] in ("lof", "missense", "inframe", "splice_region"):
        return True
    if (v.get("spliceai") or 0) >= 0.1:
        return True
    return v["chrom"] == "MT" and v["vtype"] != "synonymous"


def iter_variants(records: Iterable[VcfRecord], sexes: dict[str, str | None]):
    for rec in records:
        if settings.require_pass and not rec.passed:
            continue
        yield from record_to_variants(rec, sexes)
