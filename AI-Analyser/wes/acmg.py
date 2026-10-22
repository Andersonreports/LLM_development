"""Automated ACMG/AMP 2015 criteria with ClinGen SVI refinements.

Only criteria computable from annotation are applied here. Phenotype-specific
(PP4), functional (PS3/BS3), segregation (PP1/BS4) and protein-position
(PS1/PM5/PM1) evidence are left to the AI interpreter and the reviewing
scientist. Classification uses the Bayesian points system (Tavtigian 2020).
"""
from __future__ import annotations

import re

from .config import settings
from .resources import GeneInfo
from .variants import clinvar_pathogenic, established_pathogenic, exon_position

# an offset is only captured when it directly follows the exon-boundary position number,
# so a 5' UTR position marker such as "c.-29-11C>T" is not mistaken for an intron offset
_INTRON_OFFSET_RE = re.compile(r"(?<=\d)[+-]\d+")


def _min_intron_offset(hgvsc: str | None) -> int | None:
    """Smallest distance (bp) from an exon boundary implied by HGVSc intron notation,
    e.g. "c.147+77_147+121del" -> 77, "c.1822-2A>G" -> 2. None if not intron-adjacent."""
    if not hgvsc:
        return None
    offsets = [abs(int(m)) for m in _INTRON_OFFSET_RE.findall(hgvsc)]
    return min(offsets) if offsets else None

# Genes with a near-identical paralog or segmental duplication, where short-read WES
# routinely misaligns reads between the two copies and produces spurious SNV/indel calls.
# Real disease risk at several of these loci (C4A/C4B, STRC) is driven by COPY NUMBER,
# which this pipeline does not call at all - a single SNV/indel here is not trustworthy
# without orthogonal confirmation (long-read sequencing, MLPA, ddPCR, Sanger with
# paralog-specific primers). Several of these are already named in the report's standard
# disclaimer (SMN1/2, GBA, PMS2, CYP21A2) for the same reason; this list is broader and
# used to automatically soften PVS1 rather than rely on a human recognising the gene.
PARALOG_PRONE_GENES = {
    "C4A", "C4B",            # RCCX module, chr6 MHC class III - complement C4, SLE risk
    "SMN1", "SMN2",          # spinal muscular atrophy
    "GBA", "GBAP1",          # Gaucher disease
    "PMS2", "PMS2CL",        # Lynch syndrome
    "STRC", "STRCP1",        # hearing loss
    "CYP21A2", "CYP21A1P",   # congenital adrenal hyperplasia
    "NCF1", "NCF1B", "NCF1C",  # chronic granulomatous disease
    "HBA1", "HBA2",          # alpha thalassaemia
}

POINTS = {"VeryStrong": 8, "Strong": 4, "Moderate": 2, "Supporting": 1}
CLASS_ORDER = ["Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"]
SHORT = {"Pathogenic": "P", "Likely pathogenic": "LP", "Uncertain significance": "VUS",
         "Likely benign": "LB", "Benign": "B"}


def _crit(code: str, strength: str, reason: str, source: str = "auto") -> dict:
    benign = code.startswith("B")
    pts = POINTS.get(strength, 0) if strength != "StandAlone" else 0
    return {"code": code, "strength": strength, "points": -pts if benign else pts,
            "reason": reason, "source": source}


def classify_points(criteria: list[dict]) -> tuple[int, str]:
    if any(c["code"] == "BA1" for c in criteria):
        return -99, "Benign"
    pts = sum(c["points"] for c in criteria)
    if pts >= 10:
        cls = "Pathogenic"
    elif pts >= 6:
        cls = "Likely pathogenic"
    elif pts >= 0:
        cls = "Uncertain significance"
    elif pts >= -6:
        cls = "Likely benign"
    else:
        cls = "Benign"
    return pts, cls


def _pvs1(v: dict, g: GeneInfo) -> dict | None:
    cons = set(v["consequence"])
    if v["vtype"] != "lof":
        return None
    if v.get("gene") in PARALOG_PRONE_GENES:
        # not withheld entirely - the variant may be real - but short-read WES cannot
        # reliably tell which paralog a read actually came from, so this is never
        # treated as a confirmed null variant without orthogonal confirmation
        return _crit("PVS1", "Supporting",
                    f"{v['gene']} has a near-identical paralog/segmental duplication; short-read "
                    "sequencing cannot reliably distinguish between them, so this call is not treated "
                    "as confirmed LoF. Confirm with an orthogonal method (long-read sequencing, MLPA, "
                    "ddPCR, or paralog-specific Sanger) before relying on it")
    modes = g.inheritance
    lof_mechanism = g.lof_intolerant or ("AR" in modes) or ("XLR" in modes) or (g.hi_score in ("2", "3"))
    if "start_lost" in cons:
        return _crit("PVS1", "Moderate", "Start-loss; alternative start codon not assessed")
    if not lof_mechanism:
        if modes:
            return _crit("PVS1", "Supporting",
                         "Null variant but loss-of-function is not an established disease mechanism for this gene")
        return None
    ep = exon_position(v)
    nmd_escape = False
    if ep and {"stop_gained", "frameshift_variant"} & cons:
        exon, total = ep
        nmd_escape = exon == total  # last exon -> escapes NMD
    if nmd_escape:
        return _crit("PVS1", "Strong", "Truncating variant in the last exon (predicted to escape NMD)")
    if {"splice_acceptor_variant", "splice_donor_variant"} & cons:
        offset = _min_intron_offset(v.get("hgvsc"))
        if offset is None or offset <= 2:
            return _crit("PVS1", "VeryStrong", "Canonical +/-1,2 splice site variant; LoF is a known disease mechanism")
        # VEP's consequence term says "canonical", but the HGVSc notation places the
        # variant well past the +/-1,2 dinucleotide (seen with some multi-base indels in
        # repetitive intronic sequence, where VCF- and HGVS-normalised coordinates
        # disagree) - do not assume confirmed canonical splice disruption from the
        # consequence term alone
        sai = v.get("spliceai")
        if sai is not None and sai >= 0.5:
            return _crit("PVS1", "Strong",
                        f"VEP labelled this a canonical splice-site variant, but HGVSc ({v.get('hgvsc')}) places it "
                        f"{offset} bp from the exon boundary, not the +/-1,2 dinucleotide. SpliceAI ({sai:.2f}) "
                        "supports a splicing effect; confirm with RNA/transcript analysis before treating as a "
                        "confirmed null variant")
        return _crit("PVS1", "Supporting",
                    f"VEP labelled this a canonical splice-site variant, but HGVSc ({v.get('hgvsc')}) places it "
                    f"{offset} bp from the exon boundary, not the +/-1,2 dinucleotide - possible annotation "
                    "artifact for this indel. Treated as an uncertain splice-region effect pending manual review "
                    "of the raw alignment")
    return _crit("PVS1", "VeryStrong", "Predicted null variant (NMD expected); LoF is a known disease mechanism")


def evaluate(v: dict, g: GeneInfo, case_type: str, parentage_confirmed: bool) -> dict:
    crit: list[dict] = []
    af = v.get("pop_af")
    modes = g.inheritance
    recessive_gene = bool(modes & {"AR", "XLR"}) and not (modes & {"AD", "XLD"})

    # --- population data ---
    pop = f" ({v['pop_af_pop']})" if v.get("pop_af_pop") else ""
    bs1_cut = 0.01 if recessive_gene else 0.001
    audit_pop = None
    if af is not None and af > bs1_cut and established_pathogenic(v):
        # ClinGen BA1/BS1 exception: common but well-established pathogenic (founder) variants
        audit_pop = (f"Frequency {af:.3%}{pop} would trigger BA1/BS1, but ClinVar classifies the variant as "
                     f"{v.get('clinvar_sig')} ({v.get('clinvar_stars')}*): treated as a common pathogenic "
                     f"(founder/low-penetrance) variant - frequency criteria not applied")
    elif af is not None and af > 0.05:
        crit.append(_crit("BA1", "StandAlone", f"gnomAD grpmax allele frequency {af:.3%}{pop} > 5%"))
    elif af is not None and af > bs1_cut:
        crit.append(_crit("BS1", "Strong", f"gnomAD grpmax allele frequency {af:.3%}{pop} greater than expected for disorder"))
    elif not af:
        crit.append(_crit("PM2", "Supporting", "Absent from gnomAD exomes/genomes"))
    elif recessive_gene and af <= 0.0001:
        crit.append(_crit("PM2", "Supporting", f"Very rare in gnomAD ({af:.4%}), consistent with recessive disease"))

    # --- null variant ---
    p = _pvs1(v, g)
    if p:
        crit.append(p)

    # --- protein length change ---
    cons = set(v["consequence"])
    if cons & {"inframe_insertion", "inframe_deletion", "stop_lost"} and v["vtype"] != "lof":
        crit.append(_crit("PM4", "Moderate", "In-frame indel / stop-loss altering protein length"))

    # --- de novo ---
    if "de_novo" in v.get("models", []) or "XL_de_novo" in v.get("models", []):
        if v.get("assumed_ref"):
            crit.append(_crit("PM6", "Moderate",
                              "Assumed de novo: absent from parental VCF(s), parental coverage not verified"))
        elif parentage_confirmed:
            crit.append(_crit("PS2", "Strong", "De novo with confirmed parentage (trio Mendelian QC passed)"))
        else:
            crit.append(_crit("PM6", "Moderate", "Assumed de novo (parentage not confirmed)"))
    elif "de_novo_possible" in v.get("models", []):
        crit.append(_crit("PM6", "Supporting", "Possibly de novo - parental genotype not called"))

    # --- in trans with pathogenic (recessive) ---
    if "AR_comphet" in v.get("models", []) and v.get("_partner_pathogenic"):
        strength = "Moderate" if case_type == "trio" else "Supporting"
        crit.append(_crit("PM3", strength, "Observed in trans/possibly in trans with a pathogenic variant"
                          if strength == "Moderate" else "Observed with a pathogenic variant (phase unknown)"))
    if "AR_hom" in v.get("models", []) and af is not None and af < 0.0001:
        crit.append(_crit("PM3", "Supporting", "Homozygous in an affected individual"))

    # --- computational evidence ---
    if v["vtype"] == "missense":
        rv = v.get("revel")
        am = v.get("am_score")
        if rv is not None:
            if rv >= 0.932:
                crit.append(_crit("PP3", "Strong", f"REVEL {rv:.3f} (>=0.932)"))
            elif rv >= 0.773:
                crit.append(_crit("PP3", "Moderate", f"REVEL {rv:.3f} (>=0.773)"))
            elif rv >= 0.644:
                crit.append(_crit("PP3", "Supporting", f"REVEL {rv:.3f} (>=0.644)"))
            elif rv <= 0.016:
                crit.append(_crit("BP4", "Strong", f"REVEL {rv:.3f} (<=0.016)"))
            elif rv <= 0.183:
                crit.append(_crit("BP4", "Moderate", f"REVEL {rv:.3f} (<=0.183)"))
            elif rv <= 0.290:
                crit.append(_crit("BP4", "Supporting", f"REVEL {rv:.3f} (<=0.290)"))
        elif am is not None:
            if am >= 0.906:
                crit.append(_crit("PP3", "Moderate", f"AlphaMissense {am:.3f} (>=0.906)"))
            elif am >= 0.792:
                crit.append(_crit("PP3", "Supporting", f"AlphaMissense {am:.3f} (likely pathogenic)"))
            elif am <= 0.169:
                crit.append(_crit("BP4", "Supporting", f"AlphaMissense {am:.3f} (likely benign)"))
        if g.mis_z is not None and g.mis_z >= 3.09 and "AD" in modes:
            crit.append(_crit("PP2", "Supporting", f"Missense-constrained gene (Z={g.mis_z:.2f})"))
    sai = v.get("spliceai")
    if v["vtype"] != "lof" and sai is not None:
        if sai >= 0.2 and v["vtype"] != "missense":
            crit.append(_crit("PP3", "Supporting", f"SpliceAI delta score {sai:.2f} (>=0.2)"))
        elif sai <= 0.1 and v["vtype"] in ("synonymous", "splice_region"):
            crit.append(_crit("BP4", "Supporting", f"SpliceAI delta score {sai:.2f} (no splice impact)"))
    if v["vtype"] == "synonymous" and (sai is None or sai <= 0.1):
        crit.append(_crit("BP7", "Supporting", "Synonymous with no predicted splice impact"))

    # --- reputable source (legacy PP5/BP6, flagged as such) ---
    if clinvar_pathogenic(v, min_stars=2):
        crit.append(_crit("PP5", "Strong" if (v.get("clinvar_stars") or 0) >= 3 else "Supporting",
                          f"ClinVar {v.get('clinvar_sig')} ({v.get('clinvar_stars')}*)", source="clinvar"))
    elif v.get("clinvar_category") in ("B", "LB") and (v.get("clinvar_stars") or 0) >= 2:
        crit.append(_crit("BP6", "Supporting", f"ClinVar {v.get('clinvar_sig')} ({v.get('clinvar_stars')}*)",
                          source="clinvar"))

    pts, cls = classify_points(crit)
    codes = {c["code"] for c in crit}
    audit = [audit_pop] if audit_pop else []
    if "PP3" in codes and "BP4" in codes:  # ClinGen SVI: mutually exclusive
        audit.append("PP3 and BP4 both triggered - contradictory computational evidence")
    if any(not c["reason"] for c in crit):
        audit.append("Criterion without stated evidence")
    if any("not the +/-1,2 dinucleotide" in c["reason"] for c in crit):
        audit.append("Annotation flagged a canonical splice-site consequence that the HGVSc notation does not "
                     "support (see PVS1 reason) - verify the raw VCF alignment before relying on this variant")
    if v.get("gene") in PARALOG_PRONE_GENES:
        audit.append(f"{v['gene']} has a near-identical paralog - this call needs orthogonal confirmation "
                     "(not just a second look at this VCF) before being reported; see the PVS1 reason")
    return {"criteria": crit, "points": pts, "classification": cls, "short": SHORT[cls], "audit": audit}


def reclassify(acmg: dict) -> dict:
    pts, cls = classify_points(acmg["criteria"])
    return {**acmg, "points": pts, "classification": cls, "short": SHORT[cls]}


def is_plp(v: dict) -> bool:
    return v.get("acmg", {}).get("short") in ("P", "LP")
