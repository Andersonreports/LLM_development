"""Inheritance-model filtering for single and trio cases.

Each surviving variant gets:
  models            e.g. ["de_novo"], ["AR_hom"], ["AR_comphet"], ["AD"], ["XLR"]
  model_notes       human readable segregation notes
  comphet_partners  variant keys forming a compound-heterozygous pair
  inheritance_match True/False/None (does the gene's known inheritance fit?)
"""
from __future__ import annotations

from collections import defaultdict

from .config import settings
from .resources import KnowledgeBase
from .variants import clinvar_pathogenic, established_pathogenic

DOMINANT_MODES = {"AD", "XLD", "SD", "XL"}
RECESSIVE_MODES = {"AR", "XLR", "XL", "SD"}
X_MODES = {"XL", "XLR", "XLD"}


def is_candidate(v: dict, role: str = "proband") -> bool:
    """Generic rarity + consequence pre-filter."""
    if not v["gt_ok"].get(role) or v["zygosity"].get(role) in ("ref", "no_call"):
        return False
    af = v.get("pop_af") or 0.0
    known = clinvar_pathogenic(v, min_stars=1)
    if not established_pathogenic(v) and af > (settings.af_known_pathogenic if known else settings.af_recessive):
        return False
    if known:
        return True
    if v["vtype"] in ("lof", "missense", "inframe"):
        return True
    if (v.get("spliceai") or 0) >= 0.2:
        return True
    if v["vtype"] == "splice_region":
        return True
    if v["chrom"] == "MT" and v["vtype"] != "synonymous":
        return True
    return False


def _alt(z: str | None) -> bool:
    return z in ("het", "hom", "hemizygous", "homoplasmic", "heteroplasmic")


def _origin(v: dict) -> str:
    """For trio: where did the proband's ALT come from?"""
    zf, zm = v["zygosity"].get("father"), v["zygosity"].get("mother")
    f_alt, m_alt = _alt(zf), _alt(zm)
    if f_alt and m_alt:
        return "both"
    if f_alt:
        return "father"
    if m_alt:
        return "mother"
    called_ref = lambda r: v["zygosity"].get(r) == "ref" and (v["gt_ok"].get(r) or r in v.get("assumed_ref", []))
    if called_ref("father") and called_ref("mother"):
        return "de_novo"
    return "unknown"  # at least one parent not called


def _gene_fit(modes: set[str], model: str) -> bool | None:
    if not modes:
        return None
    if model in ("AR_hom", "AR_comphet", "AR_hom_upd"):
        return bool(modes & RECESSIVE_MODES)
    if model in ("AD", "de_novo", "de_novo_possible", "AD_inherited"):
        return bool(modes & DOMINANT_MODES)
    if model in ("XLR", "XLD", "XL_de_novo"):
        return bool(modes & X_MODES)
    if model == "MT":
        return "MT" in modes
    return None


def analyse(variants: list[dict], case_type: str, kb: KnowledgeBase, sexes: dict,
            affected: dict[str, bool], assumed_ref_roles: set[str] = frozenset()) -> list[dict]:
    """assumed_ref_roles: parents supplied as separate VCFs. A proband site absent
    from such a file is treated as reference (flagged, and never counted as a
    confirmed de novo)."""
    trio = case_type == "trio"
    proband_sex = sexes.get("proband")
    cands = [v for v in variants if is_candidate(v, "proband")]
    for v in cands:
        assumed = [r for r in assumed_ref_roles if v["zygosity"].get(r) == "no_call"
                   and v["genotypes"].get(r, {}).get("dp") is None]
        for r in assumed:
            v["zygosity"][r] = "ref"
        v["assumed_ref"] = assumed

    by_gene: dict[str, list[dict]] = defaultdict(list)
    for v in cands:
        v["models"], v["model_notes"], v["comphet_partners"] = [], [], []
        v["origin"] = _origin(v) if trio else None
        if v["chrom"] not in ("X", "Y", "MT") and v["zygosity"]["proband"] == "het":
            by_gene[v["gene"]].append(v)

    for v in cands:
        z = v["zygosity"]["proband"]
        af = v.get("pop_af") or 0.0
        known = clinvar_pathogenic(v)
        dom_ok = af <= settings.af_dominant or known
        origin = v.get("origin")
        chrom = v["chrom"]

        if chrom == "MT":
            v["models"].append("MT")
            continue

        if chrom == "X" and z == "hemizygous":
            if trio and origin == "de_novo":
                v["models"].append("XL_de_novo")
                v["model_notes"].append("Hemizygous, absent in both parents (de novo)")
            elif trio and _alt(v["zygosity"].get("father")) and not affected.get("father"):
                v["model_notes"].append("Also present in unaffected father - unlikely X-linked cause")
            else:
                v["models"].append("XLR")
                if trio and origin == "mother":
                    v["model_notes"].append("Hemizygous, inherited from heterozygous mother")
            continue

        if z == "hom":
            if trio:
                zf, zm = v["zygosity"].get("father"), v["zygosity"].get("mother")
                if (zf == "hom" and not affected.get("father")) or (zm == "hom" and not affected.get("mother")):
                    v["model_notes"].append("Homozygous in an unaffected parent - excluded as recessive cause")
                    continue
                if zf == "het" and zm == "het":
                    v["models"].append("AR_hom")
                    v["model_notes"].append("Homozygous; both parents heterozygous carriers")
                elif (zf == "het") != (zm == "het") and "ref" in (zf, zm):
                    v["models"].append("AR_hom_upd")
                    v["model_notes"].append(
                        "Homozygous but only one parent is a carrier - consider uniparental disomy or a deletion on the other allele")
                else:
                    v["models"].append("AR_hom")
            else:
                v["models"].append("AR_hom")
            continue

        # heterozygous (autosomal or female X)
        if chrom == "X" and proband_sex == "female":
            if dom_ok:
                v["models"].append("XLD")
            # female carrier for XLR handled as a carrier finding, not diagnostic
        if trio:
            if origin == "de_novo":
                v["models"].append("de_novo")
                v["model_notes"].append("Heterozygous, absent in both parents (de novo)" + (
                    "; parental reference genotype inferred from absence in parental VCF - confirm coverage"
                    if v["assumed_ref"] else ""))
            elif origin == "unknown" and dom_ok:
                v["models"].append("de_novo_possible")
                v["model_notes"].append("Parental genotype(s) not called - de novo status unconfirmed")
            elif origin in ("father", "mother") and dom_ok:
                if affected.get(origin):
                    v["models"].append("AD")
                    v["model_notes"].append(f"Inherited from affected {origin}")
                else:
                    v["models"].append("AD_inherited")
                    v["model_notes"].append(
                        f"Inherited from unaffected {origin} - causal only with reduced penetrance/variable expressivity")
        elif chrom != "X" and dom_ok:
            v["models"].append("AD")

    # compound heterozygotes
    for gene, vs in by_gene.items():
        if len(vs) < 2:
            continue
        for i, a in enumerate(vs):
            for b in vs[i + 1:]:
                if trio:
                    oa, ob = a.get("origin"), b.get("origin")
                    in_trans = {oa, ob} == {"father", "mother"} or (
                        "de_novo" in (oa, ob) and {oa, ob} & {"father", "mother"})
                    if oa == "both" or ob == "both" or oa == ob and oa in ("father", "mother"):
                        continue
                    if not in_trans and "unknown" not in (oa, ob):
                        continue
                    note = "in trans (one allele from each parent)" if in_trans else "phase unconfirmed"
                else:
                    ga, gb = a["genotypes"]["proband"], b["genotypes"]["proband"]
                    if ga.get("phased") and gb.get("phased") and ga.get("ps") and ga.get("ps") == gb.get("ps"):
                        if ga["gt"] == gb["gt"]:
                            continue  # read-backed phasing shows cis
                        note = "in trans (read-backed phasing)"
                    else:
                        note = "phase unknown - parental testing recommended"
                if a["chrom"] == b["chrom"] and abs(a["pos"] - b["pos"]) <= 20:
                    # two indel calls this close are a classic signature of the same allele
                    # being left-aligned differently by the caller/aligner, not two distinct
                    # variants - do not assume this is real compound heterozygosity
                    note += "; CAUTION: only {}bp apart - check these are not the same allele " \
                            "mis-represented twice (alignment ambiguity) before treating as biallelic".format(
                                abs(a["pos"] - b["pos"]))
                for x, y in ((a, b), (b, a)):
                    if "AR_comphet" not in x["models"]:
                        x["models"].append("AR_comphet")
                    x["comphet_partners"].append(y["key"])
                    x["model_notes"].append(f"Compound heterozygous with {y['hgvsc'] or y['variant_id']} ({note})")

    out = []
    for v in cands:
        if not v["models"]:
            continue
        g = kb.gene(v["gene"])
        modes = g.inheritance
        fits = [_gene_fit(modes, m) for m in v["models"]]
        v["inheritance_match"] = True if any(f is True for f in fits) else (
            None if all(f is None for f in fits) else False)
        v["gene_inheritance"] = sorted(modes)
        out.append(v)
    return out
