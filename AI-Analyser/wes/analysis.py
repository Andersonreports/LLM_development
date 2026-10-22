"""Case-level analysis: candidate prioritisation, secondary findings, carrier
screening and the rule-based (pre-AI) result flag."""
from __future__ import annotations

from collections import defaultdict

from . import acmg, inheritance
from .config import settings
from .phenotype import gene_score
from .resources import ACMG_SF_NOTES, ACMG_SF_V3_2, KnowledgeBase
from .variants import clinvar_pathogenic, established_pathogenic

FLAGS = ("POSITIVE", "INCONCLUSIVE", "NEGATIVE", "CARRIER")
CLASS_WEIGHT = {"P": 1.0, "LP": 0.8, "VUS": 0.35, "LB": 0.0, "B": 0.0}


def _score(v: dict) -> float:
    cls_w = CLASS_WEIGHT.get(v["acmg"]["short"], 0)
    pts = max(0, min(v["acmg"]["points"], 12)) / 12
    s = 0.40 * v.get("phenotype_score", 0) + 0.30 * cls_w + 0.15 * pts
    if v.get("inheritance_match") is True:
        s += 0.10
    elif v.get("inheritance_match") is False:
        s -= 0.15
    if clinvar_pathogenic(v):
        s += 0.05
    models = set(v.get("models", []))
    if models <= {"AD_inherited"}:
        s -= 0.10
    if "de_novo" in models or "XL_de_novo" in models:
        s += 0.05
    return round(s, 4)


def _annotate(v: dict, kb: KnowledgeBase, hpo_ids: list[str], case_type: str, parentage: bool) -> None:
    g = kb.gene(v["gene"])
    v["gene_info"] = g.to_dict()
    v["phenotype_score"], v["phenotype_matches"] = gene_score(kb, hpo_ids, v["gene"])
    v["acmg"] = acmg.evaluate(v, g, case_type, parentage)


def prioritise(variants: list[dict], case_type: str, kb: KnowledgeBase, hpo_ids: list[str],
               sexes: dict, affected: dict, parentage: bool,
               assumed_ref_roles: set[str] = frozenset()) -> list[dict]:
    cands = inheritance.analyse(variants, case_type, kb, sexes, affected, assumed_ref_roles)
    by_key = {v["key"]: v for v in cands}
    for v in cands:
        _annotate(v, kb, hpo_ids, case_type, parentage)
    # second pass: PM3 needs to know whether the comphet partner is P/LP
    for v in cands:
        if v["comphet_partners"]:
            v["_partner_pathogenic"] = any(
                _known_plp(by_key[k]) for k in v["comphet_partners"] if k in by_key)
            if v["_partner_pathogenic"]:
                v["acmg"] = acmg.evaluate(v, kb.gene(v["gene"]), case_type, parentage)
    for v in cands:
        v["rank_score"] = _score(v)
        v.pop("_partner_pathogenic", None)
    # benign/likely benign never ranked as candidates
    cands = [v for v in cands if v["acmg"]["short"] not in ("B", "LB")]
    cands.sort(key=lambda v: -v["rank_score"])
    for i, v in enumerate(cands, 1):
        v["rank"] = i
    return cands


def _known_plp(v: dict) -> bool:
    return acmg.is_plp(v) or clinvar_pathogenic(v, min_stars=1)


def secondary_findings(variants: list[dict], kb: KnowledgeBase, role: str, sex: str | None,
                       case_type: str) -> list[dict]:
    """P/LP variants in ACMG SF v3.2 genes, respecting each gene's reportable genotype."""
    hits: dict[str, list[dict]] = defaultdict(list)
    for v in variants:
        if v["gene"] not in ACMG_SF_V3_2:
            continue
        if not v["gt_ok"].get(role) or v["zygosity"].get(role) in ("ref", "no_call"):
            continue
        if (v.get("pop_af") or 0) > settings.af_known_pathogenic and not established_pathogenic(v):
            continue
        if "acmg" not in v:
            v = dict(v, models=[])
            v["acmg"] = acmg.evaluate(v, kb.gene(v["gene"]), case_type, False)
        if not _known_plp(v):
            continue
        hits[v["gene"]].append(v)
    out = []
    for gene, vs in hits.items():
        mode = ACMG_SF_V3_2[gene]
        if gene == "TTN":
            vs = [v for v in vs if v["vtype"] == "lof"]
        if gene == "HFE":
            vs = [v for v in vs if "Cys282Tyr" in (v.get("hgvsp") or "") and v["zygosity"][role] == "hom"]
        if mode == "AR":
            if not (any(v["zygosity"][role] == "hom" for v in vs) or len(vs) >= 2):
                continue
        for v in vs:
            out.append({**v, "sf_mode": mode, "sf_note": ACMG_SF_NOTES.get(gene)})
    return out


def carrier_findings(variants: list[dict], kb: KnowledgeBase, role: str, sex: str | None,
                     case_type: str) -> list[dict]:
    """Heterozygous P/LP variants in genes with autosomal-recessive or X-linked disease."""
    out = []
    for v in variants:
        z = v["zygosity"].get(role)
        if not v["gt_ok"].get(role) or z not in ("het",):
            continue
        if (v.get("pop_af") or 0) > settings.af_known_pathogenic and not established_pathogenic(v):
            continue
        g = kb.gene(v["gene"])
        modes = g.inheritance
        x_carrier = v["chrom"] == "X" and sex == "female" and modes & {"XLR", "XL"}
        if not (modes & {"AR"} or x_carrier):
            continue
        if v["chrom"] == "X" and not x_carrier:
            continue
        if not inheritance.is_candidate(v, role):
            continue
        vv = dict(v, models=[])
        vv["gene_info"] = g.to_dict()
        vv["acmg"] = acmg.evaluate(vv, g, case_type, False)
        if not _known_plp(vv):
            continue
        vv["carrier_mode"] = "XLR" if x_carrier else "AR"
        out.append(vv)
    return out


def couple_analysis(variants: list[dict], kb: KnowledgeBase, sexes: dict) -> dict:
    p1 = carrier_findings(variants, kb, "partner1", sexes.get("partner1"), "couple")
    p2 = carrier_findings(variants, kb, "partner2", sexes.get("partner2"), "couple")
    genes1 = {v["gene"] for v in p1 if v["carrier_mode"] == "AR"}
    genes2 = {v["gene"] for v in p2 if v["carrier_mode"] == "AR"}
    shared = sorted(genes1 & genes2)
    xl = [v for v in p1 + p2 if v["carrier_mode"] == "XLR"]  # only female partners yield XLR
    at_risk = []
    for gene in shared:
        at_risk.append({
            "gene": gene, "mode": "AR", "risk": "25% risk of an affected child per pregnancy",
            "partner1": [v["key"] for v in p1 if v["gene"] == gene],
            "partner2": [v["key"] for v in p2 if v["gene"] == gene],
            "diseases": kb.gene(gene).to_dict()["diseases"],
        })
    for v in xl:
        at_risk.append({
            "gene": v["gene"], "mode": "XLR",
            "risk": "50% of sons affected, 50% of daughters carriers",
            "partner1": [v["key"]] if v["zygosity"].get("partner1") == "het" else [],
            "partner2": [v["key"]] if v["zygosity"].get("partner2") == "het" else [],
            "diseases": kb.gene(v["gene"]).to_dict()["diseases"],
        })
    return {"partner1": p1, "partner2": p2, "shared_genes": shared, "at_risk": at_risk}


def rule_based_flag(case_type: str, candidates: list[dict], couple: dict | None,
                    has_phenotype: bool) -> dict:
    """Deterministic flag used as a baseline and as fallback if AI is unavailable."""
    if case_type == "couple":
        c = couple or {}
        if c.get("at_risk"):
            genes = ", ".join(sorted({r["gene"] for r in c["at_risk"]}))
            return {"flag": "POSITIVE", "reason": f"At-risk couple: shared/X-linked carrier status in {genes}",
                    "keys": [k for r in c["at_risk"] for k in r["partner1"] + r["partner2"]]}
        if c.get("partner1") or c.get("partner2"):
            return {"flag": "CARRIER", "reason": "Carrier status identified in one or both partners for different genes",
                    "keys": [v["key"] for v in c.get("partner1", []) + c.get("partner2", [])]}
        return {"flag": "NEGATIVE", "reason": "No P/LP carrier variants identified in either partner", "keys": []}

    if not has_phenotype:
        # No HPO terms were available (no indication, no AI extraction, nothing entered
        # manually) - there is no basis to confirm ANY candidate explains this patient's
        # presentation. A P/LP classification alone is not evidence of phenotype relevance:
        # most exomes carry several rare LoF/missense variants in genes unrelated to the
        # indication. Never auto-declare POSITIVE here; surface the strongest ACMG hits
        # for a human (or AI, once phenotype data is supplied) to judge on clinical grounds.
        candidate_hits = [v for v in candidates if v["acmg"]["short"] in ("P", "LP")]
        if candidate_hits:
            candidate_hits.sort(key=lambda v: -v["rank_score"])
            return {"flag": "INCONCLUSIVE",
                    "reason": "No phenotype information (clinical indication or HPO terms) was available, so "
                              "candidate P/LP variant(s) could not be checked against the patient's presentation. "
                              "Add the indication/HPO terms and re-run, or assess gene-disease relevance manually "
                              "before reporting any of these as explaining the phenotype.",
                    "keys": [v["key"] for v in candidate_hits[:5]]}
        return {"flag": "NEGATIVE",
                "reason": "No P/LP variants identified. No phenotype information was available to check relevance "
                          "of any uncertain findings.",
                "keys": []}

    min_pheno = 0.3
    by_key = {v["key"]: v for v in candidates}
    positive, inconclusive = [], []
    for v in candidates:
        cls = "LP" if v["acmg"]["short"] == "VUS" and _known_plp(v) else v["acmg"]["short"]
        fit = v.get("inheritance_match") is not False
        pheno = v.get("phenotype_score", 0) >= min_pheno
        # a gene with no known disease at all cannot be confirmed to explain this patient's
        # phenotype, regardless of how the (vacuous) inheritance-fit check comes out
        known_disease = bool(v.get("gene_info", {}).get("diseases"))
        models = set(v["models"])
        recessive_only = models <= {"AR_comphet"}
        if cls in ("P", "LP") and fit and pheno and known_disease:
            if recessive_only:
                partners = [by_key[k] for k in v["comphet_partners"] if k in by_key]
                if any(_known_plp(p) for p in partners):
                    positive.append(v)
                elif partners:
                    inconclusive.append(v)
            elif models & {"AD_inherited"} and not models - {"AD_inherited"}:
                inconclusive.append(v)
            else:
                positive.append(v)
        elif cls in ("P", "LP") and pheno and (v.get("inheritance_match") is None or not known_disease):
            inconclusive.append(v)
        elif cls == "VUS" and fit and v.get("phenotype_score", 0) >= max(min_pheno, 0.45):
            inconclusive.append(v)
    if positive:
        return {"flag": "POSITIVE", "reason": "P/LP variant(s) with consistent inheritance and phenotype match",
                "keys": [v["key"] for v in positive[:5]]}
    if inconclusive:
        return {"flag": "INCONCLUSIVE",
                "reason": "Variant(s) of uncertain significance or incomplete genotype in phenotype-relevant gene(s)",
                "keys": [v["key"] for v in inconclusive[:5]]}
    return {"flag": "NEGATIVE", "reason": "No P/LP or phenotype-relevant VUS identified", "keys": []}
