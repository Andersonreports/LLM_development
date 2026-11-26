"""The genetics tools as the model sees them (OpenAI/Groq function-calling format), the
system prompt that tells it to use them, and one dispatcher that runs a call."""
from __future__ import annotations

import json

from . import lookup

SYSTEM_PROMPT = """You are ADVAT, a genetics and bioinformatics assistant for a clinical genomics laboratory (Anderson Diagnostics & Labs). Your users are geneticists and bioinformaticians.

You have lookup tools backed by the lab's local copies of ClinVar, VEP (gene, consequence, HGVS), AlphaMissense, gnomAD (gene constraint and variant frequencies), ClinGen dosage sensitivity, HPO and the lab's exome capture kit.

Rules:
1. For any specific fact about a variant (classification, review status, rsID, position, population frequency), a gene (constraint, dosage sensitivity, associated diseases, inheritance, ClinVar counts) or an HPO term, call the matching tool first and answer from its result.
2. Cite where each fact came from, with its release, e.g. "ClinVar (local, 2026-10-19): Pathogenic, 3 stars". Include the ClinVar or gnomAD link when one is returned.
3. If a tool returns an error, nothing, or several candidates, say so plainly. Never fill the gap from memory with variant IDs, frequencies, classifications, scores or PMIDs.
4. General knowledge (ACMG/AMP criteria, HGVS rules, file formats, pipelines, inheritance concepts) you may answer yourself; say when you are unsure.
5. Coordinates are GRCh38. An allele frequency is per allele (chromosome), not per person: write e.g. "0.00012, about 1 in 8,300 alleles". For a rare heterozygous variant the carrier frequency is about twice the allele frequency; say so if you mention carriers.
6. Use plain text: short paragraphs and simple "-" lists. No markdown tables or headings, and no citation markers such as 【...】; name the source in words instead. **Bold** is fine for key results.
7. Look each variant or gene up once; do not repeat a lookup by another name for something already returned.
8. Variant evidence: consequence and HGVS come from VEP on the MANE transcript. AlphaMissense is a computational predictor (supporting-level evidence at most), not a classification. "same_amino_acid" lists pathogenic ClinVar variants at the same residue (relevant to PS1/PM5, for the expert to judge). PubMed IDs are linked to the position's rsID and may concern another allele; never state what a paper says, only that it exists. If the lookup says the REF does not match GRCh38, tell the user the position or build is wrong.
9. You support the lab's experts; you do not make the final classification. Do not invent patient details."""

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_variant",
        "description": (
            "Look up one human variant: gene, consequence, HGVS c./p. on the MANE transcript "
            "and exon (VEP), AlphaMissense score, ClinVar classification/stars/conditions, "
            "pathogenic ClinVar variants at the same amino acid or nearby, gnomAD v4.1 "
            "population frequencies, linked PubMed IDs, and whether the lab's exome capture "
            "kit covers it. Works for variants not in ClinVar too. Accepts an HGVS name "
            "with a gene or transcript (BRCA1:c.68_69del, NM_007294.4:c.68_69del, "
            "CFTR p.Phe508del), an rsID (rs80357713), a ClinVar Variation ID (VCV000017662 or "
            "17662), or GRCh38 chrom-pos-ref-alt (17-43124027-ACT-A). One variant per call."),
        "parameters": {"type": "object", "properties": {
            "variant": {"type": "string", "description": "the variant, in one of the forms above"}},
            "required": ["variant"]}}},
    {"type": "function", "function": {
        "name": "lookup_gene",
        "description": (
            "Look up one gene by official HGNC symbol: gnomAD v4.1 constraint (pLI, LOEUF, "
            "missense z), ClinGen dosage sensitivity (haploinsufficiency/triplosensitivity "
            "scores), diseases linked in HPO/OMIM/Orphanet with inheritance, and counts of "
            "ClinVar variants by classification."),
        "parameters": {"type": "object", "properties": {
            "gene": {"type": "string", "description": "HGNC gene symbol, e.g. SCN1A"}},
            "required": ["gene"]}}},
    {"type": "function", "function": {
        "name": "lookup_phenotype",
        "description": (
            "Look up an HPO phenotype term by ID (HP:0001250) or by name (seizure, global "
            "developmental delay): definition, synonyms, parent terms, and the genes most "
            "often annotated to it."),
        "parameters": {"type": "object", "properties": {
            "phenotype": {"type": "string", "description": "HPO ID or phenotype name"}},
            "required": ["phenotype"]}}},
]

# Each tool takes one string. Accept it under any argument name: models sometimes
# rename the parameter (e.g. "query" for "variant").
_FUNCS = {
    "lookup_variant": lookup.lookup_variant,
    "lookup_gene": lookup.lookup_gene,
    "lookup_phenotype": lookup.lookup_phenotype,
}

LABELS = {"lookup_variant": "variant", "lookup_gene": "gene", "lookup_phenotype": "phenotype"}


def available() -> tuple[bool, str]:
    return lookup.available()


def run(name: str, raw_args: str) -> tuple[dict, dict]:
    """Run one tool call. Returns (result for the model, short event for the browser)."""
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": f"bad arguments: {e}"}, {"name": name, "query": raw_args,
                                                  "ok": False, "summary": "bad arguments"}
    query = next(iter(args.values()), "") if args else ""
    fn = _FUNCS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}, {"name": name, "query": query, "ok": False,
                                                   "summary": "unknown tool"}
    try:
        result = fn(str(query))
    except lookup.LookupError_ as e:
        result = {"error": str(e)}
    except Exception as e:  # noqa: BLE001 - a broken lookup must not kill the chat
        result = {"error": f"{type(e).__name__}: {e}"}
    return result, {"name": name, "label": LABELS.get(name, name), "query": query,
                    "ok": "error" not in result, "summary": _summary(name, result),
                    "sources": result.get("sources", []), "links": _links(result)}


def _summary(name: str, r: dict) -> str:
    if "error" in r:
        return r["error"][:160]
    if name == "lookup_variant":
        cv = r.get("clinvar") or []
        bits = []
        a = r.get("annotation")
        a = a[0] if isinstance(a, list) else a
        if a and a.get("gene"):
            bits.append(f"{a['gene']} {a.get('hgvs_p') or a.get('hgvs_c', '').split(':')[-1]}"
                        f" ({a.get('consequence', '')})")
            if a.get("alphamissense"):
                bits.append(f"AlphaMissense {a['alphamissense']['score']}")
        if cv:
            bits.append(f"ClinVar: {cv[0]['classification']}"
                        + (f" ({cv[0]['review_stars']}★)" if cv[0]["review_stars"] is not None
                           else "") + (f" +{len(cv) - 1} more" if len(cv) > 1 else ""))
        else:
            bits.append("not in ClinVar")
        g = (r.get("gnomad") or [None])[0]
        local = (a or {}).get("gnomad_local") or {}
        if g and "error" not in g:
            bits.append(f"gnomAD AF {g['combined_allele_frequency']}" if g.get("found")
                        else "absent from gnomAD")
        elif local.get("exomes_af") or local.get("genomes_af"):
            bits.append(f"gnomAD AF {local.get('exomes_af') or local.get('genomes_af')} (local)")
        elif local:
            bits.append("absent from gnomAD (local)")
        if isinstance(r.get("clinvar_nearby"), dict) and r["clinvar_nearby"]["same_amino_acid"]:
            bits.append(f"{len(r['clinvar_nearby']['same_amino_acid'])} P/LP at same residue")
        if r.get("capture_kit"):
            bits.append("in capture kit" if r["capture_kit"]["in_target"]
                        else "outside capture kit")
        return " · ".join(bits)
    if name == "lookup_gene":
        bits = []
        c = r.get("gnomad_constraint")
        if c:
            bits.append(f"LOEUF {c['LOEUF']}, pLI {c['pLI']:.2g}" if c["pLI"] is not None
                        else f"LOEUF {c['LOEUF']}")
        if r.get("clingen_dosage_sensitivity"):
            bits.append(f"HI score {r['clingen_dosage_sensitivity']['haploinsufficiency_score']}")
        if r.get("diseases"):
            bits.append(f"{len(r['diseases'])} linked diseases")
        return " · ".join(bits) or "found"
    if name == "lookup_phenotype":
        t = r.get("term", {})
        return f"{t.get('id')} {t.get('name')} · {r.get('genes_annotated', {}).get('total', 0)} genes"
    return "done"


def _links(r: dict) -> list[str]:
    links = [x.get("clinvar_url") for x in r.get("clinvar") or [] if x.get("clinvar_url")]
    links += [g.get("url") for g in r.get("gnomad") or [] if g and g.get("url")]
    for key in ("clingen_dosage_sensitivity", "term"):
        if isinstance(r.get(key), dict) and r[key].get("url"):
            links.append(r[key]["url"])
    return links[:4]
