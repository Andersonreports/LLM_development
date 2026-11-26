"""The genetics tools as the model sees them (OpenAI/Groq function-calling format), the
system prompt that tells it to use them, and one dispatcher that runs a call."""
from __future__ import annotations

import json

from . import acmg, lookup

SYSTEM_PROMPT = """You are ADVAT, a genetics and bioinformatics assistant for a clinical genomics laboratory (Anderson Diagnostics & Labs). Your users are geneticists and bioinformaticians.

You have lookup tools backed by downloaded copies of PUBLIC databases (they contain no data from this lab): ClinVar, VEP (gene, consequence, HGVS), AlphaMissense, gnomAD (gene constraint and variant frequencies), ClinGen dosage sensitivity, HPO plus the lab's exome capture kit regions (the only lab-specific data). Never call these databases "the lab's" databases; say e.g. "ClinVar (public, 2026-10-19 release)".
When a tool returns a number, quote it exactly; never add a related figure from memory (e.g. do not state pLI if only LOEUF was returned, and never contradict a returned value).

Rules:
1. For any specific fact about a variant (classification, review status, rsID, position, population frequency), a gene (constraint, dosage sensitivity, associated diseases, inheritance, ClinVar counts) or an HPO term, call the matching tool first and answer from its result.
2. Cite where each fact came from, e.g. "ClinVar (local, 2026-10-19): Pathogenic, 3 stars". Give a release date ONLY if it appears in that same source's own entry in "sources"; never copy one source's date (e.g. ClinVar's) onto another. Include the ClinVar or gnomAD link when one is returned.
3. If a tool returns an error, nothing, or several candidates, say so plainly. Never fill the gap from memory with variant IDs, frequencies, classifications, scores or PMIDs.
4. General knowledge (ACMG/AMP criteria, HGVS rules, file formats, pipelines, inheritance concepts) you may answer yourself; say when you are unsure.
5. Coordinates are GRCh38. An allele frequency is per allele (chromosome), not per person: write e.g. "0.00012, about 1 in 8,300 alleles". For a rare heterozygous variant the carrier frequency is about twice the allele frequency; say so if you mention carriers.
6. Format: short paragraphs, "-" lists, **bold** for key results, and a markdown table when listing several criteria or variants. No headings, and no citation markers such as 【...】; name the source in words instead. Be concise: one row or line per point, no repetition between a table and a summary.
7. Look each variant or gene up once; do not repeat a lookup by another name for something already returned.
8. Variant evidence: consequence and HGVS come from VEP on the MANE transcript. "same_protein_change_established" is PS1-type evidence; "different_missense_same_residue_established" is PM5-type evidence (never call it PS1). "same_residue_weak_records" (0-1 star) support neither. PubMed IDs are linked to the position's rsID and may concern another allele; never state what a paper says, only that it exists. If the lookup says the REF does not match GRCh38, tell the user the position or build is wrong.
9. ACMG: the variant lookup returns "acmg_notes", computed from the evidence by rule. Follow them, including the strength modifiers they give (e.g. PM2_Supporting, not PM2 Moderate), and do not contradict them from memory. Never invent a strength modifier the notes or the official definition do not give. Before stating what a criterion means, get its official definition from acmg_criterion rather than recalling it - request, in a single call, only the codes you will discuss that acmg_notes does not already settle (at most 8).
10. Every acmg_notes item marked NOT ASSESSABLE, NOT MET, "Do NOT" or "MUST" is binding; restate those verdicts as given. Never compute a frequency threshold (e.g. for BS1) yourself.
11. Use three distinct verdicts and never blur them: "met" (the evidence satisfies it), "not met" (the evidence shows it does not apply), "not assessable" (the data needed are missing). Missing data is never "not met".
12. PP4 depends on the PATIENT's phenotype: if the user gave none, PP4 is "not assessable". If given, PP4 needs a phenotype highly specific to a disease with a single genetic cause (e.g. cystic fibrosis with raised sweat chloride -> CFTR). "Genetically heterogeneous" means many different GENES can cause it (developmental and epileptic encephalopathy, intellectual disability, autism, ataxia, cardiomyopathy); many variants in one gene is not heterogeneity.
13. State ClinGen gene-disease validity ("Definitive", "Strong", ...) ONLY from lookup_gene's clingen_gene_validity, quoting the disease, classification and date; never from memory, and never from the HPO "diseases" list. Mention that gene-specific ClinGen VCEP rules may override general ones.
14. You support the lab's experts; you do not make the final classification. Do not invent patient details."""

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
            "Look up one gene by official HGNC symbol: ClinGen gene-disease validity "
            "(Definitive/Strong/Moderate/...), gnomAD v4.1 constraint (pLI, LOEUF, "
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
    {"type": "function", "function": {
        "name": "acmg_criterion",
        "description": (
            "Official ACMG/AMP 2015 definitions of variant-classification criteria, with "
            "default strength, cautions and ClinGen SVI modifications. Ask for ALL the codes "
            "you need in ONE call, comma-separated (e.g. 'PVS1, PM2, PM4, BS1, BS2'). Use "
            "'combining' for the rules that combine criteria into Pathogenic/Likely "
            "pathogenic/VUS/Likely benign/Benign, or 'all' for a one-line list."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "one or more criterion codes, comma-separated; or 'combining' or 'all'"}},
            "required": ["code"]}}},
]

# Each tool takes one string. Accept it under any argument name: models sometimes
# rename the parameter (e.g. "query" for "variant").
_FUNCS = {
    "lookup_variant": lookup.lookup_variant,
    "lookup_gene": lookup.lookup_gene,
    "lookup_phenotype": lookup.lookup_phenotype,
    "acmg_criterion": acmg.acmg_criterion,
}

LABELS = {"lookup_variant": "variant", "lookup_gene": "gene", "lookup_phenotype": "phenotype",
          "acmg_criterion": "ACMG rule"}


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
        nb = r.get("clinvar_nearby") or {}
        if nb.get("same_protein_change_established"):
            bits.append(f"{len(nb['same_protein_change_established'])} established P/LP same "
                        f"protein change (PS1-type)")
        if isinstance(nb.get("different_missense_same_residue_established"), list) and \
                nb["different_missense_same_residue_established"]:
            bits.append(f"{len(nb['different_missense_same_residue_established'])} established "
                        f"P/LP other missense same residue (PM5-type)")
        if nb.get("same_residue_weak_records"):
            bits.append(f"{len(nb['same_residue_weak_records'])} weak (0-1★) same-residue record")
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
        v = r.get("clingen_gene_validity")
        if isinstance(v, list) and v:
            best = [x for x in v if x["classification"] == "Definitive"]
            bits.append(f"ClinGen validity: {len(v)} curation(s)"
                        + (f", {len(best)} Definitive" if best else ""))
        elif v:
            bits.append("no ClinGen validity curation")
        if r.get("diseases"):
            bits.append(f"{len(r['diseases'])} linked diseases")
        return " · ".join(bits) or "found"
    if name == "acmg_criterion":
        if r.get("code"):
            return f"{r['code']}: {r['strength']}"
        if r.get("combining_rules"):
            return "combining rules"
        crit = r.get("criteria") or {}
        return ", ".join(crit)[:120] if crit and all(isinstance(v, dict) for v in crit.values())             else "criteria list"
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
