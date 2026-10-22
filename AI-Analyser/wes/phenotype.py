"""Patient-phenotype to gene similarity (Resnik best-match, normalised 0-1)."""
from __future__ import annotations

import re

from .resources import KnowledgeBase


def validate_terms(kb: KnowledgeBase, terms: list[dict]) -> list[dict]:
    """Resolve IDs/alt IDs/labels against hp.obo; drop anything unresolvable."""
    onto = kb.ontology
    out, seen = [], set()
    for t in terms:
        tid = None
        if onto:
            if t.get("id"):
                tid = onto.resolve(t["id"])
            if tid is None and t.get("name"):
                tid = onto.by_name(t["name"])
        else:
            tid = t.get("id")
        if tid and tid not in seen:
            seen.add(tid)
            out.append({"id": tid, "name": onto.name(tid) if onto else t.get("name", tid),
                        "excluded": bool(t.get("excluded"))})
    return out


def gene_score(kb: KnowledgeBase, patient_terms: list[str], gene: str) -> tuple[float, list[dict]]:
    onto = kb.ontology
    g = kb.genes.get(gene)
    if not onto or not g or not g.hpo or not patient_terms:
        return 0.0, []
    ic = kb.term_ic
    total = 0.0
    self_total = 0.0
    matches = []
    for t in patient_terms:
        anc = onto.ancestors(t)
        self_ic = ic.get(t, max((ic.get(a, 0.0) for a in anc), default=0.0))
        self_total += self_ic
        best_ic, best_term = 0.0, None
        for a in anc & g.hpo:
            v = ic.get(a, 0.0)
            if v > best_ic:
                best_ic, best_term = v, a
        total += best_ic
        if best_term and best_ic > 0:
            matches.append({"patient": t, "patient_name": onto.name(t), "match": best_term,
                            "match_name": onto.name(best_term), "ic": round(best_ic, 2),
                            "exact": best_term == t})
    if self_total <= 0:
        return 0.0, matches
    matches.sort(key=lambda m: -m["ic"])
    return round(min(1.0, total / self_total), 3), matches


def indication_genes(kb: KnowledgeBase, patient_terms: list[str], extra: list[str], limit: int,
                     min_score: float) -> list[str]:
    """Genes relevant to the indication: best phenotype matches plus any named
    explicitly (e.g. genes the clinician suspects). Alphabetical for the report."""
    scored = []
    if kb.ontology and patient_terms:
        for sym, g in kb.genes.items():
            if not g.hpo or not g.diseases:
                continue
            sc, _ = gene_score(kb, patient_terms, sym)
            if sc >= min_score:
                scored.append((sc, sym))
        scored.sort(reverse=True)
    genes = {sym for _, sym in scored[:limit]}
    genes |= {g.strip().upper() for g in extra if g and g.strip()}
    return sorted(genes)


def read_coverage(path) -> dict[str, float]:
    """Gene -> % of coding region covered, from a 2+ column TSV/CSV (header optional;
    first column gene symbol, last numeric column the percentage)."""
    import csv
    out: dict[str, float] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            text = fh.read()
    except OSError:
        return out
    delim = "\t" if "\t" in text.splitlines()[0] else ","
    for row in csv.reader(text.splitlines(), delimiter=delim):
        if len(row) < 2:
            continue
        nums = []
        for cell in row[1:]:
            try:
                nums.append(float(cell.strip().rstrip("%")))
            except ValueError:
                pass
        if nums:
            out[row[0].strip().upper()] = nums[-1]
    return out


_MODIFIER_STOPWORDS = {
    "severe", "moderate", "mild", "occasional", "frequent", "rare", "peripheral", "central",
    "bilateral", "unilateral", "progressive", "congenital", "acquired", "chronic", "acute",
    "early", "late", "generalized", "focal", "diffuse", "multiple", "single", "normal",
    "abnormal", "increased", "decreased", "elevated", "low", "high", "type", "onset",
    "common", "variable", "typical", "recurrent", "persistent", "isolated", "classic",
}


# a negation word/prefix immediately before a match - "non-consanguineous", "no evidence
# of X", "denies X", "without X" - means the opposite of the matched term, not its presence.
# Checked against the ~30 characters right before each match, not the whole document.
_NEGATION_RE = re.compile(
    r"(?:\bnon|\bno|\bnot|\bwithout|\babsent|\bnegative for|\bdenies|\bdenied|\bruled out|"
    r"\bexclud(?:e|es|ed|ing)|\bunremarkable for)"
    r"(?:\s+(?:evidence|signs?|history|features?|findings?)\s+of)?[\s-]*$"
)


def extract_from_text(kb: KnowledgeBase, text: str) -> list[dict]:
    """Deterministic, LLM-independent HPO extraction: matches clinical-history text
    against known HPO term names/synonyms. Used whenever no AI is configured, and as a
    floor/safety-net even when AI is available - free text should never go completely
    unused just because an LLM isn't reachable. A term immediately preceded by a negation
    ("non-consanguineous", "no family history of X") is reported as excluded, not present."""
    onto = kb.ontology
    if not onto or not text or not text.strip():
        return []
    text_l = text.lower()
    names = sorted(onto._name_index.items(), key=lambda kv: -len(kv[0]))
    seen_ids: set[str] = set()
    out = []
    for name, tid in names:
        if len(name) < 4 or name in _MODIFIER_STOPWORDS or tid in seen_ids:
            continue
        m = re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", text_l)
        if m:
            seen_ids.add(tid)
            negated = bool(_NEGATION_RE.search(text_l[max(0, m.start() - 30):m.start()]))
            out.append({"id": tid, "name": onto.name(tid), "excluded": negated})
    return out
