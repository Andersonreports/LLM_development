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
    # laterality/direction words: too generic, they appear in every clinical note
    "right", "left", "upper", "lower", "proximal", "distal", "anterior", "posterior",
    "medial", "lateral", "superior", "inferior", "ipsilateral", "contralateral",
}

# HPO terms that are clinical modifiers, not symptoms. "Family history revealed..." is a
# section header in a referral, not a phenotype, and matching it adds noise to scoring.
_HPO_META_BLOCKLIST: frozenset[str] = frozenset({
    "HP:0032316",  # Family history
    "HP:0032443",  # Past medical history
    "HP:0012823",  # Clinical modifier
    "HP:0000005",  # Mode of inheritance
    "HP:0000006",  # Autosomal dominant inheritance
    "HP:0000007",  # Autosomal recessive inheritance
    "HP:0003674",  # Onset
    "HP:0003581",  # Adult onset
    "HP:0003593",  # Infantile onset
    "HP:0003621",  # Juvenile onset
    "HP:0003623",  # Neonatal onset
    "HP:0011420",  # Age of death
    "HP:0003829",  # Typified by incomplete penetrance
    "HP:0031797",  # Clinical course
    "HP:0040006",  # Mortality/Aging
    "HP:0034335",  # Inheritance qualifier
    "HP:0034345",  # Mendelian inheritance
})

# clinical shorthand vs HPO compound words: "upward gazing" in a note, "upgaze" in HPO
_TERM_NORMALISE = [
    (re.compile(r"\bupgaz"), "upward gaz"),
    (re.compile(r"\bdowngaz"), "downward gaz"),
]

# qualifiers that HPO names carry but clinical prose often drops ("paroxysmal tonic
# upgaze" written as "paroxysmal upward gaze"); a 3+ word name is also tried without one
_QUALIFIER_SKIP_WORDS = {
    "tonic", "clonic", "atonic", "spastic", "intermittent", "episodic",
    "idiopathic", "essential", "benign", "postnatal", "prenatal",
}

# British -> American, applied to both the note and HPO names. HPO uses American spelling;
# referral notes here are mostly British ("oedema", "anaemia", "haematuria", "diarrhoea").
_BRITISH_TO_AMERICAN = [
    (re.compile(r"\boe"), "e"),
    (re.compile(r"ae(?=[a-z])"), "e"),
    (re.compile(r"rrhoe"), "rrhea"),
    (re.compile(r"our\b"), "or"),
    (re.compile(r"ise\b"), "ize"),
    (re.compile(r"isation\b"), "ization"),
    (re.compile(r"lyse\b"), "lyze"),
    (re.compile(r"centres?\b"), "center"),
    (re.compile(r"fibre"), "fiber"),
]

_ARTICLE_RE = re.compile(r"\b(?:the|an?)\s+")

# a negation word/prefix immediately before a match - "non-consanguineous", "no evidence
# of X", "denies X", "without X" - means the opposite of the matched term, not its presence.
# Checked against the ~50 characters right before each match, not the whole document.
_NEGATION_RE = re.compile(
    r"(?:\bnon|\bno|\bnot|\bwithout|\babsent|\bnegative for|\bdenies|\bdenied|\bruled out|"
    r"\bexclud(?:e|es|ed|ing)|\bunremarkable for|\bno\s+known|\bno\s+prior|\bno\s+past)"
    r"(?:\s+(?:evidence|signs?|history|features?|findings?|complaints?)\s+of)?[\s\-,;]*$"
)

# a term after "family history", "sister has", "mother with" describes a relative, not the
# patient, so it must not count towards the patient's phenotype. Same sentence, within
# ~120 characters before the match.
_FAMILY_HIST_RE = re.compile(
    r"\b(?:family history|family members?|relatives?|"
    r"(?:(?:his|her|their)\s+)?(?:mother|father|sister|brother|sibling|parent|grandparent|"
    r"uncle|aunt|cousin|son|daughter|child(?:ren)?)\b[^.]{0,40}(?:has|had|have|having|with|of|reported))"
    r"[^.]{0,80}$",
    re.IGNORECASE,
)

_SUFFIX = r"(?:s|es|'s|ic|al|ity|ous|ing|ed)?(?![a-z])"


def _sub_all(text: str, rules) -> str:
    for pat, rep in rules:
        text = pat.sub(rep, text)
    return text


def _normalise(text: str) -> str:
    return _sub_all(_sub_all(text, _BRITISH_TO_AMERICAN), _TERM_NORMALISE)


def _strip_articles(text: str) -> str:
    return " ".join(_ARTICLE_RE.sub(" ", text).split())


def _search_forms(name: str) -> list[tuple[str, bool]]:
    """The ways an HPO name can appear in prose, each paired with whether it is searched
    in the article-stripped copy of the note: as written; without articles; singular;
    without a final 'e' (gaze -> gazing); without one qualifier word."""
    forms = [(name, False)]
    bare = _strip_articles(name)
    if bare != name:
        forms.append((bare, True))
    if bare.endswith("s") and len(bare) > 5 and not bare.endswith("ss"):
        forms.append((bare[:-1], True))
    if bare.endswith("e") and len(bare) > 4:
        forms.append((bare[:-1], True))
    words = bare.split()
    if len(words) >= 3:
        for i, w in enumerate(words):
            if w in _QUALIFIER_SKIP_WORDS:
                short = " ".join(words[:i] + words[i + 1:])
                forms.append((short, True))
                if short.endswith("e") and len(short) > 4:
                    forms.append((short[:-1], True))
    return forms


def extract_from_text(kb: KnowledgeBase, text: str) -> list[dict]:
    """Deterministic, LLM-independent HPO extraction: matches clinical-history text
    against known HPO term names/synonyms. Used whenever no AI is configured, and as a
    floor/safety-net even when AI is available - free text should never go completely
    unused just because an LLM isn't reachable.

    Tolerates British spelling, plurals and common suffixes, dropped articles and
    qualifiers. A term immediately preceded by a negation ("non-consanguineous", "no
    family history of X") or describing a relative ("sister has seizures") is reported
    as excluded, not present. HPO meta-terms (onset, inheritance, "family history") are
    skipped."""
    onto = kb.ontology
    if not onto or not text or not text.strip():
        return []
    texts = {False: _normalise(text.lower())}
    texts[True] = _strip_articles(texts[False])

    names = sorted(onto._name_index.items(), key=lambda kv: -len(kv[0]))
    seen_ids: set[str] = set()
    # longest names go first; a shorter name inside an already-matched span is the same
    # mention ("seizure" inside "febrile seizure"), not a second finding. Spans are kept
    # per copy of the note because the two copies have different offsets.
    spans: dict[bool, list[tuple[int, int]]] = {False: [], True: []}
    out = []
    for name, tid in names:
        if len(name) < 4 or name in _MODIFIER_STOPWORDS or tid in seen_ids or tid in _HPO_META_BLOCKLIST:
            continue
        for form, stripped in _search_forms(_normalise(name)):
            hay = texts[stripped]
            if form not in hay:   # cheap pre-check; the regex below is the real test
                continue
            m = re.search(r"(?<![a-z])" + re.escape(form) + _SUFFIX, hay)
            if m:
                break
        else:
            continue
        if any(s <= m.start() and m.end() <= e for s, e in spans[stripped]):
            continue
        seen_ids.add(tid)
        spans[stripped].append((m.start(), m.end()))
        negated = bool(_NEGATION_RE.search(hay[max(0, m.start() - 50):m.start()]))
        relative = bool(_FAMILY_HIST_RE.search(hay[max(0, m.start() - 120):m.start()]))
        out.append({"id": tid, "name": onto.name(tid), "excluded": negated or relative})
    return out
