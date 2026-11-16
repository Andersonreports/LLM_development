"""Patient-phenotype to gene similarity (Resnik best-match, normalised 0-1)."""
from __future__ import annotations

import re

from .resources import KnowledgeBase


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalise(s.lower())))


def _resolve_term(kb: KnowledgeBase, t: dict) -> str | None:
    """The HPO id a term means. When an id and a label disagree, the LABEL wins: a small
    local model writes the right label with a wrong number far more often than the
    reverse ("Hepatomegaly" as HP:0001894, which is Thrombocytosis). On the phenotype
    bench, trusting the id put 33 wrong terms into 13 notes."""
    onto = kb.ontology
    by_id = onto.resolve(t["id"]) if t.get("id") else None
    name = (t.get("name") or "").strip()
    if not name:
        return by_id
    by_name = onto.by_name(name)
    if by_name:
        return by_name
    # label not an exact HPO name: keep the id only if its label says much the same thing
    if by_id:
        want, have = _words(name), _words(onto.name(by_id))
        if want and len(want & have) / len(want) >= 0.6:
            return by_id
    # otherwise read the label as a tiny clinical note ("thinning of corpus callosum")
    hits = [h for h in extract_from_text(kb, name) if not h["excluded"]]
    if hits:
        return max(hits, key=lambda h: len(onto.name(h["id"])))["id"]
    return None


def validate_terms(kb: KnowledgeBase, terms: list[dict]) -> list[dict]:
    """Resolve IDs/alt IDs/labels against hp.obo; drop anything unresolvable."""
    onto = kb.ontology
    out, seen = [], set()
    for t in terms:
        tid = _resolve_term(kb, t) if onto else t.get("id")
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
    (re.compile(r"foet"), "fet"),
    (re.compile(r"ae(?=[a-z])"), "e"),
    (re.compile(r"rhoea"), "rhea"),        # diarrhoea, steatorrhoea, amenorrhoea
    (re.compile(r"our\b"), "or"),
    # generalised, hospitalisation; the 3-letter lookbehind leaves "raised" alone
    (re.compile(r"(?<=[a-z]{3})is(e|ed|es|ing|ation)\b"), r"iz\1"),
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
    r"(?:\s+(?:(?:evidence|signs?|history|features?|findings?|complaints?)\s+of|h/o|c/o))?[\s\-,;]*$"
)

# "No seizures or hypotonia": the negation covers the whole list, not just its first item.
# A negation trigger earlier in the same clause still applies when only list items and
# joiners stand between it and the term - and not when a verb or "but" intervenes
# ("no seizures but has hypotonia").
_LIST_TRIGGER_RE = re.compile(r"\b(?:no|without|denies|denied|negative for|absence of|nor)\b")
_LIST_JOINER_RE = re.compile(r"(?:,|\bor|\bnor|\band)\s*$")
_LIST_BREAK = {"but", "however", "although", "though", "except", "has", "have", "had", "having",
               "with", "shows", "showed", "show", "presents", "presented", "noted", "is", "are",
               "was", "were", "present", "seen", "revealed", "reveals", "able"}

# negation written after the term: "Seizures: absent", "visual impairment not present"
_TRAILING_NEG_RE = re.compile(
    r"^[\s:,\-]*(?:(?:is|are|was|were)\s+)?(?:absent|not\s+(?:present|seen|noted|observed|detected)|"
    r"ruled\s+out|excluded|negative|nil)\b")
_TRAILING_POS_RE = re.compile(r"^[\s:,\-]*(?:(?:is|are|was|were)\s+)?(?:present|noted|seen|observed|positive)\b")

# a term after "family history", "maternal uncle had", "sister has" describes a relative,
# not the patient, so it must not count towards the patient's phenotype. Same sentence,
# within ~120 characters before the match. "Child", "son" and "daughter" count only with a
# possessive ("her son has"): in a paediatric referral "child with ataxia" IS the patient.
_RELATIVE = (r"(?:(?:maternal|paternal|elder|older|younger|twin)\s+)*"
             r"(?:mother|father|sister|brother|sibling|parent|grandparent|grandmother|grandfather|"
             r"uncle|aunt|cousin|niece|nephew)s?")
_FAMILY_HIST_RE = re.compile(
    r"\b(?:family history|fh\b|f/h|family members?|relatives?|"
    rf"(?:(?:his|her|their|patient'?s)\s+)?{_RELATIVE}\b[^.]{{0,40}}(?:has|had|have|having|with|of|reported)|"
    r"(?:his|her|their)\s+(?:son|daughter|child(?:ren)?)\b[^.]{0,40}(?:has|had|have|having|with|of|reported))"
    r"[^.]{0,80}$",
    re.IGNORECASE,
)

# Abbreviations HPO lists as synonyms ("GDD", "FTT", "IUGR") are too short to match safely
# in lower case, so a 2-3 letter synonym only counts when written in capitals as its own
# token. These are ambiguous or ordinary words in a referral and are never used.
_ABBREV_DENY = {"all", "asd", "bo", "ra", "mi", "rad", "lft", "odd", "ddn", "cyn", "clm", "s3", "s4",
                "ome", "aob", "sua", "aea", "rbd", "pvl", "dnt"}

# common referral wording that HPO does not carry as a synonym; mapped to an HPO label
# (resolved at load time, so a renamed label is dropped rather than mis-coded)
_EXTRA_PHRASES = {
    "global delay": "Global developmental delay",
    "fits": "Seizure",
    "convulsions": "Seizure",
}
_EXTRA_ABBREVIATIONS = {
    "HSM": "Hepatosplenomegaly",
    "SNHL": "Sensorineural hearing impairment",
}

# measurements that imply a term: "HC below 3rd centile" is microcephaly even though the
# word never appears. Run on the normalised (lower-case, American-spelling) note.
_CENTILE_LOW = (r"(?:(?:<|below|less than|under)\s*(?:the\s*)?(?:3|2|1|third|second|first)(?:st|nd|rd|th)?"
                r"\s*(?:centile|percentile|%ile)|-\s*[2-9](?:\.\d+)?\s*sd)")
_CENTILE_HIGH = (r"(?:(?:>|above|more than|over)\s*(?:the\s*)?(?:97|98|99)(?:th)?\s*(?:centile|percentile|%ile)"
                 r"|\+\s*[2-9](?:\.\d+)?\s*sd)")
_HEAD = r"\b(?:hc|ofc|head circumference|head size)\b[^.;]{0,30}?"
_HEIGHT = r"\b(?:height|length|stature|ht)\b[^.;]{0,30}?"
_MEASUREMENT_RULES = [
    (re.compile(_HEAD + _CENTILE_LOW), "Microcephaly"),
    (re.compile(_HEAD + _CENTILE_HIGH), "Macrocephaly"),
    (re.compile(_HEIGHT + _CENTILE_LOW), "Short stature"),
    (re.compile(_HEIGHT + _CENTILE_HIGH), "Tall stature"),
    (re.compile(r"\b(?:cpk|ck|creatine (?:phospho)?kinase)\b[^.;]{0,15}?"
                r"(?:raised|elevated|high|increased|\d{4,}|\d{1,3},\d{3})"),
     "Elevated circulating creatine kinase activity"),
]

# "elevated sweat chloride" is often written "sweat chloride elevated"
_INVERTIBLE = {
    "elevated": ("elevated", "raised", "high", "increased"),
    "increased": ("increased", "raised", "high", "elevated"),
    "decreased": ("decreased", "reduced", "low"),
    "reduced": ("reduced", "decreased", "low"),
    "delayed": ("delayed",),
    "advanced": ("advanced",),
}

_SUFFIX = r"(?:s|es|'s|ic|al|ity|ous|ing|ed)?(?![a-z])"


def _sub_all(text: str, rules) -> str:
    for pat, rep in rules:
        text = pat.sub(rep, text)
    return text


def _normalise(text: str) -> str:
    return _sub_all(_sub_all(text, _BRITISH_TO_AMERICAN), _TERM_NORMALISE)


def _strip_articles(text: str) -> str:
    return " ".join(_ARTICLE_RE.sub(" ", text).split())


def _search_forms(name: str, all_names: frozenset[str] = frozenset()) -> list[tuple[str, bool]]:
    """The ways an HPO name can appear in prose, each paired with whether it is searched
    in the article-stripped copy of the note: as written; without articles; singular;
    without a final 'e' (gaze -> gazing); without one qualifier word.

    A qualifier-dropped form that is itself an HPO name belongs to that (less specific)
    term: "generalized clonic seizure" minus "clonic" is "generalized seizure", and a note
    saying "generalised seizures" must not be read as clonic."""
    forms = [(name, False)]
    bare = _strip_articles(name)
    if bare != name:
        forms.append((bare, True))
    if bare.endswith("s") and len(bare) > 5 and not bare.endswith("ss"):
        forms.append((bare[:-1], True))
    if bare.endswith("e") and len(bare) > 4:
        forms.append((bare[:-1], True))
    words = bare.split()
    if len(words) >= 2 and words[0] in _INVERTIBLE:
        rest = " ".join(words[1:])
        forms += [(f"{rest} {w}", True) for w in _INVERTIBLE[words[0]]]
    if len(words) >= 3:
        for i, w in enumerate(words):
            if w in _QUALIFIER_SKIP_WORDS:
                short = " ".join(words[:i] + words[i + 1:])
                if short in all_names or f"{short}s" in all_names:
                    continue
                forms.append((short, True))
                if short.endswith("e") and len(short) > 4:
                    forms.append((short[:-1], True))
    return forms


_ENTRY_CACHE: dict[int, tuple[list[tuple[str, str, bool]], frozenset[str]]] = {}


def _entries(onto) -> tuple[list[tuple[str, str, bool]], frozenset[str]]:
    """(name, hpo_id, is_abbreviation) for every name worth searching, longest first, plus
    the set of normalised names. Built once per ontology - it is ~60,000 entries."""
    key = id(onto)
    if key not in _ENTRY_CACHE:
        entries = []
        for name, tid in onto._name_index.items():
            if name in _MODIFIER_STOPWORDS:
                continue
            if len(name) < 4:
                if name.isalnum() and len(name) >= 2 and name not in _ABBREV_DENY:
                    entries.append((name.upper(), tid, True))
                continue
            entries.append((name, tid, False))
        for phrase, label in _EXTRA_PHRASES.items():
            if tid := onto.by_name(label):
                entries.append((phrase, tid, False))
        for abbr, label in _EXTRA_ABBREVIATIONS.items():
            if tid := onto.by_name(label):
                entries.append((abbr, tid, True))
        entries.sort(key=lambda e: -len(e[0]))
        all_names = frozenset(_strip_articles(_normalise(n)) for n, _, a in entries if not a)
        _ENTRY_CACHE[key] = (entries, all_names)
    return _ENTRY_CACHE[key]


def _list_negated(pre: str) -> bool:
    clause = re.split(r"[.;:]", pre)[-1]
    if not _LIST_JOINER_RE.search(clause):
        return False
    trigger = None
    for trigger in _LIST_TRIGGER_RE.finditer(clause):
        pass
    if trigger is None:
        return False
    words = re.findall(r"[a-z/'-]+", clause[trigger.end():])
    return len(words) <= 10 and not (set(words) & _LIST_BREAK)


def _context(hay: str, start: int, end: int) -> tuple[bool, bool]:
    """(negated, describes a relative) for a match at hay[start:end]."""
    pre = hay[max(0, start - 120):start]
    post = hay[end:end + 30]
    negated = (bool(_NEGATION_RE.search(hay[max(0, start - 50):start]))
               or bool(_TRAILING_NEG_RE.search(post))
               or (_list_negated(pre) and not _TRAILING_POS_RE.search(post)))
    return negated, bool(_FAMILY_HIST_RE.search(pre))


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
    # three copies of the note: normalised, normalised without articles, and the raw text
    # (case kept) for capitalised abbreviations. Each has its own character offsets.
    texts = {False: _normalise(text.lower())}
    texts[True] = _strip_articles(texts[False])
    texts["raw"] = text.lower()

    entries, all_names = _entries(onto)
    seen_ids: set[str] = set()
    # longest names go first; a shorter name inside an already-matched span is the same
    # mention ("seizure" inside "febrile seizure"), not a second finding
    spans: dict = {False: [], True: [], "raw": []}
    out = []

    def record(tid: str, key, m) -> None:
        seen_ids.add(tid)
        spans[key].append((m.start(), m.end()))
        negated, relative = _context(texts[key], m.start(), m.end())
        out.append({"id": tid, "name": onto.name(tid), "excluded": negated or relative})

    for name, tid, abbr in entries:
        if tid in seen_ids or tid in _HPO_META_BLOCKLIST:
            continue
        if abbr:
            if name.lower() not in texts["raw"]:
                continue
            m = re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"s?(?![A-Za-z0-9])", text)
            if not m:
                continue
            key = "raw"
        else:
            for form, key in _search_forms(_normalise(name), all_names):
                hay = texts[key]
                if form not in hay:   # cheap pre-check; the regex below is the real test
                    continue
                m = re.search(r"(?<![a-z])" + re.escape(form) + _SUFFIX, hay)
                if m:
                    break
            else:
                continue
        if any(s <= m.start() and m.end() <= e for s, e in spans[key]):
            continue
        record(tid, key, m)

    for rx, label in _MEASUREMENT_RULES:
        tid = onto.by_name(label)
        m = rx.search(texts[False]) if tid and tid not in seen_ids else None
        if m:
            record(tid, False, m)
    # "generalised seizures ... his sister has epilepsy": the sister's mention can mark
    # Seizure excluded while the patient's own (more specific) seizure term is present.
    # A term the patient has, directly or through a more specific term, is not excluded.
    present = {h["id"] for h in out if not h["excluded"]}
    has = set().union(*(onto.ancestors(t) for t in present)) if present else set()
    return [h for h in out if not (h["excluded"] and h["id"] in has)]
