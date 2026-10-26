"""AI steps: clinical-history -> HPO extraction, and case-level variant
interpretation (ranking, ACMG refinement, result flag, report text).

Two interchangeable backends, selected with WES_AI_BACKEND:
  "local"  (default) a model served by Ollama on this machine - no case data
           leaves the network
  "claude" the Anthropic API - de-identified data only (age, sex, clinical text
           with names/IDs scrubbed, HPO terms, variant evidence), and it has to
           be enabled deliberately

Both receive the same prompts and must return the same validated schema, so the
rest of the pipeline never knows which model answered. Whatever the backend
proposes is re-scored in code (see pipeline._run_ai) and shown to the reviewer.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .config import settings

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AIError(RuntimeError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ schemas
class HpoTermOut(_Strict):
    id: str
    name: str
    excluded: bool


class PhenotypeOut(_Strict):
    summary: str
    hpo_terms: list[HpoTermOut]
    age_of_onset: Optional[str]
    consanguinity: Optional[bool]
    family_history: Optional[str]
    suspected_conditions: list[str]
    suspected_genes: list[str]


class Criterion(_Strict):
    code: str
    strength: Literal["VeryStrong", "Strong", "Moderate", "Supporting"]
    reason: str


class Finding(_Strict):
    variant_key: str
    report_section: Literal["primary", "vus", "secondary", "carrier", "not_reported"]
    classification: Literal["Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"]
    criteria_added: list[Criterion]
    criteria_removed: list[str]
    disease: str
    disease_inheritance: str
    phenotype_fit: Literal["strong", "partial", "weak", "not_applicable"]
    interpretation: str


class InterpretationOut(_Strict):
    result_flag: Literal["POSITIVE", "INCONCLUSIVE", "NEGATIVE", "CARRIER"]
    flag_rationale: str
    summary_line: str
    findings: list[Finding]
    excluded_candidates: list[str]
    clinical_interpretation: str
    recommendations: list[str]
    limitations: list[str]
    reviewer_attention: list[str]


# ------------------------------------------------------------------ backends
def backend() -> str:
    return "claude" if settings.ai_backend == "claude" else "local"


def describe_backend() -> str:
    """One line for the /system page and the audit log."""
    return (f"Anthropic API ({settings.anthropic_model})" if backend() == "claude"
            else f"local model {settings.ollama_model} via Ollama")


def available() -> bool:
    """True when the configured model can actually be reached right now. When it
    cannot, the pipeline still completes and produces the rule-based draft."""
    if not settings.ai_enabled:
        return False
    if backend() == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    try:
        tags = httpx.get(f"{settings.ollama_url}/api/tags", timeout=3.0).json()
    except Exception:  # noqa: BLE001 - service down, wrong URL, bad response
        return False
    names = {m.get("name", "") for m in tags.get("models", [])}
    want = settings.ollama_model
    # a tag without a version matches "name:latest"
    return want in names or f"{want}:latest" in names or any(n.split(":")[0] == want for n in names)


def _call(system: str, user: str, schema: type[BaseModel], effort: str, max_tokens: int = 32000):
    if backend() == "local":
        return _call_ollama(system, user, schema, max_tokens)
    return _call_claude(system, user, schema, effort, max_tokens)


def _call_ollama(system: str, user: str, schema: type[BaseModel], max_tokens: int):
    """Ollama's /api/chat with `format` set to the JSON Schema of the expected model,
    so the server constrains generation to valid JSON. Temperature 0: the same case
    must produce the same draft."""
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": settings.ollama_think,
        "format": schema.model_json_schema(),
        "options": {"temperature": 0, "num_ctx": settings.ollama_num_ctx, "num_predict": max_tokens},
    }
    try:
        resp = httpx.post(f"{settings.ollama_url}/api/chat", json=payload,
                          timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0))
    except httpx.ConnectError as e:
        raise AIError(f"Local model server not reachable at {settings.ollama_url} - is Ollama running?") from e
    except httpx.TimeoutException as e:
        raise AIError(f"Local model timed out after {settings.ollama_timeout}s") from e
    if resp.status_code == 404:
        raise AIError(f"Model '{settings.ollama_model}' is not installed - run: ollama pull {settings.ollama_model}")
    if resp.status_code != 200:
        raise AIError(f"Local model server error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if data.get("done_reason") == "length":
        raise AIError("Local model response truncated - reduce WES_AI_MAX_CANDIDATES or raise WES_OLLAMA_NUM_CTX")
    content = (data.get("message") or {}).get("content") or ""
    if not content.strip():
        raise AIError("Local model returned an empty response")
    try:
        parsed = schema.model_validate_json(content)
    except ValidationError as e:
        # a small model can emit valid JSON that does not match the schema
        raise AIError(f"Local model output did not match the expected format: {str(e)[:400]}") from e
    except ValueError as e:
        raise AIError(f"Local model did not return valid JSON: {str(e)[:300]}") from e
    usage = {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count"),
             "model": data.get("model", settings.ollama_model), "backend": "local", "request_id": None,
             "duration_s": round((data.get("total_duration") or 0) / 1e9, 1)}
    return parsed, usage


def _call_claude(system: str, user: str, schema: type[BaseModel], effort: str, max_tokens: int):
    import anthropic  # optional dependency: only needed for this backend

    def _client() -> "anthropic.Anthropic":
        return anthropic.Anthropic(max_retries=3, timeout=900)

    try:
        with _client().beta.messages.stream(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            output_format=schema,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise AIError("Claude API authentication failed - check ANTHROPIC_API_KEY") from e
    except anthropic.RateLimitError as e:
        raise AIError("Claude API rate limit reached - retry the AI step later") from e
    except anthropic.APIStatusError as e:
        raise AIError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AIError("Could not reach the Claude API") from e

    if msg.stop_reason == "refusal":
        cat = getattr(msg.stop_details, "category", None) if msg.stop_details else None
        raise AIError(f"Claude declined the request (category: {cat})")
    if msg.stop_reason == "max_tokens":
        raise AIError("Claude response truncated (max_tokens) - reduce WES_AI_MAX_CANDIDATES")
    parsed = msg.parsed_output
    if parsed is None:
        raise AIError("Claude returned no structured output")
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
             "model": msg.model, "backend": "claude", "request_id": getattr(msg, "_request_id", None)}
    return parsed, usage


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d -]{8,}\d(?!\w)")
# words of hospital names / titles that must not be redacted on their own
_GENERIC_WORDS = {
    "hospital", "hospitals", "clinic", "clinics", "medical", "centre", "center", "college", "institute",
    "children's", "childrens", "child", "health", "care", "research", "sciences", "general", "city",
    "the", "and", "for", "of", "mrs", "miss", "prof", "baby", "master", "proband", "father", "mother",
    "partner1", "partner2",
}
_DATE_RE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")


def scrub(text: str, identifiers: list[str]) -> str:
    """Remove identifiers before text is sent to the API. Names are matched as whole
    words only (a patient called "Ram" must not turn "parameter" into "pa[REDACTED]eter");
    e-mail addresses, phone numbers and calendar dates are removed regardless."""
    out = text or ""
    toks = {t.strip(" ,.") for ident in identifiers if ident for t in [ident, *ident.split()]}
    toks = {t for t in toks if len(t) >= 3 and t.lower() not in _GENERIC_WORDS}
    # longest first, so a full name is replaced before its parts
    for tok in sorted(toks, key=len, reverse=True):
        out = re.sub(r"(?<!\w)" + re.escape(tok) + r"(?!\w)", "[REDACTED]", out, flags=re.IGNORECASE)
    out = _EMAIL_RE.sub("[REDACTED]", out)
    out = _PHONE_RE.sub("[REDACTED]", out)
    return _DATE_RE.sub("[DATE]", out)


# ------------------------------------------------------------------ prompts
PHENO_SYSTEM = """You are a clinical geneticist preparing a case for exome analysis.
Convert the referral clinical history into Human Phenotype Ontology (HPO) terms.

- Use precise, current HPO IDs and their exact labels (e.g. HP:0001250 "Seizure"). Prefer the most specific term the text supports; do not infer features that are not stated.
- Mark features the text explicitly says are absent/normal with excluded=true.
- Do not code inheritance-mode or onset terms as phenotypes; report onset in age_of_onset.
- consanguinity: true/false only if stated; otherwise null.
- suspected_conditions / suspected_genes: only what the referring clinician mentions or what is strongly implied by a pathognomonic combination; leave empty otherwise.
- summary: 2-3 sentence neutral clinical summary in third person suitable for a lab report (no names)."""


INTERP_SYSTEM = """You are a senior clinical molecular geneticist (ACMG/AMP certified) interpreting exome sequencing for a diagnostic laboratory. Your output is a DRAFT that a qualified scientist will review and sign; be accurate and conservative.

You receive: case type (single proband, trio, or couple carrier screening), de-identified clinical data with HPO terms, QC metrics, and pre-filtered candidate variants. Each candidate carries annotation evidence, genotype/segregation data, gene-disease associations, phenotype-similarity score and automatically applied ACMG criteria (Tavtigian points: VeryStrong 8, Strong 4, Moderate 2, Supporting 1; P >= 10, LP 6-9, VUS 0-5, LB -1 to -6, B <= -7).

How to work:
1. Decide which variant(s), if any, explain the phenotype. Weigh gene-disease validity, inheritance fit (zygosity, segregation, sex), phenotype overlap, and variant evidence together.
2. Refine ACMG criteria only where you have a concrete basis:
   - PP4 when the phenotype is highly specific for the gene's disease.
   - PS1/PM5/PM1/PS4/PP1/BS2/BS4 only from well-established knowledge of this exact variant/residue/domain; state the basis in the reason and prefix it "Knowledge-based:" so the reviewer can verify it.
   - criteria_removed for auto criteria you judge misapplied (e.g. PVS1 where LoF is not the mechanism, PP3 contradicted by known data).
   Never invent population frequencies, ClinVar entries, functional studies, or publications. If unsure, do not add the criterion.
3. classification must be consistent with the auto criteria plus your additions minus removals under the points system.
4. Choose result_flag:
   - single/trio: POSITIVE = P/LP variant(s) whose genotype fits the disease inheritance (biallelic for AR, heterozygous/de novo for AD, hemizygous for XLR males) and whose disease fits the phenotype. INCONCLUSIVE = VUS in a strong candidate gene, a single P/LP allele in an AR gene matching the phenotype, P/LP with only partial phenotype overlap, or a variant needing segregation. NEGATIVE = nothing reportable for the indication. CARRIER is not used for single/trio.
   - couple: POSITIVE = both partners carry P/LP variants in the same AR gene, or the female partner carries a P/LP X-linked recessive variant (at-risk couple). CARRIER = P/LP carrier variant(s) without shared-gene risk. NEGATIVE = none.
5. report_section: "primary" (explains the phenotype, or at-risk couple finding), "vus" (VUS reported for the indication), "carrier" (carrier finding), "secondary" (ACMG SF), "not_reported". Include in findings only variants with a section other than not_reported; list other variant_keys you deliberately dismissed in excluded_candidates.
6. interpretation (per finding): 3-6 sentences in formal report language: the variant, its predicted effect, population frequency, ClinVar/evidence, gene-disease association with inheritance, and how it relates to this patient.
7. clinical_interpretation: report-ready paragraph(s) summarising the result for the referring clinician. recommendations: concrete actions (genetic counselling, orthogonal confirmation, parental segregation, cascade testing, specialist referral, reanalysis). limitations: case-specific (coverage, CNVs/repeat expansions/deep-intronic not assessed, QC caveats).
8. reviewer_attention: anything the reviewing scientist must check (QC warnings, borderline calls, knowledge-based criteria, possible mosaicism, sex mismatch).

Use only the variant_key values provided. Write in British/Indian clinical English with HGVS nomenclature as given."""


def extract_phenotype(clinical_text: str, age: str | None, sex: str | None,
                      identifiers: list[str]) -> tuple[PhenotypeOut, dict]:
    text = scrub(clinical_text, identifiers)
    user = (f"Patient age: {age or 'not stated'}\nSex: {sex or 'not stated'}\n\n"
            f"Clinical history:\n<clinical_history>\n{text}\n</clinical_history>")
    return _call(PHENO_SYSTEM, user, PhenotypeOut, effort="medium", max_tokens=16000)


_VARIANT_FIELDS = [
    "key", "gene", "transcript", "hgvsc", "hgvsp", "consequence_str", "exon", "intron", "zygosity",
    "models", "model_notes", "comphet_partners", "origin", "pop_af", "gnomad_exome_af", "gnomad_genome_af",
    "max_af_pops", "clinvar_sig", "clinvar_stars", "clinvar_disease", "am_score", "am_class", "revel",
    "spliceai", "cadd", "sift", "polyphen", "phenotype_score", "inheritance_match", "rsid",
]


def _compact_variant(v: dict) -> dict:
    d = {k: v.get(k) for k in _VARIANT_FIELDS if v.get(k) not in (None, [], "", {})}
    d["genotype_quality"] = {r: {"dp": g.get("dp"), "gq": g.get("gq"), "ab": round(g["ab"], 2) if g.get("ab") is not None else None}
                             for r, g in v.get("genotypes", {}).items()}
    gi = v.get("gene_info") or {}
    d["gene"] = {"symbol": v["gene"], "diseases": gi.get("diseases", [])[:5],
                 "loeuf": gi.get("loeuf"), "clingen_hi": gi.get("clingen_hi")}
    d["phenotype_matches"] = [f"{m['patient_name']} ~ {m['match_name']}" for m in (v.get("phenotype_matches") or [])[:6]]
    d["acmg_auto"] = {"criteria": [f"{c['code']}_{c['strength']}: {c['reason']}" for c in v["acmg"]["criteria"]],
                      "points": v["acmg"]["points"], "classification": v["acmg"]["classification"]}
    return d


def interpret(context: dict) -> tuple[InterpretationOut, dict]:
    payload = {
        "case_type": context["case_type"],
        "clinical": context["clinical"],
        "qc": context["qc"],
        "rule_based_flag": context["rule_flag"],
        "candidates": [_compact_variant(v) for v in context["candidates"]],
    }
    if context.get("couple"):
        payload["couple_carrier_summary"] = {
            "shared_genes": context["couple"]["shared_genes"],
            "at_risk": [{k: r[k] for k in ("gene", "mode", "partner1", "partner2")} for r in context["couple"]["at_risk"]],
        }
    if context.get("secondary"):
        payload["secondary_findings_candidates"] = [_compact_variant(v) for v in context["secondary"]]
    if context.get("carrier"):
        payload["carrier_candidates"] = [_compact_variant(v) for v in context["carrier"]]
    user = ("Interpret this exome case and return the structured result.\n\n<case>\n"
            + json.dumps(payload, indent=1, default=str) + "\n</case>")
    return _call(INTERP_SYSTEM, user, InterpretationOut, effort=settings.ai_effort)


def check() -> dict:
    """Backend reachability, for the /system page."""
    info = {"backend": backend(), "description": describe_backend(), "enabled": settings.ai_enabled,
            "available": available()}
    if backend() == "local":
        info["url"] = settings.ollama_url
        info["hint"] = f"run: ollama pull {settings.ollama_model}"
    return info
