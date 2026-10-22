"""Claude-powered steps: clinical-history -> HPO extraction, and case-level
variant interpretation (ranking, ACMG refinement, result flag, report text).

Only de-identified data is sent: age, sex, clinical text with the patient's
name/ID scrubbed, HPO terms, and variant-level evidence.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, ConfigDict

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


# ------------------------------------------------------------------ client
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=3, timeout=900)


def available() -> bool:
    if not settings.ai_enabled:
        return False
    import os
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _call(system: str, user: str, schema: type[BaseModel], effort: str, max_tokens: int = 32000):
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
             "model": msg.model, "request_id": getattr(msg, "_request_id", None)}
    return parsed, usage


def scrub(text: str, identifiers: list[str]) -> str:
    out = text or ""
    for ident in identifiers:
        for tok in [ident] + ident.split():
            tok = tok.strip()
            if len(tok) >= 3:
                out = re.sub(re.escape(tok), "[REDACTED]", out, flags=re.IGNORECASE)
    return out


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
