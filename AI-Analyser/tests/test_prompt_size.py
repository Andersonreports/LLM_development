"""The prompt must fit the model's context window.

A real case overflowed it: Ollama silently truncated the start of the prompt and the model
answered from a case it had only half received, taking 10 minutes to fail. These tests pin
the two defences - dropping weak candidates to fit, and leaving out phenotype matches that
carry no information."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from wes import ai
from wes.config import settings


def _variant(key, gene="GENE1", matches=(), diseases=3):
    return {
        "key": key, "gene": gene, "hgvsc": "c.100A>G", "hgvsp": "p.Lys34Glu",
        "consequence_str": "missense variant", "zygosity": {"proband": "het"},
        "models": ["AD"], "pop_af": 0.0001, "phenotype_score": 0.4,
        "model_notes": ["Heterozygous, absent in both parents (de novo)"] * 2,
        "gene_info": {"diseases": [{"name": f"Some long disease name number {i}",
                                    "inheritance": ["AD"]} for i in range(diseases)],
                      "loeuf": 0.3, "clingen_hi": "3"},
        "phenotype_matches": list(matches),
        "acmg": {"criteria": [{"code": "PM2", "strength": "Supporting",
                               "reason": "Absent from gnomAD exomes and genomes"}],
                 "points": 1, "classification": "Uncertain significance"},
        "genotypes": {"proband": {"dp": 40, "gq": 99, "ab": 0.5}},
    }


def _match(patient, match, ic):
    return {"patient": "HP:1", "patient_name": patient, "match": "HP:2",
            "match_name": match, "ic": ic, "exact": False}


def test_uninformative_phenotype_matches_are_left_out():
    """"Anaemia ~ Phenotypic abnormality" (ic 0.0) is not evidence - it is noise that
    reads to the model as support."""
    v = _variant("k", matches=[
        _match("Seizure", "Seizure", 1.28),
        _match("Anemia", "Phenotypic abnormality", 0.0),
        _match("Facial palsy", "Abnormality of the face", 0.51),
    ])
    out = ai._compact_variant(v)["phenotype_matches"]
    assert "Seizure ~ Seizure" in out
    assert "Facial palsy ~ Abnormality of the face" in out
    assert not any("Phenotypic abnormality" in m for m in out)


def test_at_most_three_matches_and_three_diseases_are_sent():
    v = _variant("k", matches=[_match(f"Term {i}", f"Match {i}", 1.0) for i in range(8)], diseases=9)
    c = ai._compact_variant(v)
    assert len(c["phenotype_matches"]) == 3
    assert len(c["gene"]["diseases"]) == 3


def _payload():
    return {"case_type": "single", "clinical": {"summary": "x" * 400}, "qc": {},
            "rule_based_flag": {"flag": "NEGATIVE"}, "candidates": []}


def test_candidates_are_dropped_until_the_prompt_fits():
    many = [_variant(f"k{i}", gene=f"GENE{i}",
                     matches=[_match(f"Term {j}", f"Match {j}", 1.0) for j in range(3)])
            for i in range(200)]
    payload = _payload()
    kept = ai._fit_candidates(payload, many, system_chars=7000)
    assert 5 <= len(kept) < len(many)
    budget = int(settings.ollama_num_ctx * ai.CHARS_PER_TOKEN) - 7000 - ai.MAX_TOKENS_INTERPRET * ai.CHARS_PER_TOKEN
    assert len(json.dumps(payload, default=str)) <= budget


def test_a_small_case_is_sent_whole():
    few = [_variant(f"k{i}") for i in range(8)]
    payload = _payload()
    assert ai._fit_candidates(payload, few, system_chars=7000) == few
    assert len(payload["candidates"]) == 8


def test_at_least_five_candidates_survive_even_a_tiny_window(monkeypatch):
    """Better to send a few variants than none: the reviewer still gets a draft."""
    monkeypatch.setattr(ai, "settings", replace(settings, ollama_num_ctx=2048))
    many = [_variant(f"k{i}", gene=f"GENE{i}") for i in range(50)]
    assert len(ai._fit_candidates(_payload(), many, system_chars=7000)) == 5


@pytest.mark.parametrize("name,cap", [("MAX_TOKENS_PHENOTYPE", 3000), ("MAX_TOKENS_INTERPRET", 8000)])
def test_output_caps_stay_modest(name, cap):
    """Generous caps let a looping model burn ten minutes before failing."""
    assert getattr(ai, name) == cap


def test_the_character_budget_is_conservative():
    """Measured, not guessed: a prompt budgeted at 2.5 chars/token still overflowed."""
    assert ai.CHARS_PER_TOKEN <= 2.0


def test_a_full_candidate_list_fits_the_real_context_window():
    """40 realistic candidates must leave room for the answer in a 32k window."""
    many = [_variant(f"k{i}", gene=f"GENE{i}",
                     matches=[_match(f"Term {j}", f"Match {j}", 1.0) for j in range(3)])
            for i in range(40)]
    payload = _payload()
    ai._fit_candidates(payload, many, system_chars=7000)
    input_tokens = len(json.dumps(payload, default=str)) / ai.CHARS_PER_TOKEN + 7000 / 4
    assert input_tokens + ai.MAX_TOKENS_INTERPRET < settings.ollama_num_ctx
