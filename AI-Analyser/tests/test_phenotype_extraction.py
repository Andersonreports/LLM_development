"""Phenotype extraction from the clinical history: the rules that the phenotype bench
(bench/run_phenotype.py, full HPO) showed were needed, pinned on the small test ontology."""
from __future__ import annotations

from wes.phenotype import _normalise, extract_from_text, validate_terms
from wes.resources import kb

SEIZURE, GDD, MYOPATHY = "HP:0001250", "HP:0001263", "HP:0003198"


def _hits(text):
    return {h["id"]: h["excluded"] for h in extract_from_text(kb(), text)}


# ---------------------------------------------------------------- AI id vs label
def test_label_wins_when_the_ai_writes_a_wrong_id():
    # the right label with another term's number - the common small-model mistake
    out = validate_terms(kb(), [{"id": MYOPATHY, "name": "Seizure", "excluded": False}])
    assert [t["id"] for t in out] == [SEIZURE]


def test_id_kept_when_label_is_a_close_paraphrase():
    out = validate_terms(kb(), [{"id": GDD, "name": "Developmental delay, global", "excluded": False}])
    assert [t["id"] for t in out] == [GDD]


def test_unrecognisable_label_with_unrelated_id_is_dropped():
    out = validate_terms(kb(), [{"id": MYOPATHY, "name": "Retinal coloboma", "excluded": False}])
    assert out == []


def test_id_only_terms_still_resolve():
    assert [t["id"] for t in validate_terms(kb(), [{"id": SEIZURE}])] == [SEIZURE]


# ---------------------------------------------------------------- negation
def test_negation_covers_a_list():
    hits = _hits("No seizures or myopathy.")
    assert hits[SEIZURE] is True and hits[MYOPATHY] is True


def test_negation_stops_at_but():
    hits = _hits("No seizures but has myopathy.")
    assert hits[SEIZURE] is True and hits[MYOPATHY] is False


def test_negation_written_after_the_term():
    assert _hits("Myopathy: absent. Seizures present.") == {MYOPATHY: True, SEIZURE: False}
    assert _hits("Myopathy not present.")[MYOPATHY] is True


def test_no_h_o_is_a_negation():
    assert _hits("No h/o seizures.")[SEIZURE] is True


# ---------------------------------------------------------------- relatives
def test_child_with_is_the_patient_not_a_relative():
    assert _hits("Child with myopathy.")[MYOPATHY] is False


def test_maternal_uncle_is_a_relative():
    assert _hits("Global developmental delay. Maternal uncle has myopathy.") == {GDD: False, MYOPATHY: True}


def test_her_son_is_a_relative():
    assert _hits("Her son has seizures.")[SEIZURE] is True


# ---------------------------------------------------------------- wording
def test_colloquial_phrases_map_to_hpo():
    assert _hits("Frequent fits and global delay.") == {SEIZURE: False, GDD: False}


def test_british_spelling_normalisation():
    assert _normalise("steatorrhoea") == "steatorrhea"      # used to become "steatorrheaa"
    assert _normalise("diarrhoea") == "diarrhea"
    assert _normalise("generalised seizures") == "generalized seizures"
    assert _normalise("foetal") == "fetal"
    assert _normalise("ammonia raised") == "ammonia raised"   # not "raized"
