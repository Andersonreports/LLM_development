"""Every list the model fills must be bounded.

An unbounded list gives a small model no reason to stop. On a real case Qwen 3.5 4B
emitted HPO terms until it exhausted the entire token budget, and the whole phenotype
step failed. The caps appear as "maxItems" in the JSON Schema sent to Ollama, so they
constrain generation rather than merely requesting good behaviour."""
from __future__ import annotations

import pytest

from wes.ai import InterpretationOut, PhenotypeOut

SCHEMAS = {"PhenotypeOut": PhenotypeOut, "InterpretationOut": InterpretationOut}


def _array_fields(schema: dict, defs: dict, path: str = ""):
    """Yield (path, subschema) for every array in the schema, following $ref."""
    for name, prop in (schema.get("properties") or {}).items():
        here = f"{path}.{name}" if path else name
        if "$ref" in prop:
            yield from _array_fields(defs[prop["$ref"].rsplit("/", 1)[-1]], defs, here)
        elif prop.get("type") == "array":
            yield here, prop
            items = prop.get("items") or {}
            if "$ref" in items:
                yield from _array_fields(defs[items["$ref"].rsplit("/", 1)[-1]], defs, here + "[]")


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_every_list_is_capped(name):
    schema = SCHEMAS[name].model_json_schema()
    defs = schema.get("$defs", {})
    arrays = dict(_array_fields(schema, defs))
    assert arrays, f"{name} has no arrays - test is not checking anything"
    uncapped = [p for p, s in arrays.items() if "maxItems" not in s]
    assert not uncapped, f"unbounded list(s) in {name}: {uncapped}"


def test_caps_are_large_enough_for_real_cases():
    """A cap that is too tight would silently cut a legitimate answer short."""
    p = PhenotypeOut.model_json_schema()["properties"]
    assert p["hpo_terms"]["maxItems"] >= 30          # complex syndromes carry many terms
    i = InterpretationOut.model_json_schema()["properties"]
    assert i["findings"]["maxItems"] >= 20
    assert i["excluded_candidates"]["maxItems"] >= 40  # must hold the dismissed candidates


def test_values_within_the_caps_still_validate():
    PhenotypeOut.model_validate({
        "summary": "x", "age_of_onset": None, "consanguinity": None, "family_history": None,
        "hpo_terms": [{"id": f"HP:{i:07d}", "name": "t", "excluded": False} for i in range(40)],
        "suspected_conditions": ["c"] * 10, "suspected_genes": ["G"] * 20,
    })


def test_values_over_the_caps_are_rejected():
    with pytest.raises(Exception):
        PhenotypeOut.model_validate({
            "summary": "x", "age_of_onset": None, "consanguinity": None, "family_history": None,
            "hpo_terms": [{"id": f"HP:{i:07d}", "name": "t", "excluded": False} for i in range(41)],
            "suspected_conditions": [], "suspected_genes": [],
        })
