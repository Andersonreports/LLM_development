"""Score phenotype extraction on the synthetic notes in bench/phenotype_cases.py.

    python -m bench.run_phenotype            # deterministic extractor only (no AI, fast)
    python -m bench.run_phenotype --ai       # also the AI extractor and the merged result
    python -m bench.run_phenotype -v         # list every miss and false term

Needs the full HPO files (resources/hpo). For each note it counts:
  hit      an expected present term was extracted as present (or an accepted alternative)
  missed   an expected present term was not extracted
  wrong    a term was extracted as present that the note does not support
  neg ok   an expected excluded term was extracted as excluded
  neg bad  an expected excluded term came out as present (worse than missing it)

A "wrong" term is only counted when it is not an ancestor or descendant of an expected
term: "Generalized-onset seizure" for a note saying "generalised seizures" is not wrong
just because the expected list says "Seizure".
"""
from __future__ import annotations

import argparse

from wes.phenotype import extract_from_text, validate_terms
from wes.resources import kb

from .phenotype_cases import CASES, HELD_OUT


def _resolve(onto, names):
    ids, bad = [], []
    for n in names:
        tid = onto.by_name(n)
        (ids if tid else bad).append(tid or n)
    return ids, bad


def score(onto, case, got: list[dict]) -> dict:
    present_ids, bad1 = _resolve(onto, case["present"])
    excluded_ids, bad2 = _resolve(onto, case["excluded"])
    alts = {onto.by_name(k): {onto.by_name(a) for a in v} - {None}
            for k, v in case.get("alternatives", {}).items()}
    got_present = {t["id"] for t in got if not t["excluded"]}
    got_excluded = {t["id"] for t in got if t["excluded"]}

    def found(tid, pool):
        # the expected term, an accepted alternative, or a more specific child of either
        options = {tid} | alts.get(tid, set())
        return any(options & onto.ancestors(g) for g in pool)

    hits = [t for t in present_ids if found(t, got_present)]
    missed = [t for t in present_ids if t not in hits]
    related = set(present_ids) | {a for s in alts.values() for a in s}
    related_anc = set().union(*(onto.ancestors(t) for t in related)) if related else set()
    wrong = [g for g in got_present
             if g not in related_anc and not (onto.ancestors(g) & related)
             and not any(g in onto.ancestors(e) or e in onto.ancestors(g) for e in excluded_ids)]
    neg_ok = [t for t in excluded_ids if found(t, got_excluded)]
    neg_bad = [t for t in excluded_ids if found(t, got_present) and t not in neg_ok]
    return {"hit": hits, "missed": missed, "wrong": wrong, "neg_ok": neg_ok, "neg_bad": neg_bad,
            "neg_total": len(excluded_ids), "unresolved": bad1 + bad2}


def run(use_ai: bool, verbose: bool, held_out: bool = False) -> None:
    k = kb()
    onto = k.ontology
    if not onto:
        raise SystemExit("hp.obo not found - set WES_RESOURCES_DIR to the full resources folder")
    modes = ["rules"] + (["ai", "merged"] if use_ai else [])
    totals = {m: {"hit": 0, "exp": 0, "wrong": 0, "neg_ok": 0, "neg": 0, "neg_bad": 0} for m in modes}
    if use_ai:
        from wes import ai

    for case in (HELD_OUT if held_out else CASES):
        outputs = {"rules": extract_from_text(k, case["text"])}
        if use_ai:
            try:
                out, _ = ai.extract_phenotype(case["text"], None, None, [])
                ai_terms = [t.model_dump() for t in out.hpo_terms]
            except ai.AIError as e:
                print(f"  AI failed on {case['name']}: {e}")
                ai_terms = []
            outputs["ai"] = validate_terms(k, ai_terms)
            outputs["merged"] = validate_terms(k, outputs["rules"] + ai_terms)
        line = f"{case['name']:<30}"
        for m in modes:
            s = score(onto, case, outputs[m])
            if s["unresolved"]:
                print(f"  !! unresolved expected terms in {case['name']}: {s['unresolved']}")
            t = totals[m]
            t["hit"] += len(s["hit"]); t["exp"] += len(s["hit"]) + len(s["missed"])
            t["wrong"] += len(s["wrong"]); t["neg_ok"] += len(s["neg_ok"])
            t["neg"] += s["neg_total"]; t["neg_bad"] += len(s["neg_bad"])
            line += (f"  {m}: {len(s['hit'])}/{len(s['hit']) + len(s['missed'])}"
                     f" wrong {len(s['wrong'])} neg {len(s['neg_ok'])}/{s['neg_total']}"
                     + (f" NEGBAD {len(s['neg_bad'])}" if s["neg_bad"] else ""))
            if verbose:
                for label, ids in (("missed", s["missed"]), ("wrong", s["wrong"]), ("neg_bad", s["neg_bad"])):
                    if ids:
                        line += f"\n      {m} {label}: " + ", ".join(f"{i} {onto.name(i)}" for i in ids)
        print(line)

    print()
    for m, t in totals.items():
        print(f"{m:<7} found {t['hit']}/{t['exp']} present terms, {t['wrong']} wrong terms, "
              f"{t['neg_ok']}/{t['neg']} negations, {t['neg_bad']} negated terms coded as present")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai", action="store_true", help="also score the AI extractor and the merged result")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--held-out", action="store_true", help="score the held-out notes instead")
    a = ap.parse_args()
    run(a.ai, a.verbose, a.held_out)


if __name__ == "__main__":
    main()
