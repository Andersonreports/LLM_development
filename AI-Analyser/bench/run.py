"""Score an interpretation model on the synthetic cases in bench/cases.py.

    python -m bench.run                           # all cases, current .env settings
    python -m bench.run --cases dominant_de_novo  # one case
    python -m bench.run --label "strict+think" --out bench/results

Checks, per case:
  flag          the result flag matches the expected one
  primary       exactly the expected variants are in the "primary" report section
  no_removal    the model did not ask to delete automatically derived ACMG criteria
  consistent    its stated classification matches the ACMG points of its own criteria
  grounded      no forbidden claim (e.g. calling an autosomal gene X-linked)

Every run writes a JSON file with the full output, so drafts can be compared later.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from wes import acmg, ai
from wes.config import settings

from .cases import BY_NAME, CASES

CHECKS = ["flag", "primary", "no_removal", "consistent", "grounded"]


# A forbidden phrase only counts when the model ASSERTS it. "precludes a definitive
# diagnosis" is the correct thing to say about a VUS, so a negation shortly before the
# match clears it. Same approach as _NEGATION_RE in wes/phenotype.py.
_NEG = (r"\bnot\b|\bno\b|\bnor\b|\bcannot\b|\bcan ?not\b|\bcould not\b|\bdoes not\b|\bdo not\b|"
        r"\bdid not\b|\bis not\b|\bare not\b|\bwas not\b|\bwere not\b|\bn't\b|"
        r"\bprecludes?\b|\bprevents?\b|\bexcludes?\b|\bwithout\b|\black(?:s|ing)?\b|\bunlike\b|"
        r"\brather than\b|\binstead of\b|\brules? out\b|\bunable\b|\binsufficient\b")
# "...do not establish a definitive diagnosis" - the negation sits before the phrase
_NEGATED_BEFORE = re.compile(rf"(?:{_NEG})[^.;:]{{0,70}}$", re.I)
# "a definitive diagnosis cannot be established" - and sometimes after it
_NEGATED_AFTER = re.compile(rf"^[^.;:]{{0,50}}?(?:{_NEG})", re.I)


def forbidden_hits(text: str, phrases: list[str]) -> list[str]:
    """Phrases the model must not ASSERT. Clinical writing negates these constantly
    ("does not establish a definitive diagnosis"), and the negation can sit on either
    side of the phrase, so both directions are checked within the sentence."""
    hits = []
    for p in phrases:
        for m in re.finditer(re.escape(p), text):
            start = max(text.rfind(".", 0, m.start()), text.rfind(";", 0, m.start())) + 1
            end = min([i for i in (text.find(".", m.end()), text.find(";", m.end())) if i >= 0]
                      or [len(text)])
            before = text[max(start, m.start() - 90):m.start()]
            after = text[m.end():end]
            if not (_NEGATED_BEFORE.search(before) or _NEGATED_AFTER.search(after)):
                hits.append(p)
                break
    return hits


def _all_text(d: dict) -> str:
    parts = [d["flag_rationale"], d["summary_line"], d["clinical_interpretation"]]
    parts += d["recommendations"] + d["limitations"] + d["reviewer_attention"]
    parts += [f["interpretation"] + " " + f["disease"] + " " + f["disease_inheritance"] for f in d["findings"]]
    return " ".join(parts).lower()


def score(case: dict, out: dict) -> dict:
    want = case["expect"]
    pool = {v["key"]: v for v in case["context"]["candidates"]}
    primary = {f["variant_key"] for f in out["findings"] if f["report_section"] == "primary"}
    removed = {c for f in out["findings"] for c in f["criteria_removed"]}
    hits = forbidden_hits(_all_text(out), want["forbidden"])

    inconsistent = []
    for f in out["findings"]:
        v = pool.get(f["variant_key"])
        if not v:
            inconsistent.append(f"unknown key {f['variant_key']}")
            continue
        crit = [c for c in v["acmg"]["criteria"] if c["code"] not in set(f["criteria_removed"])]
        crit += [acmg._crit(c["code"], c["strength"], c["reason"], source="ai") for c in f["criteria_added"]]
        _, cls = acmg.classify_points(crit)
        if cls != f["classification"]:
            inconsistent.append(f"{v['gene']}: said {f['classification']}, criteria give {cls}")

    return {
        "flag": out["result_flag"] == want["flag"],
        "primary": primary == set(want["primary"]),
        "no_removal": not removed,
        "consistent": not inconsistent,
        "grounded": not hits,
        "_detail": {"got_flag": out["result_flag"], "got_primary": sorted(primary),
                    "removed": sorted(removed), "inconsistent": inconsistent, "forbidden_hits": hits},
    }


def run(names: list[str], label: str, out_dir: Path) -> list[dict]:
    cases = [BY_NAME[n] for n in names] if names else CASES
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    config = {"backend": ai.backend(), "model": settings.ollama_model if ai.backend() == "local"
              else settings.anthropic_model, "think": settings.ollama_think,
              "strict_prompt": settings.ai_strict_prompt, "label": label}
    print(f"{label or 'run'}: {ai.describe_backend()} | think={settings.ollama_think} "
          f"| strict_prompt={settings.ai_strict_prompt}\n")

    results = []
    for case in cases:
        t0 = time.time()
        try:
            parsed, usage = ai.interpret(case["context"])
            data = parsed.model_dump()
            res = score(case, data)
            err = None
        except ai.AIError as e:
            data, usage, err = None, {}, str(e)
            res = {c: False for c in CHECKS} | {"_detail": {"error": err}}
        secs = round(time.time() - t0)
        passed = sum(bool(res[c]) for c in CHECKS)
        marks = " ".join(("ok " if res[c] else "FAIL") for c in CHECKS)
        print(f"{case['name']:<20} {passed}/{len(CHECKS)}  {marks}  {secs:>4}s"
              + (f"  ERROR: {err[:80]}" if err else ""))
        for k, v in res["_detail"].items():
            if v:
                print(f"    {k}: {v}")
        results.append({"case": case["name"], "seconds": secs, "passed": passed,
                        "checks": {c: res[c] for c in CHECKS}, "detail": res["_detail"],
                        "usage": usage, "output": data})

    total = sum(r["passed"] for r in results)
    print(f"\nTOTAL {total}/{len(results) * len(CHECKS)} checks passed, "
          f"{sum(r['seconds'] for r in results)}s")
    path = out_dir / f"{stamp}_{(label or 'run').replace(' ', '_').replace('+', '-')}.json"
    path.write_text(json.dumps({"config": config, "total": total,
                                "max": len(results) * len(CHECKS), "results": results},
                               indent=1, default=str), encoding="utf-8")
    print(f"saved {path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=[], choices=sorted(BY_NAME))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    a = ap.parse_args()
    run(a.cases, a.label, a.out)


if __name__ == "__main__":
    main()
