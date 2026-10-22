"""Loads gene / phenotype knowledge bases into memory (once per process).

All loaders degrade gracefully: if a resource file is missing the corresponding
lookup simply returns empty results and the pipeline records a warning.
"""
from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)

INHERITANCE_TERMS = {
    "HP:0000006": "AD",
    "HP:0000007": "AR",
    "HP:0001417": "XL",
    "HP:0001419": "XLR",
    "HP:0001423": "XLD",
    "HP:0001427": "MT",
    "HP:0001450": "YL",
    "HP:0032113": "SD",   # semidominant
    "HP:0010984": "DIGENIC",
}

# ACMG SF v3.2 (Miller et al. 2023). Value = how the finding is reportable.
# "AD" = single P/LP variant; "AR" = biallelic P/LP; special-cased genes noted.
ACMG_SF_V3_2: dict[str, str] = {
    # Cancer predisposition
    **{g: "AD" for g in [
        "APC", "BMPR1A", "BRCA1", "BRCA2", "MAX", "MEN1", "MLH1", "MSH2", "MSH6", "NF2",
        "PALB2", "PMS2", "PTEN", "RB1", "RET", "SDHAF2", "SDHB", "SDHC", "SDHD", "SMAD4",
        "STK11", "TMEM127", "TP53", "TSC1", "TSC2", "VHL", "WT1"]},
    "MUTYH": "AR",
    # Cardiovascular
    **{g: "AD" for g in [
        "ACTA2", "ACTC1", "APOB", "BAG3", "CALM1", "CALM2", "CALM3", "COL3A1", "DES", "DSC2",
        "DSG2", "DSP", "FBN1", "FLNC", "KCNH2", "KCNQ1", "LDLR", "LMNA", "MYBPC3", "MYH11",
        "MYH7", "MYL2", "MYL3", "PCSK9", "PKP2", "PRKAG2", "RBM20", "RYR2", "SCN5A", "SMAD3",
        "TGFBR1", "TGFBR2", "TMEM43", "TNNC1", "TNNI3", "TNNT2", "TPM1", "TTN", "TTR"]},
    "GLA": "XL",
    "CASQ2": "AR", "TRDN": "AR",
    # Inborn errors of metabolism
    "BTD": "AR", "GAA": "AR", "OTC": "XL",
    # Miscellaneous
    "ACVRL1": "AD", "ENG": "AD", "HNF1A": "AD", "RYR1": "AD", "CACNA1S": "AD",
    "ATP7B": "AR", "RPE65": "AR",
    "HFE": "AR",  # p.Cys282Tyr homozygotes only
}
ACMG_SF_NOTES = {
    "TTN": "Only truncating variants in A-band are reportable.",
    "HFE": "Only p.Cys282Tyr homozygosity is reportable.",
}


# ---------------------------------------------------------------- HPO ontology
@dataclass
class HpoTerm:
    id: str
    name: str
    parents: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)


class Ontology:
    def __init__(self, terms: dict[str, HpoTerm], alt_ids: dict[str, str]):
        self.terms = terms
        self.alt_ids = alt_ids
        self._anc: dict[str, frozenset[str]] = {}
        self._name_index = {}
        for t in terms.values():
            self._name_index[t.name.lower()] = t.id
            for s in t.synonyms:
                self._name_index.setdefault(s.lower(), t.id)

    def resolve(self, hpo_id: str) -> str | None:
        hpo_id = hpo_id.strip().upper().replace("HP_", "HP:")
        if hpo_id in self.terms:
            return hpo_id
        return self.alt_ids.get(hpo_id)

    def by_name(self, name: str) -> str | None:
        return self._name_index.get(name.strip().lower())

    def name(self, hpo_id: str) -> str:
        t = self.terms.get(hpo_id)
        return t.name if t else hpo_id

    def ancestors(self, hpo_id: str) -> frozenset[str]:
        if hpo_id in self._anc:
            return self._anc[hpo_id]
        seen = {hpo_id}
        stack = [hpo_id]
        while stack:
            cur = stack.pop()
            t = self.terms.get(cur)
            if not t:
                continue
            for p in t.parents:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        res = frozenset(seen)
        self._anc[hpo_id] = res
        return res

    def search(self, query: str, limit: int = 15) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        hits = []
        for name, tid in self._name_index.items():
            if q in name:
                hits.append((0 if name.startswith(q) else 1, len(name), tid, name))
        hits.sort()
        out, seen = [], set()
        for _, _, tid, _ in hits:
            if tid in seen:
                continue
            seen.add(tid)
            out.append({"id": tid, "name": self.name(tid)})
            if len(out) >= limit:
                break
        return out


def _parse_obo(path: Path) -> Ontology:
    terms: dict[str, HpoTerm] = {}
    alt: dict[str, str] = {}
    cur: dict | None = None

    def flush():
        if cur and cur.get("id") and not cur.get("obsolete"):
            terms[cur["id"]] = HpoTerm(cur["id"], cur.get("name", ""), cur.get("is_a", []), cur.get("syn", []))
            for a in cur.get("alt", []):
                alt[a] = cur["id"]

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush()
                cur = {}
            elif line.startswith("[") and line.endswith("]"):
                flush()
                cur = None
            elif cur is not None and ": " in line:
                k, v = line.split(": ", 1)
                if k == "id":
                    cur["id"] = v
                elif k == "name":
                    cur["name"] = v
                elif k == "is_a":
                    cur.setdefault("is_a", []).append(v.split(" ")[0])
                elif k == "alt_id":
                    cur.setdefault("alt", []).append(v)
                elif k == "is_obsolete" and v == "true":
                    cur["obsolete"] = True
                elif k == "synonym" and '"' in v:
                    cur.setdefault("syn", []).append(v.split('"')[1])
    flush()
    return Ontology(terms, alt)


# ------------------------------------------------------------ Knowledge base
@dataclass
class Disease:
    id: str
    name: str
    inheritance: set[str] = field(default_factory=set)


@dataclass
class GeneInfo:
    symbol: str
    diseases: list[Disease] = field(default_factory=list)
    hpo: set[str] = field(default_factory=set)
    loeuf: float | None = None
    pli: float | None = None
    mis_z: float | None = None
    hi_score: str | None = None     # ClinGen haploinsufficiency score
    ts_score: str | None = None

    @property
    def inheritance(self) -> set[str]:
        out: set[str] = set()
        for d in self.diseases:
            out |= d.inheritance
        return out

    @property
    def lof_intolerant(self) -> bool:
        return (self.hi_score == "3") or (self.loeuf is not None and self.loeuf < 0.35) or (
            self.pli is not None and self.pli >= 0.9)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "diseases": [{"id": d.id, "name": d.name, "inheritance": sorted(d.inheritance)} for d in self.diseases[:8]],
            "inheritance": sorted(self.inheritance),
            "loeuf": self.loeuf, "pli": self.pli, "mis_z": self.mis_z,
            "clingen_hi": self.hi_score, "acmg_sf": ACMG_SF_V3_2.get(self.symbol),
        }


class KnowledgeBase:
    def __init__(self):
        self.warnings: list[str] = []
        self.ontology: Ontology | None = None
        self.genes: dict[str, GeneInfo] = {}
        self.term_ic: dict[str, float] = {}
        self._load()

    def gene(self, symbol: str) -> GeneInfo:
        g = self.genes.get(symbol)
        if g is None:
            g = GeneInfo(symbol)
        return g

    def _load(self) -> None:
        hpo = settings.hpo_dir
        obo = hpo / "hp.obo"
        if obo.exists():
            self.ontology = _parse_obo(obo)
        else:
            self.warnings.append("HPO ontology (hp.obo) missing - phenotype matching disabled")

        diseases: dict[str, Disease] = {}
        hpoa = hpo / "phenotype.hpoa"
        if hpoa.exists():
            with open(hpoa, encoding="utf-8") as fh:
                rows = (l for l in fh if not l.startswith("#"))
                rdr = csv.DictReader(rows, delimiter="\t")
                for r in rdr:
                    did = r.get("database_id")
                    if not did:
                        continue
                    d = diseases.setdefault(did, Disease(did, r.get("disease_name", "")))
                    if r.get("aspect") == "I" and r.get("hpo_id") in INHERITANCE_TERMS:
                        d.inheritance.add(INHERITANCE_TERMS[r["hpo_id"]])
        else:
            self.warnings.append("phenotype.hpoa missing - disease inheritance unknown")

        g2d = hpo / "genes_to_disease.txt"
        if g2d.exists():
            with open(g2d, encoding="utf-8") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    sym = r.get("gene_symbol")
                    did = r.get("disease_id")
                    if not sym or not did:
                        continue
                    if r.get("association_type", "MENDELIAN").upper() not in ("MENDELIAN", "UNKNOWN"):
                        continue
                    g = self.genes.setdefault(sym, GeneInfo(sym))
                    d = diseases.get(did) or Disease(did, did)
                    if all(x.id != did for x in g.diseases):
                        g.diseases.append(d)
        else:
            self.warnings.append("genes_to_disease.txt missing - gene-disease associations unavailable")

        g2p = hpo / "genes_to_phenotype.txt"
        term_genes: dict[str, set[str]] = defaultdict(set)
        if g2p.exists():
            with open(g2p, encoding="utf-8") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    sym, tid = r.get("gene_symbol"), r.get("hpo_id")
                    if sym and tid:
                        self.genes.setdefault(sym, GeneInfo(sym)).hpo.add(tid)
        else:
            self.warnings.append("genes_to_phenotype.txt missing - phenotype matching disabled")

        # information content of each term = -log(fraction of genes annotated to it or descendants)
        if self.ontology and self.genes:
            for sym, g in self.genes.items():
                closure: set[str] = set()
                for t in g.hpo:
                    rt = self.ontology.resolve(t)
                    if rt:
                        closure |= self.ontology.ancestors(rt)
                g.hpo = closure
                for t in closure:
                    term_genes[t].add(sym)
            n = max(1, sum(1 for g in self.genes.values() if g.hpo))
            self.term_ic = {t: -math.log(len(gs) / n) for t, gs in term_genes.items()}

        self._load_constraint()
        self._load_clingen()

    def _load_constraint(self) -> None:
        p = settings.gene_dir / "gnomad.v4.1.constraint_metrics.tsv"
        if not p.exists():
            self.warnings.append("gnomAD constraint metrics missing - LoF intolerance unknown")
            return

        def f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if str(r.get("mane_select", "")).lower() != "true":
                    continue
                sym = r.get("gene")
                if not sym:
                    continue
                g = self.genes.setdefault(sym, GeneInfo(sym))
                g.loeuf = f(r.get("lof.oe_ci.upper"))
                g.pli = f(r.get("lof.pLI"))
                g.mis_z = f(r.get("mis.z_score"))

    def _load_clingen(self) -> None:
        p = settings.gene_dir / "ClinGen_gene_curation_list_GRCh38.tsv"
        if not p.exists():
            self.warnings.append("ClinGen dosage list missing")
            return
        with open(p, encoding="utf-8") as fh:
            header = None
            for line in fh:
                if line.startswith("#Gene Symbol"):
                    header = line.lstrip("#").rstrip("\n").split("\t")
                    continue
                if line.startswith("#") or header is None:
                    continue
                r = dict(zip(header, line.rstrip("\n").split("\t")))
                sym = r.get("Gene Symbol")
                if not sym:
                    continue
                g = self.genes.setdefault(sym, GeneInfo(sym))
                g.hi_score = r.get("Haploinsufficiency Score") or None
                g.ts_score = r.get("Triplosensitivity Score") or None


@lru_cache(maxsize=1)
def kb() -> KnowledgeBase:
    log.info("loading knowledge base from %s", settings.resources_dir)
    return KnowledgeBase()
