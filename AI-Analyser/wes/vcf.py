"""Minimal, dependency-free VCF reader with VEP CSQ parsing.

Handles plain or (b)gzipped VCF, multi-sample genotypes and the VEP ``CSQ``
INFO field. Multi-allelic records are expected to be split beforehand
(``bcftools norm -m -any``); if they are not, each ALT is emitted as its own
record with genotypes re-coded relative to that ALT.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

CSQ_RE = re.compile(r'Format: ([^"]+)"')


@dataclass
class Genotype:
    gt: str = "./."  # normalised to 0/0, 0/1, 1/1, 0, 1, ./.
    phased: bool = False
    dp: int | None = None
    gq: int | None = None
    ad_ref: int | None = None
    ad_alt: int | None = None
    ps: str | None = None

    @property
    def alleles(self) -> list[str]:
        return re.split(r"[/|]", self.gt)

    @property
    def called(self) -> bool:
        return "." not in self.alleles

    @property
    def alt_count(self) -> int:
        return sum(1 for a in self.alleles if a == "1")

    @property
    def is_het(self) -> bool:
        return self.called and len(self.alleles) == 2 and self.alt_count == 1

    @property
    def is_hom_alt(self) -> bool:
        return self.called and self.alt_count == len(self.alleles) and self.alt_count > 0

    @property
    def is_ref(self) -> bool:
        return self.called and self.alt_count == 0

    @property
    def has_alt(self) -> bool:
        return self.alt_count > 0

    @property
    def allele_balance(self) -> float | None:
        if self.ad_ref is None or self.ad_alt is None:
            return None
        tot = self.ad_ref + self.ad_alt
        return self.ad_alt / tot if tot else None

    def to_dict(self) -> dict:
        return {
            "gt": self.gt, "phased": self.phased, "dp": self.dp, "gq": self.gq,
            "ad": [self.ad_ref, self.ad_alt], "ab": self.allele_balance, "ps": self.ps,
        }


@dataclass
class VcfRecord:
    chrom: str
    pos: int
    id: str
    ref: str
    alt: str
    qual: float | None
    filter: str
    info: dict[str, str | bool]
    genotypes: dict[str, Genotype] = field(default_factory=dict)
    csq: list[dict[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def passed(self) -> bool:
        if self.filter == "PASS":
            return True
        if self.filter in (".", ""):
            # unfiltered callers (e.g. DNAscope): require a minimum QUAL instead
            from .config import settings
            return self.qual is None or self.qual >= settings.min_qual_unfiltered
        return False


@dataclass
class VcfHeader:
    lines: list[str]
    samples: list[str]
    csq_fields: list[str]
    contigs: list[str]

    @property
    def is_vep_annotated(self) -> bool:
        return bool(self.csq_fields)

    def guess_build(self) -> str | None:
        text = "\n".join(self.lines).lower()
        if "grch38" in text or "hg38" in text:
            return "GRCh38"
        if "grch37" in text or "hg19" in text or "b37" in text:
            return "GRCh37"
        for line in self.lines:
            m = re.match(r"##contig=<ID=(?:chr)?1,length=(\d+)", line)
            if m:
                return {"248956422": "GRCh38", "249250621": "GRCh37"}.get(m.group(1))
        return None


def _open(path: Path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def normalise_chrom(chrom: str) -> str:
    c = chrom[3:] if chrom.lower().startswith("chr") else chrom
    return "MT" if c in ("M", "m") else c


def read_header(path: Path) -> VcfHeader:
    lines: list[str] = []
    samples: list[str] = []
    csq_fields: list[str] = []
    contigs: list[str] = []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("##"):
                lines.append(line)
                if line.startswith("##INFO=<ID=CSQ"):
                    m = CSQ_RE.search(line)
                    if m:
                        csq_fields = m.group(1).split("|")
                elif line.startswith("##contig=<ID="):
                    contigs.append(line[13:].split(",")[0].rstrip(">"))
            elif line.startswith("#CHROM"):
                samples = line.split("\t")[9:]
                break
            else:
                break
    if not lines or not lines[0].startswith("##fileformat=VCF"):
        raise ValueError(f"{path.name} does not look like a VCF (missing ##fileformat header)")
    return VcfHeader(lines, samples, csq_fields, contigs)


def _int(v: str | None) -> int | None:
    if v is None or v in (".", ""):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _parse_gt(raw: str, alt_index: int) -> tuple[str, bool]:
    phased = "|" in raw
    parts = re.split(r"[/|]", raw)
    out = []
    for a in parts:
        if a == ".":
            out.append(".")
        elif a == "0":
            out.append("0")
        elif a == str(alt_index):
            out.append("1")
        else:
            # another ALT allele of a multi-allelic site: treat as non-ref "other"
            out.append("0" if a.isdigit() else ".")
    sep = "|" if phased else "/"
    return sep.join(out), phased


def _parse_info(raw: str) -> dict[str, str | bool]:
    info: dict[str, str | bool] = {}
    if raw == ".":
        return info
    for item in raw.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            info[k] = v
        elif item:
            info[item] = True
    return info


def iter_records(path: Path, header: VcfHeader | None = None) -> Iterator[VcfRecord]:
    header = header or read_header(path)
    samples = header.samples
    csq_fields = header.csq_fields
    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            chrom, pos, vid, ref, alts, qual, flt, info_raw = cols[:8]
            info = _parse_info(info_raw)
            fmt = cols[8].split(":") if len(cols) > 8 else []
            sample_cols = cols[9:]
            alt_list = alts.split(",")
            csq_all = []
            if csq_fields and isinstance(info.get("CSQ"), str):
                for entry in info["CSQ"].split(","):
                    vals = entry.split("|")
                    csq_all.append(dict(zip(csq_fields, vals)))
            for ai, alt in enumerate(alt_list, start=1):
                if alt in ("*", ".") or alt.startswith("<"):
                    continue
                gts: dict[str, Genotype] = {}
                for sname, scol in zip(samples, sample_cols):
                    vals = dict(zip(fmt, scol.split(":")))
                    gt_raw = vals.get("GT", "./.")
                    gt, phased = _parse_gt(gt_raw, ai)
                    ad = vals.get("AD")
                    ad_ref = ad_alt = None
                    if ad and ad != ".":
                        ad_parts = ad.split(",")
                        ad_ref = _int(ad_parts[0])
                        ad_alt = _int(ad_parts[ai]) if len(ad_parts) > ai else None
                    gts[sname] = Genotype(
                        gt=gt, phased=phased, dp=_int(vals.get("DP")), gq=_int(vals.get("GQ")),
                        ad_ref=ad_ref, ad_alt=ad_alt, ps=vals.get("PS") if vals.get("PS") not in (None, ".") else None,
                    )
                csq = csq_all
                if len(alt_list) > 1 and csq_all and "Allele" in csq_fields:
                    csq = [c for c in csq_all if c.get("ALLELE_NUM") in (None, "", str(ai))] or csq_all
                yield VcfRecord(
                    chrom=normalise_chrom(chrom), pos=int(pos), id=vid, ref=ref, alt=alt,
                    qual=float(qual) if qual not in (".", "") else None, filter=flt,
                    info=info, genotypes=gts, csq=csq,
                )
