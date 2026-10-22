"""VCF preprocessing (bcftools) and offline annotation (Ensembl VEP).

Produces one normalised, multi-sample, VEP-annotated VCF per case whose sample
columns are renamed to their roles (proband/father/mother/partner1/partner2).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .vcf import normalise_chrom, read_header

log = logging.getLogger(__name__)

PRIMARY_CONTIGS = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]


class PipelineError(RuntimeError):
    pass


@dataclass
class SampleInput:
    role: str           # proband | father | mother | partner1 | partner2
    vcf: Path
    sample_id: str      # column name inside the VCF


def _tool_env() -> dict[str, str]:
    # VEP's shebang is "env perl": the bioconda perl must win over the system one
    env = dict(os.environ)
    env["PATH"] = f"{settings.tools_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run(cmd: list[str], log_file: Path, stdin=None) -> None:
    log.info("run: %s", " ".join(cmd))
    with open(log_file, "a") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=lf, stdin=stdin, env=_tool_env())
    if proc.returncode != 0:
        tail = log_file.read_text()[-2000:]
        raise PipelineError(f"Command failed ({proc.returncode}): {cmd[0]}\n{tail}")


def tools_available() -> dict[str, bool]:
    return {
        "bcftools": shutil.which(settings.tool("bcftools")) is not None,
        "tabix": shutil.which(settings.tool("tabix")) is not None,
        "vep": shutil.which(settings.tool("vep")) is not None,
        "vep_cache": settings.vep_cache.exists() and any(settings.vep_cache.glob("homo_sapiens*/*GRCh38")),
        "fasta": settings.fasta.exists(),
        "clinvar": settings.clinvar_vcf.exists(),
        "alphamissense": settings.alphamissense.exists(),
        "revel": settings.revel.exists(),
        "spliceai": settings.spliceai_snv.exists(),
    }


def _chr_map(work: Path) -> Path:
    p = work / "chr_rename.txt"
    lines = [f"chr{c}\t{c}" for c in PRIMARY_CONTIGS if c != "MT"] + ["chrM\tMT", "chrMT\tMT"]
    p.write_text("\n".join(lines) + "\n")
    return p


def _targets_bed(work: Path) -> Path | None:
    """Capture-target BED in Ensembl contig names, padded to keep splice sites."""
    src = settings.targets_bed
    if not src or not Path(src).exists():
        return None
    out = work / "targets.padded.bed"
    pad = settings.targets_padding
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            c, start, end = line.split("\t")[:3]
            c = normalise_chrom(c)
            fout.write(f"{c}\t{max(0, int(start) - pad)}\t{int(end) + pad}\n")
    return out


def preprocess(inputs: list[SampleInput], work: Path, log_file: Path) -> Path:
    """Subset, rename, normalise and merge input VCFs. Returns a bgzipped VCF
    containing only sites where at least one sample carries an ALT allele."""
    bcftools = settings.tool("bcftools")
    if shutil.which(bcftools) is None:
        raise PipelineError("bcftools not found - run scripts/setup_tools.sh")
    work.mkdir(parents=True, exist_ok=True)
    chr_map = _chr_map(work)
    targets = ",".join(PRIMARY_CONTIGS)
    use_fasta = settings.fasta.exists()
    target_bed = _targets_bed(work)

    # group by source file so one multi-sample VCF is processed once
    by_file: dict[Path, list[SampleInput]] = {}
    for si in inputs:
        by_file.setdefault(si.vcf, []).append(si)

    normed: list[Path] = []
    for idx, (vcf, sis) in enumerate(by_file.items()):
        rename = work / f"samples_{idx}.txt"
        rename.write_text("".join(f"{si.sample_id}\t{si.role}\n" for si in sis))
        sub = work / f"in_{idx}.subset.vcf.gz"
        _run([bcftools, "view", "-s", ",".join(si.sample_id for si in sis), "--force-samples",
              "-Oz", "-o", str(sub), str(vcf)], log_file)
        ren = work / f"in_{idx}.renamed.vcf.gz"
        _run([bcftools, "annotate", "--rename-chrs", str(chr_map), "-Oz", "-o", str(ren), str(sub)], log_file)
        ren2 = work / f"in_{idx}.roles.vcf.gz"
        _run([bcftools, "reheader", "-s", str(rename), "-o", str(ren2), str(ren)], log_file)
        out = work / f"in_{idx}.norm.vcf.gz"
        norm = [bcftools, "norm", "-m", "-any", "-t", targets, "-Oz", "-o", str(out)]
        if use_fasta:
            norm += ["-f", str(settings.fasta), "--check-ref", "ws"]
        _run(norm + [str(ren2)], log_file)
        _run([bcftools, "sort", "-Oz", "-o", str(out) + ".sorted.vcf.gz", str(out)], log_file)
        Path(str(out) + ".sorted.vcf.gz").replace(out)
        if target_bed:
            _run([bcftools, "view", "-T", str(target_bed), "-Oz", "-o", str(out) + ".ontarget.vcf.gz", str(out)], log_file)
            Path(str(out) + ".ontarget.vcf.gz").replace(out)
        _run([bcftools, "index", "-f", "-t", str(out)], log_file)
        normed.append(out)

    if len(normed) > 1:
        merged = work / "merged.vcf.gz"
        _run([bcftools, "merge", "-m", "none", "-Oz", "-o", str(merged)] + [str(p) for p in normed], log_file)
    else:
        merged = normed[0]

    final = work / "case.alt.vcf.gz"
    # keep sites where any sample carries an ALT; drop gVCF symbolic alleles
    alt_only = work / "case.alt.tmp.vcf.gz"
    _run([bcftools, "view", "-i", 'GT="alt"', "-Oz", "-o", str(alt_only), str(merged)], log_file)
    _run([bcftools, "view", "-e", 'ALT="<NON_REF>" || ALT="<*>"', "-Oz", "-o", str(final), str(alt_only)], log_file)
    _run([bcftools, "index", "-f", "-t", str(final)], log_file)
    return final


def vep_command(inp: Path, out: Path) -> list[str]:
    s = settings
    cmd = [
        s.tool("vep"), "--offline", "--cache", "--dir_cache", str(s.vep_cache),
        "--assembly", "GRCh38", "--species", "homo_sapiens",
        "-i", str(inp), "-o", str(out), "--vcf", "--compress_output", "bgzip",
        "--force_overwrite", "--no_stats", "--fork", str(s.vep_forks), "--buffer_size", str(s.vep_buffer),
        "--symbol", "--numbers", "--canonical", "--mane", "--biotype", "--protein", "--variant_class",
        "--sift", "b", "--polyphen", "b",
        "--af_gnomade", "--af_gnomadg", "--max_af", "--check_existing",
        "--per_gene", "--pick_order", "mane_select,canonical,appris,tsl,biotype,ccds,rank,length",
    ]
    plugin_dirs = sorted((s.tools_bin.parent / "share").glob("ensembl-vep-*"))
    if plugin_dirs:  # bioconda ships the plugin modules here
        cmd += ["--dir_plugins", str(plugin_dirs[-1])]
    if s.fasta.exists():
        cmd += ["--fasta", str(s.fasta), "--hgvs"]
    if s.clinvar_vcf.exists():
        cmd += ["--custom",
                f"file={s.clinvar_vcf},short_name=ClinVar,format=vcf,type=exact,coords=0,"
                "fields=CLNSIG%CLNREVSTAT%CLNDN%CLNSIGCONF%ALLELEID"]
    if s.alphamissense.exists():
        cmd += ["--plugin", f"AlphaMissense,file={s.alphamissense}"]
    if s.revel.exists():
        cmd += ["--plugin", f"REVEL,file={s.revel}"]
    if s.spliceai_snv.exists() and s.spliceai_indel.exists():
        cmd += ["--plugin", f"SpliceAI,snv={s.spliceai_snv},indel={s.spliceai_indel}"]
    return cmd


def annotate(inp: Path, work: Path, log_file: Path) -> Path:
    avail = tools_available()
    if not avail["vep"]:
        raise PipelineError("VEP not found - run scripts/setup_tools.sh")
    if not avail["vep_cache"]:
        raise PipelineError("VEP GRCh38 cache missing - run scripts/download_resources.sh")
    out = work / "case.vep.vcf.gz"
    _run(vep_command(inp, out), log_file)
    return out


def prepare_case_vcf(inputs: list[SampleInput], work: Path, log_file: Path) -> Path:
    """Full preprocessing + annotation. If a single input VCF is already
    VEP-annotated (has CSQ), it is used as-is after sample renaming in Python."""
    if len({si.vcf for si in inputs}) == 1 and read_header(inputs[0].vcf).is_vep_annotated:
        log.info("input already VEP-annotated; skipping preprocessing/annotation")
        return inputs[0].vcf
    pre = preprocess(inputs, work, log_file)
    return annotate(pre, work, log_file)
