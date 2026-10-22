"""Runtime configuration. Every value can be overridden with an environment
variable of the same name (or a line in ./.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        return Path(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _env("WES_DATA_DIR", ROOT / "data")
    resources_dir: Path = _env("WES_RESOURCES_DIR", ROOT / "resources")
    tools_bin: Path = _env("WES_TOOLS_BIN", ROOT / "tools" / "mamba" / "envs" / "wes" / "bin")
    genome_build: str = "GRCh38"

    # VEP
    vep_forks: int = _env("WES_VEP_FORKS", 8)
    vep_buffer: int = _env("WES_VEP_BUFFER", 5000)

    # Genotype quality gates
    min_dp: int = _env("WES_MIN_DP", 8)
    min_gq: int = _env("WES_MIN_GQ", 20)
    min_ab_het: float = _env("WES_MIN_AB_HET", 0.2)
    require_pass: bool = _env("WES_REQUIRE_PASS", True)
    # FILTER "." (unfiltered callers such as DNAscope) is accepted only above this QUAL
    min_qual_unfiltered: float = _env("WES_MIN_QUAL_UNFILTERED", 30.0)
    # Exome capture targets (chr-prefixed or not); calls outside +/- padding are dropped
    targets_bed: Path = _env("WES_TARGETS_BED", ROOT / "resources" / "targets" / "hg38_exome_comp_spikein_v2.0.2.bed")
    targets_padding: int = _env("WES_TARGETS_PADDING", 50)
    # Trio from separate per-person VCFs: treat a proband site absent from a
    # parent's file as reference (flagged; de novo then gets PM6, never PS2)
    trio_missing_as_ref: bool = _env("WES_TRIO_MISSING_AS_REF", True)

    # Population frequency thresholds (gnomAD, max across populations)
    af_dominant: float = _env("WES_AF_DOMINANT", 0.0005)
    af_recessive: float = _env("WES_AF_RECESSIVE", 0.01)
    af_known_pathogenic: float = _env("WES_AF_KNOWN_PATHOGENIC", 0.05)

    # How many ranked candidates are handed to the AI interpreter
    ai_max_candidates: int = _env("WES_AI_MAX_CANDIDATES", 40)

    # Claude
    anthropic_model: str = _env("WES_MODEL", "claude-opus-5-5")
    ai_effort: str = _env("WES_AI_EFFORT", "high")
    ai_enabled: bool = _env("WES_AI_ENABLED", True)

    # Lab / report identity
    lab_name: str = _env("WES_LAB_NAME", "Anderson Diagnostics & Labs")
    lab_address: str = _env("WES_LAB_ADDRESS", "")
    lab_accreditation: str = _env("WES_LAB_ACCREDITATION", "")
    # Leave empty to use the standard text in wes/report_text.py (sequencing + this pipeline's analysis)
    report_methodology: str = _env("WES_REPORT_METHODOLOGY", "")
    # Optional pre-printed letterhead image for page 1 (otherwise lab name/address text is used)
    letterhead_image: Path = _env("WES_LETTERHEAD_IMAGE", ROOT / "resources" / "report" / "letterhead.png")
    # Signature strip (image with signatories' names) printed on APPROVED reports only
    signature_image: Path = _env("WES_SIGNATURE_IMAGE", ROOT / "resources" / "report" / "signatures.jpg")
    signatures_on_final: bool = _env("WES_SIGNATURES_ON_FINAL", True)
    # Max genes listed in the indication-based gene appendix
    gene_list_size: int = _env("WES_GENE_LIST_SIZE", 80)
    gene_list_min_score: float = _env("WES_GENE_LIST_MIN_SCORE", 0.45)

    # Optional HTTP basic auth, "user:password,user2:password2"
    users: str = _env("WES_USERS", "")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'wes.db'}"

    def tool(self, name: str) -> str:
        p = self.tools_bin / name
        return str(p) if p.exists() else name

    # ---- resource paths ----
    @property
    def vep_cache(self) -> Path:
        return self.resources_dir / "vep_cache"

    @property
    def fasta(self) -> Path:
        return self.resources_dir / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

    @property
    def clinvar_vcf(self) -> Path:
        return self.resources_dir / "clinvar" / "clinvar.vcf.gz"

    @property
    def alphamissense(self) -> Path:
        return self.resources_dir / "plugins" / "AlphaMissense_hg38.tsv.gz"

    @property
    def revel(self) -> Path:
        return self.resources_dir / "plugins" / "new_tabbed_revel_grch38.tsv.gz"

    @property
    def spliceai_snv(self) -> Path:
        return self.resources_dir / "plugins" / "spliceai_scores.raw.snv.hg38.vcf.gz"

    @property
    def spliceai_indel(self) -> Path:
        return self.resources_dir / "plugins" / "spliceai_scores.raw.indel.hg38.vcf.gz"

    @property
    def hpo_dir(self) -> Path:
        return self.resources_dir / "hpo"

    @property
    def gene_dir(self) -> Path:
        return self.resources_dir / "gene"


settings = Settings()

for d in (settings.uploads_dir, settings.work_dir, settings.reports_dir):
    d.mkdir(parents=True, exist_ok=True)
