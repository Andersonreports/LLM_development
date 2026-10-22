# System status page — reference

The **System status** page (`/system` in the sidebar, under Admin) is a diagnostics view.
It doesn't control anything — it just reports whether the pipeline has everything it needs
and what settings it's currently running with. Check it after setup, after changing `.env`,
or whenever a case fails with an unclear error.

## Tools & annotation resources

Each row is a hard dependency check. **AVAILABLE** (green) means the pipeline found it;
anything else means that case analysis will fail or be degraded until it's fixed.

| Row | What it is | If missing |
|---|---|---|
| `bcftools` | Normalises VCFs, splits multi-allelic sites, merges trio/couple samples | Run `./scripts/setup_tools.sh` |
| `tabix / htslib` | Indexes compressed VCFs/BED files so lookups are fast | Run `./scripts/setup_tools.sh` |
| `Ensembl VEP` | Annotates every variant: gene, consequence, HGVS, gnomAD, ClinVar, etc. | Run `./scripts/setup_tools.sh` |
| `VEP cache (GRCh38)` | The ~27 GB offline Ensembl annotation database VEP reads from | Run `./scripts/download_resources.sh` |
| `Reference FASTA (GRCh38)` | The genome sequence, needed for HGVS notation and normalisation | Run `./scripts/download_resources.sh` |
| `ClinVar` | Known pathogenic/benign variant classifications | Run `./scripts/download_resources.sh` |
| `AlphaMissense` | AI-predicted pathogenicity score for missense variants | Run `./scripts/download_resources.sh` |
| `REVEL` *(optional)* | An alternative, ClinGen-calibrated missense pathogenicity predictor. Used instead of AlphaMissense for PP3/BP4 scoring when present | Needs manual licence registration — see README "Optional predictors" |
| `SpliceAI` *(optional)* | Predicts whether a variant disrupts splicing | Needs manual licence registration (Illumina BaseSpace) — see README "Optional predictors" |
| `Claude API` | Writes the narrative interpretation, double-checks the automatic result, extracts phenotype terms from clinical text with more nuance than the built-in text matcher | Add `ANTHROPIC_API_KEY` to `.env` and restart |

**Without REVEL/SpliceAI or the Claude API key, the pipeline still runs completely** — it
just has slightly less evidence to work with (AlphaMissense substitutes for REVEL; the
built-in HPO text-matcher substitutes for AI phenotype extraction; a rules-only draft
substitutes for the AI-written interpretation).

## Knowledge base & thresholds

Read-only snapshot of what's loaded and configured right now.

| Row | Meaning | Set via |
|---|---|---|
| Genes with annotations | How many genes have gene-disease/inheritance data loaded | Comes from the HPO download (`genes_to_disease.txt`, `genes_to_phenotype.txt`) |
| HPO terms | Size of the phenotype ontology used for matching clinical text to genes | Comes from `hp.obo` |
| Capture targets | The exome kit BED file variants are restricted to, and how much padding is added around each region (so splice-site variants just outside the kit aren't lost) | `WES_TARGETS_BED`, `WES_TARGETS_PADDING` in `.env` |
| Max AF dominant / recessive | Population-frequency ceilings: a variant commoner than this in gnomAD is filtered out (recessive genes allow a higher ceiling, since carriers are more common than affected individuals) | `WES_AF_DOMINANT`, `WES_AF_RECESSIVE` |
| Min DP / GQ / allele balance | Genotype quality gates: how much read depth, genotype quality score, and (for hets) allele balance a call needs to be trusted | `WES_MIN_DP`, `WES_MIN_GQ`, `WES_MIN_AB_HET` |
| AI candidates per case | How many of the top-ranked variants get sent to Claude for interpretation (only relevant once the API key is set) | `WES_AI_MAX_CANDIDATES` |
| AI effort | How much reasoning effort Claude uses per case — higher is slower and more thorough | `WES_AI_EFFORT` (`low` / `medium` / `high` / `xhigh` / `max`) |

All of these are environment variables read from `.env` (see `.env.example` in the project
root for the full list with defaults) — edit `.env`, restart the server, and this page will
reflect the change.

## Quick troubleshooting

- **A tool/resource shows red/missing** → re-run the matching setup script (see table above),
  then refresh this page.
- **Claude API shows "NO API KEY"** and you want AI interpretation → add
  `ANTHROPIC_API_KEY=sk-ant-...` to `.env`, restart the server (`./run.sh`).
- **Settings look wrong** → check `.env` against `.env.example`; a typo'd variable name is
  silently ignored and falls back to the default shown here.
