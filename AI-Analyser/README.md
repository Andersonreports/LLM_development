# WES AI Analyser

Web tool for clinical whole-exome interpretation. Upload VCF(s) and clinical history for a
**single**, **trio** or **couple** case. The backend normalises and annotates the variants offline,
runs QC, filters by inheritance model, applies ACMG/AMP criteria, ranks candidates against the
patient's HPO phenotype, and asks Claude for an interpretation draft. A scientist reviews and edits
the draft, then approves it, and the tool issues a PDF report matched to the case type and result
flag.

> Every report is a **draft until a qualified scientist approves it**. AI output is decision
> support, not a diagnosis.

## Pipeline

```
VCF(s) ──► bcftools: subset samples, chr→Ensembl names, split multi-allelics, left-normalise,
           restrict to exome capture targets (±50 bp),
           merge per-person files, keep ALT sites, drop gVCF <NON_REF>
       ──► Ensembl VEP 116 offline (GRCh38, MANE Select per gene, gnomAD v4.1 exome/genome AF,
           ClinVar, AlphaMissense, SIFT/PolyPhen; optional REVEL, SpliceAI)
       ──► QC: Ti/Tv, het/hom, genetic sex vs declared, runs of homozygosity (consanguinity),
           trio Mendelian error rate / parentage, couple relatedness
Clinical history ──► Claude: HPO term extraction (identifiers scrubbed), checked against hp.obo
       ──► Inheritance analysis
             single: AD, AR hom, AR compound-het (read-backed phase if present), XLR/XLD, MT
             trio:   de novo, AR hom (biallelic carriers), compound-het in trans, XLR from
                     carrier mother, inherited-from-affected-parent, UPD hint
             couple: per-partner P/LP carriers, shared AR genes, female X-linked carriers
       ──► ACMG/AMP auto-criteria (PVS1 with LoF-mechanism and NMD checks, PS2/PM6, PM2_P, PM3,
           PM4, PP3/BP4 with ClinGen-calibrated REVEL or AlphaMissense, PP2, BA1/BS1, BP7,
           ClinVar ≥2★ as flagged PP5/BP6). Bayesian points classification (Tavtigian 2020).
       ──► Phenotype similarity (HPO information content, best-match) and ranking
       ──► Secondary findings (ACMG SF v3.2, consent-gated) and carrier findings
       ──► Rule-based result flag (deterministic baseline)
       ──► Claude interpretation: chooses reportable variants, adds or removes criteria with
           stated reasons (PP4, knowledge-based PS1/PM5/... marked for verification),
           result flag, report text, recommendations, reviewer-attention list.
           Criteria are re-scored in code; mismatches are surfaced to the reviewer.
       ──► Scientist review UI ──► approve ──► versioned PDF report
```

### Result flags and report templates

| Case type | Flags | Report |
|---|---|---|
| Single | POSITIVE / INCONCLUSIVE / NEGATIVE | Clinical Exome Sequencing Report |
| Trio | POSITIVE / INCONCLUSIVE / NEGATIVE (plus segregation column) | Trio Exome Sequencing Report |
| Couple | POSITIVE (high-risk couple) / CARRIER / NEGATIVE | Couple Carrier Screening Report |

Every report can include a **secondary findings** section (ACMG SF v3.2) when consent is given, and
single/trio reports can include a **carrier status** section. The wording, sections and colour
coding change with the case type and flag (`wes/templates/report/report.html`).

## Setup

```bash
# 1. Python environment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Bioinformatics tools (user-space micromamba + bioconda: VEP 116, bcftools, htslib, samtools)
./scripts/setup_tools.sh

# 3. Annotation resources (~30 GB: VEP cache, GRCh38 FASTA, ClinVar, AlphaMissense, HPO,
#    gnomAD constraint, ClinGen dosage). Re-runnable; finished files are skipped.
./scripts/download_resources.sh

# 4. Configuration
cp .env.example .env   # set ANTHROPIC_API_KEY, lab name/address, optional WES_USERS

# 5. Run
./run.sh               # http://<server>:8000
```

`/system` shows which tools and resources the server can find — see
[docs/system-status.md](docs/system-status.md) for what each row means and how to fix it.

**Optional predictors.** Place these in `resources/plugins/` and VEP will use them
automatically. Both need licence registration, so they are not downloaded by the script:
- REVEL: `new_tabbed_revel_grch38.tsv.gz` + `.tbi` (prepared as in the VEP REVEL plugin docs).
  When present, the PP3/BP4 thresholds use the ClinGen-calibrated REVEL bands. Otherwise
  AlphaMissense is used.
- SpliceAI: `spliceai_scores.raw.snv.hg38.vcf.gz` and `spliceai_scores.raw.indel.hg38.vcf.gz`
  (Illumina BaseSpace) + `.tbi`.

**ClinVar updates.** ClinVar is released weekly. To refresh it, delete
`resources/clinvar/clinvar.vcf.gz*` and re-run the download script.

## Using it

1. **New case**: choose single / trio / couple. Upload one VCF per person, or one combined
   multi-sample VCF. Enter demographics, clinical history (free text), any known HPO terms, and
   the consent options. If a VCF has several sample columns and the matching is ambiguous, the
   case page asks you to assign columns to people.
2. **Processing**: progress is shown live. A whole exome takes a few minutes, mostly in VEP.
3. **Review & report** tab: the AI draft (or a rules-only draft if no API key is set) lists
   findings with criteria. Hover a criterion for its reason. AI-added criteria have a dashed
   outline. For each finding you can include or exclude it, change its section, override its
   classification, and edit the disease, inheritance and interpretation text. You can also edit
   the overall flag, summary, interpretation, recommendations and limitations.
4. **Candidates** tab: all ranked variants with evidence. Use "+ report" to add one to the report.
5. **QC** tab: sex check, Ti/Tv, ROH, Mendelian errors. Warnings also appear in the Activity log.
6. **Approve**: freezes the case and writes `data/reports/<accession>_v<N>.pdf`. To issue an
   amended report, use "Reopen / amend"; the version number increments.

## Data handling

- Only de-identified content is sent to the Claude API: age, sex, clinical text with the
  patient's name and ID removed, HPO terms, and variant evidence for the top candidates. Names,
  IDs and VCFs stay on the server.
- Uploads, working files, the SQLite database and reports live in `data/`. Back it up and restrict
  access to it.
- If no API key is set, or the API call fails, the pipeline still completes with a rule-based
  draft, and the reviewer writes the narrative.
- Set `WES_USERS` to turn on HTTP basic auth. For production, run it behind HTTPS (for example
  nginx) and use the lab's SSO where possible.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Enables AI phenotype extraction and interpretation |
| `WES_MODEL` | `claude-opus-5-5` | Claude model |
| `WES_AI_EFFORT` | `high` | Reasoning effort for interpretation |
| `WES_AI_MAX_CANDIDATES` | 40 | Ranked candidates sent to the AI |
| `WES_AF_DOMINANT` / `WES_AF_RECESSIVE` | 0.0005 / 0.01 | gnomAD max-AF ceilings |
| `WES_MIN_DP` / `WES_MIN_GQ` / `WES_MIN_AB_HET` | 8 / 20 / 0.2 | Genotype quality gates |
| `WES_TARGETS_BED` | `resources/targets/hg38_exome_comp_spikein_v2.0.2.bed` | Exome capture targets; calls outside them (plus padding) are dropped. Set to an empty path to disable |
| `WES_TARGETS_PADDING` | 50 | bp added to each target so splice-site variants are kept |
| `WES_MIN_QUAL_UNFILTERED` | 30 | Records with FILTER `.` (e.g. DNAscope) are kept only at QUAL >= this |
| `WES_TRIO_MISSING_AS_REF` | true | For parents supplied as separate VCFs, treat a site absent from the parent's file as reference (flagged; de novo then gets PM6, never PS2) |
| `WES_VEP_FORKS` | 8 | VEP parallelism |
| `WES_LAB_NAME`, `WES_LAB_ADDRESS`, `WES_LAB_ACCREDITATION`, `WES_REPORT_METHODOLOGY` | | Report header and methods text |
| `WES_USERS` | – | `user:pass,user2:pass2` for basic auth |

## Interpretation model

The AI draft is written by a **local model by default** (`WES_AI_BACKEND=local`): Ollama
serving `qwen3.5:4b`, so no case data leaves the network. `WES_AI_BACKEND=claude` switches
to the Anthropic API, which sends de-identified data off-site and must be enabled
deliberately. Both backends get identical prompts and must return the same schema; whatever
they propose is re-scored by `wes/acmg.py` and shown to the reviewer.

```bash
ollama pull qwen3.5:4b          # once
```

Safeguards, all on by default. Each exists because the model failed that way in testing,
and in every case an instruction was not enough - the rule had to be enforced in code:

| Setting | What it prevents |
|---|---|
| `WES_AI_ALLOW_CRITERIA_REMOVAL=false` | Deleting an ACMG criterion the engine derived. Removals become reviewer notes |
| `WES_MIN_PHENOTYPE_POSITIVE=0.3` | Calling a case POSITIVE when no reported variant matches the patient's phenotype. The AI's narrative is withdrawn with the flag, so the report cannot contradict itself |
| criterion-code normalisation | `PVS1_VeryStrong` was accepted as a new criterion, counting PVS1 twice (9 points became 17). Codes are now normalised and validated |
| `WES_AI_STRICT_PROMPT=true` | Helps, but does not bind: the supplied data is authoritative, chromosomes come from the variant key, common variants are not pathogenic |

`WES_OLLAMA_THINK` should stay **false**. Thinking mode existed to stop criteria deletion,
which code now blocks; with the longer rule set it runs away and hits the token limit
(benchmark 10/20 at 13 min per case, versus 20/20 at 28 s with it off).

### Benchmark

`bench/` holds four synthetic cases with known-correct answers (no patient data, so they
live in git) and five automated checks each: correct flag, correct variants in the primary
section, no criteria deleted, stated classification consistent with its own criteria, and
no fabricated claims. Run `python -m bench.run`; results land in `bench/results/`.

**qwen3.5:4b, thinking off, strict prompt on (2026-10-26): 20/20 in 109 s**

| Case | Score | Time |
|---|---|---|
| dominant_de_novo (SCN1A, *de novo*) | 5/5 | 30 s |
| recessive_comphet (PAH, in trans) | 5/5 | 28 s |
| negative | 5/5 | 25 s |
| vus_inconclusive (MYH7) | 5/5 | 26 s |

How it got there, measured one change at a time:

| Configuration | Score | Time/case |
|---|---|---|
| Baseline, thinking off | 3/5 on SCN1A | 42 s |
| Thinking on | 19/20 | ~210 s |
| Thinking on, after the guardrails and longer rules | **10/20** | ~770 s - two cases ran to the token limit and failed |
| **Thinking off, guardrails in code** | **20/20** | **28 s** |

The jump came from moving rules into code, not from the prompt. Once criteria deletion was
blocked in code, thinking mode had nothing left to protect against, and removing it fixed
the regression and made the tool 28x faster.

### Validation status

**Not validated for clinical use.** Four synthetic cases and one real case can show that
specific failures are fixed; they cannot show the tool is right in general.

| Check | Result |
|---|---|
| Synthetic benchmark | 20/20 |
| One real case with a signed report (WES, Sep 2026) | tool NEGATIVE = lab NEGATIVE |

That real case is also where the current limits were found. Before clinical use: add more
signed-off cases (patient data - keep them out of git), and have scientists compare drafts
against what they actually issued.

### Known gaps found on the real case

- **PVS1 ignores allele frequency.** A frameshift at 0.96% population frequency received
  the full 8 points. Frequency should attenuate or withhold PVS1.
- **BS1 uses one generic threshold.** 1% for any recessive gene; the variants above sat at
  0.956% and escaped. Disease-appropriate ceilings are needed.
- **PM3 can be circular.** Two variants in the same gene each counted the other as the
  "pathogenic" partner, when both were only pathogenic by virtue of the same analysis.
- **No indication-based gene restriction.** The whole exome is ranked equally, so a
  microcephaly gene surfaced in a child investigated for gastrointestinal bleeding. The
  laboratory's own workflow narrows to an indication gene list first.

## Known limitations

- SNVs and small indels from the VCF only. No CNV, repeat-expansion or mitochondrial-heteroplasmy
  analysis, and no coverage/BAM-level checks.
- PS1/PM5/PM1 and PS3/BS3, which need variant-level literature or functional evidence, are not
  computed automatically. The AI can propose them, labelled "Knowledge-based", and they must be
  verified by the reviewer.
- Gene-specific ClinGen VCEP rules and frequency thresholds are not yet encoded. Generic
  thresholds are used.
- ACMG SF list is v3.2 (`wes/resources.py`). Update it when ACMG publishes a new version.
- GRCh38 only. GRCh37 input is rejected.

## Development

```bash
.venv/bin/python -m pytest -q          # unit + end-to-end tests on synthetic annotated VCFs
```

Code map: `wes/annotate.py` (bcftools/VEP), `wes/vcf.py` (parser), `wes/variants.py`,
`wes/qc.py`, `wes/inheritance.py`, `wes/acmg.py`, `wes/phenotype.py`, `wes/analysis.py`
(ranking, SF, carrier, couple, rule flag), `wes/ai.py` (Claude), `wes/pipeline.py`
(orchestration), `wes/report.py` + `templates/report/`, `wes/web/app.py` (FastAPI).
