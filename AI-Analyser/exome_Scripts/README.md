# exome_Scripts

ClawBio exome skills plus the helper scripts and reference data needed to analyse
an exome from a VCF. Runs fully offline; no patient data is sent anywhere.

## Run

```bash
~/Desktop/whole_exome_sequencing/exome_Scripts/run_exome.sh  <sample.vcf.gz>  <output_folder>
```

Input: one sample VCF, GRCh38, `chr` chromosome names, bgzipped. Filtered (PASS)
or unfiltered DNAscope VCFs both work (FILTER "." kept if QUAL >= 30; LowQual dropped).
Takes about 5-10 minutes.

Steps: restrict to exome targets, remove reference-only calls, match ClinVar,
annotate with VEP (offline), ACMG/AMP classification, rare loss-of-function count.

Outputs in the output folder:

- `vep_table.tsv`: every variant with gene, consequence, gnomAD, SIFT/PolyPhen, genotype
- `clinvar_table.tsv`: ClinVar match for every variant
- `acmg_out/tables/acmg_classifications.tsv` and `acmg_out/report.md`: ACMG class for every variant
- `acmg_out/tables/secondary_findings.tsv`: variants in ACMG SF v3.2 genes
- `rare_out/report.md`: rare loss-of-function count

Interpreting the results against the patient's history is a manual step.

## Folder layout

| Path | What it is |
|---|---|
| `run_exome.sh` | The runner: give it a VCF and an output folder |
| `clawbio_skills/` | ClawBio exome skills, copied from the installed package (version 0.7.1) |
| `acmg_local/` | ClawBio ACMG classifier (clinical-variant-reporter) adapted to read local evidence instead of the Ensembl web service; rules unchanged. This is the version `run_exome.sh` uses |
| `helpers/parse_vep.py` | Flattens VEP output to `vep_table.tsv` and writes the input for the rare-variant skill |
| `helpers/build_local_evidence.py` | Builds ACMG evidence from local VEP + ClinVar |
| `reference/targets.bed` | Exome capture targets (3 columns, from hg38_exome_comp_spikein_v2.0.2) |
| `reference/clinvar_chr.vcf.gz` | ClinVar GRCh38, release of 29 Sep 2026, chr-prefixed |
| `reference/chrmap.txt` | Chromosome renaming map, used when updating ClinVar |
| `environment.yml` | Exact package list of the `exome` conda env |

### ClawBio skills included

| Skill | Used by run_exome.sh? | Note |
|---|---|---|
| clinical-variant-reporter (ACMG) | Yes, as the local copy in `acmg_local/` | The original in `clawbio_skills/` sends variants to the Ensembl web service |
| rare-high-impact-variants | Yes | Offline |
| variant-annotation | No | Sends variants to the Ensembl web service; replaced by local VEP |
| wes-clinical-report-en | No | Needs a Novogene-style markdown input; marked PROPRIETARY in its metadata |
| clinical-variant-prioritizer | No | Built for array or genome genotype lists (rsID input) |
| vcf-annotator | No | Annotates with Ensembl VEP, ClinVar and gnomAD over the web; replaced by local VEP |

## Requirements

- Conda env `exome` (recreate elsewhere with `conda env create -f environment.yml`).
- **VEP cache** release 116, GRCh38, at `~/.vep` (about 52 GB; not copied here).
  To reinstall: download `homo_sapiens_vep_116_GRCh38.tar.gz` from
  ftp.ensembl.org/pub/release-116/variation/indexed_vep_cache/ and unpack into `~/.vep`.

## Updating ClinVar

Download `clinvar.vcf.gz` (+ `.tbi`) from ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/, then:

```bash
bcftools annotate --rename-chrs reference/chrmap.txt clinvar.vcf.gz -Oz -o reference/clinvar_chr.vcf.gz
bcftools index -t reference/clinvar_chr.vcf.gz
```

Research use only. Findings need confirmation in an accredited laboratory and clinical review.
