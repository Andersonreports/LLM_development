"""Builds exome_Scripts_contents.pdf: what was copied into exome_Scripts and from where.
Run: python make_contents_pdf.py [test-status text]"""
import os, sys, datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exome_Scripts_contents.pdf")
TEST = sys.argv[1] if len(sys.argv) > 1 else "Test runs on sample-2 and the trio are in progress; results will be added when they finish."

ss = getSampleStyleSheet(); NAVY = colors.HexColor("#1f3a5f")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=17)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=10)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.4, leading=12.8)
C = ParagraphStyle("C", parent=B, fontSize=8.3, leading=10.6)
CH = ParagraphStyle("CH", parent=C, textColor=colors.white, fontName="Helvetica-Bold")
KEY = ParagraphStyle("KEY", parent=B, backColor=colors.HexColor("#e8f1fb"), borderColor=NAVY, borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=8)
P = lambda t, s=B: Paragraph(t, s)

def size(rel):
    p = os.path.join(HERE, rel)
    if os.path.isdir(p):
        n = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    elif os.path.exists(p):
        n = os.path.getsize(p)
    else:
        return "missing"
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def table(head, rows, widths, color=NAVY):
    data = [[P(c, CH) for c in head]] + [[P(c, C) for c in r] for r in rows]
    t = Table(data, colWidths=[w*mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),color), ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#b8c2cc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f5f8")]), ("VALIGN",(0,0),(-1,-1),"TOP")]))
    return t

HDR = ["File in exome_Scripts", "Copied from", "Size", "What it does"]
W = [52, 50, 16, 62]
s = [P("exome_Scripts: contents", H1),
     P(f"Folder: ~/Desktop/whole_exome_sequencing/exome_Scripts &nbsp;|&nbsp; Updated {datetime.date.today():%d %b %Y}"),
     P("ClawBio exome skills, plus the helper scripts and reference data needed to analyse an exome from a VCF. "
       "Everything was <b>copied</b>; the original analysis folders are unchanged. Run: <b>run_exome.sh &lt;vcf&gt; &lt;output folder&gt;</b>. "
       "Works offline; no patient data is sent anywhere.", KEY)]

s.append(P("1. ClawBio skills (clawbio_skills/)", H2))
s.append(table(["Skill folder","Main script","Size","Used by run_exome.sh?"], [
 ["clinical-variant-reporter", "clinical_variant_reporter.py, acmg_engine.py, self_audit.py", size("clawbio_skills/clinical-variant-reporter"), "Yes, through the local copy in acmg_local/. The original sends variants to the Ensembl web service."],
 ["rare-high-impact-variants", "rare_high_impact_variants.py", size("clawbio_skills/rare-high-impact-variants"), "Yes (offline): rare loss-of-function count."],
 ["variant-annotation", "variant_annotation.py", size("clawbio_skills/variant-annotation"), "No: uses the Ensembl web service; replaced by local VEP."],
 ["vcf-annotator", "vcf_annotator.py", size("clawbio_skills/vcf-annotator"), "No: annotates over the web; replaced by local VEP."],
 ["wes-clinical-report-en", "wes_clinical_report_en.py", size("clawbio_skills/wes-clinical-report-en"), "No: needs a Novogene-style markdown input; marked PROPRIETARY in its metadata."],
 ["clinical-variant-prioritizer", "prioritize.py, api.py", size("clawbio_skills/clinical-variant-prioritizer"), "No: built for array or genome genotype lists (rsID input)."],
], [42, 52, 16, 70]))
s.append(P("Copied from the installed ClawBio package (version 0.7.1) in the exome conda environment. Cache folders were left out.", C))

s.append(P("2. Runner and helper scripts", H2))
s.append(table(["File","Origin","Size","What it does"], [
 ["run_exome.sh", "Written from the sample-2 pipeline", size("run_exome.sh"), "The only command you need: run_exome.sh &lt;vcf&gt; &lt;output folder&gt;. Filter to targets, ClinVar, VEP (offline), ACMG, rare loss-of-function count, ClinVar table. Works with filtered and unfiltered DNAscope VCFs."],
 ["acmg_local/", "ClawBio clinical-variant-reporter, adapted", size("acmg_local"), "ACMG/AMP 28-criteria classifier reading local evidence instead of the Ensembl web service. Rules unchanged."],
 ["helpers/parse_vep.py", "Copied from analysis/", size("helpers/parse_vep.py"), "Flattens VEP output to vep_table.tsv and prepares the rare-variant skill input."],
 ["helpers/build_local_evidence.py", "Copied from analysis/", size("helpers/build_local_evidence.py"), "Builds ACMG evidence from local VEP and ClinVar."],
 ["README.md", "-", size("README.md"), "How to run, outputs, how to update ClinVar."],
 ["environment.yml", "conda env export of 'exome'", size("environment.yml"), "Exact package versions, to rebuild the environment elsewhere."],
 ["make_contents_pdf.py", "-", size("make_contents_pdf.py"), "Builds this PDF."],
], [42, 40, 16, 82]))

s.append(P("3. Reference data", H2))
s.append(table(["File","Origin","Size","What it is"], [
 ["reference/targets.bed", "analysis/targets.bed", size("reference/targets.bed"), "Exome capture targets (3 columns from hg38_exome_comp_spikein_v2.0.2)."],
 ["reference/clinvar_chr.vcf.gz (+ .tbi)", "analysis/", size("reference/clinvar_chr.vcf.gz"), "ClinVar GRCh38, release 29 Sep 2026, chr-prefixed."],
 ["reference/chrmap.txt", "analysis/", size("reference/chrmap.txt"), "Chromosome renaming map, for updating ClinVar."],
], [42, 40, 16, 82]))

s.append(P("4. Removed on request", H2))
s.append(table(["Item", "Note"], [
 ["original_runs/ (exact sample-2 script)", "Sample-specific record; the original is still in analysis2/run_sample.sh."],
 ["trio/ (run_trio.sh, trio_qc.py, trio_analysis.py) and hpo/", "Built for the prenatal trio, with that family's HPO terms. trio_analysis.py and the HPO files are still in analysis_trio/. run_trio.sh and trio_qc.py existed only here and were deleted."],
 ["reference/epi_genes_list.txt, reference/ymsy.bed", "Case-specific (epilepsy gene list; trio sex check). Originals still in analysis/ and analysis_trio/."],
], [70, 110]))

s.append(P("5. Not included", H2))
s.append(table(["Item", "Reason"], [
 ["VEP cache (~/.vep, about 52 GB)", "Too large to duplicate; run_exome.sh uses it in place. README explains how to reinstall."],
 ["PDF report generators", "They were in a temporary folder that has since been cleared."],
 ["Patient results", "Stay in the analysis folders; exome_Scripts holds only tools and reference data."],
], [70, 110]))

s.append(P("6. Test status", H2))
s.append(P(TEST))
s.append(P("Total folder size: " + size(".")))
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=14*mm, bottomMargin=14*mm, title="exome_Scripts contents").build(s)
print(OUT)
