"""Build per-variant evidence for the ACMG engine from LOCAL data only:
 - vep_annotated.vcf  (Ensembl VEP 116, offline cache)
 - annotated_clinvar.vcf.gz (public ClinVar file matched locally)
Key format matches the skill: chrom:pos:ref:alt"""
import pysam, json
SEV = ["transcript_ablation","splice_acceptor_variant","splice_donor_variant","stop_gained","frameshift_variant","stop_lost","start_lost",
"transcript_amplification","feature_elongation","feature_truncation","inframe_insertion","inframe_deletion","missense_variant","protein_altering_variant",
"splice_donor_5th_base_variant","splice_region_variant","splice_donor_region_variant","splice_polypyrimidine_tract_variant","incomplete_terminal_codon_variant",
"start_retained_variant","stop_retained_variant","synonymous_variant","coding_sequence_variant","mature_miRNA_variant","5_prime_UTR_variant","3_prime_UTR_variant",
"non_coding_transcript_exon_variant","intron_variant","NMD_transcript_variant","non_coding_transcript_variant","upstream_gene_variant","downstream_gene_variant",
"TFBS_ablation","TFBS_amplification","regulatory_region_variant","intergenic_variant"]
rank = {c:i for i,c in enumerate(SEV)}
def sev(c): return min(rank.get(x,99) for x in c.split("&"))
def top_term(c): return sorted(c.split("&"), key=lambda x: rank.get(x,99))[0]
def veallele(ref, alt):
    if len(ref)==1 and len(alt)==1: return alt
    r,a = ref,alt
    if r[0]==a[0]: r,a = r[1:],a[1:]
    return a if a else "-"
def pred(x): return x.split("(")[0] if x else ""
def fnum(x):
    try: return float(x)
    except: return None

cv = {}
for r in pysam.VariantFile("annotated_clinvar.vcf.gz"):
    if len(r.alts)!=1: continue          # allele-specific ClinVar match is ambiguous at multiallelic sites
    sig = r.info.get("CLNSIG")
    if sig: cv[(r.chrom,r.pos,r.ref,r.alts[0])] = "&".join(sig) if isinstance(sig,tuple) else sig

vf = pysam.VariantFile("vep_annotated.vcf")
ix = {f:i for i,f in enumerate(vf.header.info["CSQ"].description.split("Format:")[1].strip().split("|"))}
out = {}
for r in vf:
    csq = [c.split("|") for c in r.info["CSQ"]]
    for alt in r.alts:
        ents = [c for c in csq if c[ix["Allele"]]==veallele(r.ref,alt)] or csq
        best = sorted(ents, key=lambda c:(sev(c[ix["Consequence"]]), 0 if c[ix["CANONICAL"]]=="YES" else 1))[0]
        cons = top_term(best[ix["Consequence"]])
        e, g, m = fnum(best[ix["gnomADe_AF"]]), fnum(best[ix["gnomADg_AF"]]), fnum(best[ix["MAX_AF"]])
        afs = [x for x in (e,g) if x is not None]
        out[f"{r.chrom}:{r.pos}:{r.ref}:{alt}"] = dict(
            gene=best[ix["SYMBOL"]], consequence=cons, impact=best[ix["IMPACT"]], hgvsc=best[ix["HGVSc"]], hgvsp=best[ix["HGVSp"]],
            transcript=best[ix["Feature"]], clinvar_significance=cv.get((r.chrom,r.pos,r.ref,alt),""), clinvar_review_stars=0,
            gnomad_af=max(afs) if afs else None, gnomad_af_popmax=m, cadd_phred=None,
            sift_prediction=pred(best[ix["SIFT"]]), polyphen_prediction=pred(best[ix["PolyPhen"]]), spliceai_max_delta=None)
json.dump(out, open("acmg_local/local_evidence.json","w"))
print(len(out), "variants;", sum(1 for v in out.values() if v["clinvar_significance"]), "with ClinVar;", sum(1 for v in out.values() if v["gnomad_af"] is None), "absent from gnomAD")
