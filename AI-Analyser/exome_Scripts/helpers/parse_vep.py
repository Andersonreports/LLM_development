import pysam, csv, re
SEV = ["transcript_ablation","splice_acceptor_variant","splice_donor_variant","stop_gained","frameshift_variant","stop_lost","start_lost",
"transcript_amplification","feature_elongation","feature_truncation","inframe_insertion","inframe_deletion","missense_variant","protein_altering_variant",
"splice_donor_5th_base_variant","splice_region_variant","splice_donor_region_variant","splice_polypyrimidine_tract_variant","incomplete_terminal_codon_variant",
"start_retained_variant","stop_retained_variant","synonymous_variant","coding_sequence_variant","mature_miRNA_variant","5_prime_UTR_variant","3_prime_UTR_variant",
"non_coding_transcript_exon_variant","intron_variant","NMD_transcript_variant","non_coding_transcript_variant","upstream_gene_variant","downstream_gene_variant",
"TFBS_ablation","TFBS_amplification","regulatory_region_variant","intergenic_variant"]
rank = {c:i for i,c in enumerate(SEV)}
vf = pysam.VariantFile("vep_annotated.vcf")
desc = vf.header.info["CSQ"].description
fields = desc.split("Format:")[1].strip().split("|")
ix = {f:i for i,f in enumerate(fields)}
print(fields)
rows=[]; out=open("for_rare_skill.vcf","w")
h=str(vf.header).rstrip("\n").split("\n")
h.insert(-1,'##INFO=<ID=MC,Number=.,Type=String,Description="Molecular consequence (from VEP)">')
h.insert(-1,'##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="Max gnomAD AF (VEP MAX_AF)">')
h.insert(-1,'##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">')
out.write("\n".join(h)+"\n")
for r in vf:
    csq=[c.split("|") for c in r.info["CSQ"]]
    # per ALT allele choose best entry
    for ai,alt in enumerate(r.alts,1):
        ents=[c for c in csq if c[ix["Allele"]]==alt] or csq
        def key(c):
            cons=c[ix["Consequence"]].split("&")
            return (0 if c[ix["CANONICAL"]]=="YES" else 1, min(rank.get(x,99) for x in cons))
        # most severe consequence first, prefer canonical among ties
        best=sorted(ents,key=lambda c:(min(rank.get(x,99) for x in c[ix["Consequence"]].split("&")), 0 if c[ix["CANONICAL"]]=="YES" else 1))[0]
        s=r.samples[0]; gt=s["GT"]; ad=s["AD"]
        maf=best[ix["MAX_AF"]] if "MAX_AF" in ix else ""
        rows.append(dict(chrom=r.chrom,pos=r.pos,ref=r.ref,alt=alt,gene=best[ix["SYMBOL"]],consequence=best[ix["Consequence"]],
            impact=best[ix["IMPACT"]],biotype=best[ix["BIOTYPE"]],sift=best[ix["SIFT"]],polyphen=best[ix["PolyPhen"]],
            max_af=maf,gnomADe_AF=best[ix["gnomADe_AF"]],gnomADg_AF=best[ix["gnomADg_AF"]],existing=best[ix["Existing_variation"]],hgvsc=best[ix["HGVSc"]],hgvsp=best[ix["HGVSp"]],exon=best[ix["EXON"]],clin_sig=best[ix["CLIN_SIG"]],
            gt="/".join("." if g is None else str(g) for g in gt),dp=s["DP"],ad=",".join(map(str,ad)),gq=s["GQ"]))
        info=r.info; 
        line=[r.chrom,str(r.pos),".",r.ref,alt,"50","PASS","MC=SO:0|%s%s%s"%(best[ix["Consequence"]].split("&")[0],"" if not maf else ";gnomAD_AF="+maf,";GENE="+best[ix["SYMBOL"]] if best[ix["SYMBOL"]] else ""),
              "GT"," / ".join([])] 
        line[8]="GT"; line[9]="/".join("." if g is None else str(g) for g in gt)
        out.write("\t".join(line)+"\n")
out.close()
w=csv.DictWriter(open("vep_table.tsv","w"),fieldnames=rows[0].keys(),delimiter="\t"); w.writeheader(); w.writerows(rows)
print(len(rows))
