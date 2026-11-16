"""Synthetic referral notes with the HPO terms a careful reader would code, for scoring
phenotype extraction (bench/run_phenotype.py). No patient data: the notes are written in
the style of real referrals - abbreviations, British spelling, relatives' history, lists
of negated findings, measurements instead of terms.

Expected terms are given by HPO label or synonym and resolved against hp.obo at run time,
so a typo shows up as "unresolved" instead of silently scoring as a miss. `present` is
what the patient has, `excluded` what the note says the patient does NOT have. A term that
only describes a relative belongs in neither list - the patient doesn't have it, and the
note doesn't say they lack it.

`alternatives` lists acceptable codings: extracting a listed alternative instead of the
expected term counts as a hit (e.g. a more specific child term the text also supports).
"""
from __future__ import annotations

CASES = [
    {
        "name": "abbreviations_consanguinity",
        "text": "8 yr old male child, k/c/o GDD and seizures since 6 months of age. Hypotonia noted. "
                "No h/o hearing loss. Born of 3rd degree consanguineous marriage. Elder sibling had "
                "similar complaints and expired at 2 years.",
        "present": ["Global developmental delay", "Seizure", "Hypotonia",
                    "Offspring of consanguineous relationship"],
        "excluded": ["Hearing impairment"],
    },
    {
        "name": "list_negation_measurement",
        "text": "Proband is a 4 year old girl with microcephaly (HC below 3rd centile), intellectual "
                "disability and autistic behaviour. MRI brain showed thin corpus callosum. No seizures "
                "or hypotonia.",
        "present": ["Microcephaly", "Intellectual disability", "Autistic behavior", "Thin corpus callosum"],
        "excluded": ["Seizure", "Hypotonia"],
    },
    {
        "name": "measurement_only",
        "text": "Referred for growth evaluation. Height below 3rd centile, head circumference less than "
                "2nd centile. Bone age delayed. Father is short.",
        "present": ["Short stature", "Microcephaly", "Delayed skeletal maturation"],
        "excluded": [],
    },
    {
        "name": "metabolic_infant",
        "text": "Baby of 3 months with hepatosplenomegaly, jaundice and failure to thrive. "
                "Cherry red spot on fundus examination.",
        "present": ["Hepatosplenomegaly", "Jaundice", "Failure to thrive", "Macular cherry red spot"],
        "excluded": [],
    },
    {
        "name": "british_spelling",
        "text": "Recurrent chest infections, steatorrhoea and poor weight gain since infancy. "
                "Sweat chloride elevated. Older brother has asthma.",
        "present": ["Recurrent respiratory infections", "Steatorrhea", "Elevated sweat chloride",
                    "Failure to thrive"],
        "excluded": [],
        "alternatives": {"Failure to thrive": ["Poor weight gain"]},
    },
    {
        "name": "relative_via_maternal_uncle",
        "text": "Progressive proximal muscle weakness since age 5, calf hypertrophy, CPK 12000 U/L. "
                "Maternal uncle had the same illness and was wheelchair-bound by 12 with cardiomyopathy.",
        "present": ["Proximal muscle weakness", "Calf muscle hypertrophy",
                    "Elevated circulating creatine kinase activity"],
        "excluded": [],
    },
    {
        "name": "trailing_negation",
        "text": "Bilateral sensorineural hearing loss detected on newborn screening. Seizures: absent. "
                "Visual impairment not present.",
        "present": ["Sensorineural hearing impairment"],
        "excluded": ["Seizure", "Visual impairment"],
    },
    {
        "name": "relatives_and_denial",
        "text": "Child with ataxia and nystagmus. Denies headache. MRI: cerebellar atrophy. "
                "Sister has epilepsy, mother has diabetes.",
        "present": ["Ataxia", "Nystagmus", "Cerebellar atrophy"],
        "excluded": ["Headache"],
    },
    {
        "name": "marfanoid",
        "text": "Tall stature with arachnodactyly and lens subluxation. Echo: aortic root dilatation. "
                "No h/o retinal detachment.",
        "present": ["Tall stature", "Arachnodactyly", "Ectopia lentis", "Aortic root aneurysm"],
        "excluded": ["Retinal detachment"],
        "alternatives": {"Ectopia lentis": ["Lens subluxation"],
                         "Aortic root aneurysm": ["Dilatation of the ascending aorta"]},
    },
    {
        "name": "colloquial_terms",
        "text": "Child presents with frequent fits, global delay, coarse facies and hepatomegaly.",
        "present": ["Seizure", "Global developmental delay", "Coarse facial features", "Hepatomegaly"],
        "excluded": [],
    },
    {
        "name": "renal_fh_abbrev",
        "text": "Known case of steroid resistant nephrotic syndrome. FH: maternal grandmother had "
                "kidney failure.",
        "present": ["Nephrotic syndrome"],
        "excluded": [],
    },
    {
        "name": "abbreviation_cluster",
        "text": "Antenatal h/o IUGR. Now 2 years with FTT, HSM and SNHL. No dysmorphism.",
        "present": ["Intrauterine growth retardation", "Failure to thrive", "Hepatosplenomegaly",
                    "Sensorineural hearing impairment"],
        "excluded": [],
    },
    {
        "name": "negation_does_not_leak",
        "text": "No family history of similar illness. Patient has seizures and spasticity, but no "
                "regression. Ataxia is present.",
        "present": ["Seizure", "Spasticity", "Ataxia"],
        "excluded": ["Developmental regression"],
    },
]

# Held-out notes: written AFTER the extractor changes and not tuned against. A score here
# is the honest one; the cases above were in view while the rules were written.
HELD_OUT = [
    {
        "name": "ho_neuro_infant",
        "text": "10 month old female infant with infantile spasms and hypsarrhythmia on EEG. Poor eye "
                "contact. Not sitting without support. No organomegaly. Paternal aunt has intellectual "
                "disability.",
        "present": ["Infantile spasms", "Hypsarrhythmia", "Poor eye contact"],
        "excluded": ["Hepatosplenomegaly"],
        "alternatives": {"Hepatosplenomegaly": ["Hepatomegaly", "Splenomegaly", "Organomegaly"]},
    },
    {
        "name": "ho_skeletal",
        "text": "Disproportionate short stature with rhizomelic shortening of limbs, frontal bossing "
                "and depressed nasal bridge. Hearing normal. Parents non-consanguineous.",
        "present": ["Disproportionate short stature", "Rhizomelia", "Frontal bossing",
                    "Depressed nasal bridge"],
        "excluded": ["Offspring of consanguineous relationship"],
    },
    {
        "name": "ho_cardio",
        "text": "Adolescent with hypertrophic cardiomyopathy on echo and recurrent syncope. "
                "Father died suddenly at 40. ECG showed left ventricular hypertrophy. No arrhythmia documented.",
        "present": ["Hypertrophic cardiomyopathy", "Syncope", "Left ventricular hypertrophy"],
        "excluded": ["Arrhythmia"],
    },
    {
        "name": "ho_renal_eye",
        "text": "Haematuria and proteinuria since age 6, progressive sensorineural deafness, and "
                "anterior lenticonus. Maternal grandfather required dialysis.",
        "present": ["Hematuria", "Proteinuria", "Sensorineural hearing impairment", "Anterior lenticonus"],
        "excluded": [],
    },
    {
        "name": "ho_metabolic_abbrev",
        "text": "Recurrent vomiting, lethargy and hypoglycaemia during illness. Hepatomegaly present. "
                "Ammonia raised. No developmental concerns.",
        "present": ["Vomiting", "Lethargy", "Hypoglycemia", "Hepatomegaly", "Hyperammonemia"],
        "excluded": [],
    },
    {
        "name": "ho_skin_neuro",
        "text": "Multiple cafe-au-lait macules, axillary freckling and two cutaneous neurofibromas. "
                "Learning difficulties at school. Mother has similar skin lesions.",
        "present": ["Cafe-au-lait spot", "Axillary freckling", "Neurofibroma",
                    "Specific learning disability"],
        "excluded": [],
        "alternatives": {"Specific learning disability": ["Intellectual disability"]},
    },
    {
        "name": "ho_muscle_negations",
        "text": "Delayed motor milestones, waddling gait and Gowers sign positive. Intelligence normal, "
                "no contractures, no cardiac involvement.",
        "present": ["Delayed gross motor development", "Waddling gait", "Gowers sign"],
        "excluded": ["Joint contracture"],
        "alternatives": {"Joint contracture": ["Contracture"]},
    },
    {
        "name": "ho_dysmorphic",
        "text": "Hypertelorism, low-set ears, webbed neck and pulmonary valve stenosis. Height on 2nd "
                "centile. Cryptorchidism noted. Sibling is normal.",
        "present": ["Hypertelorism", "Low-set ears", "Webbed neck", "Pulmonic stenosis",
                    "Short stature", "Cryptorchidism"],
        "excluded": [],
        "alternatives": {"Pulmonic stenosis": ["Pulmonary valve stenosis"]},
    },
]
