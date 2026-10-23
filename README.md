# LLM development

Anderson Diagnostics & Labs - AI tooling for clinical genomics.

| Folder | What it is |
|---|---|
| [llm/](llm/) | tinygpt: a small GPT-style language model built from scratch in PyTorch (roadmap track A). See its [README](llm/README.md). |
| [AI-Analyser/](AI-Analyser/) | WES AI Analyser: web tool for clinical whole-exome interpretation (VCF + clinical history -> ACMG-classified, AI-drafted, scientist-approved report). See its [README](AI-Analyser/README.md). |

Patient data (VCFs, `data/`, `.env`) must never be committed; each project's `.gitignore` excludes it.
