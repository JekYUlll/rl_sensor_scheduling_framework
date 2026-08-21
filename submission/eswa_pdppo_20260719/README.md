# ESWA submission package

This directory is the local, upload-ready package for the manuscript
"Reinforcement learning with forecast-loss rewards for sensor scheduling under power
constraints."

## Review files

- `anonymous_manuscript.pdf`: double-anonymized review manuscript.
- `supplementary_material.pdf`: anonymous, self-contained supplementary file.
- `highlights.txt`: five editable highlights, each at most 85 characters.
- `figures/`: individual vector PDF artwork used in the main manuscript.
- `anonymous_source_bundle.zip`: LaTeX source required to compile the anonymous
  manuscript and supplement.

## Files submitted outside the anonymous manuscript

- `title_page.pdf`: author names, affiliations, correspondence, CRediT, and
  acknowledgements.
- `cover_letter.pdf`: editor-facing cover letter.
- `declarations.txt`: editable funding, competing-interest, data-availability,
  and generative-AI declarations.
- `title_page_source.zip`: editable source for the title page and cover letter.

## Anonymous evidence archive

`anonymous_repository_upload.zip` contains the core scheduler implementation,
training and evaluation scripts, tests, seed-level aggregate results, and an automated aggregate
check. It is the anonymous review version of the repository that will be made
publicly available upon acceptance. No public URL is claimed during
double-anonymous review.

## Build commands

From the unpacked anonymous source bundle:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error anonymous_manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

The title page and cover letter are intentionally kept outside the anonymous
source bundle.
