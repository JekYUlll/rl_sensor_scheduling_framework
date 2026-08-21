# PD-PPO citation revision, 2026-07-22

## Scope

The audit reconciled `docs/07-22-02-citation.md` with the active manuscript
defined by `paper/main.tex` and `paper/sections/*.tex`. The compiled main
bibliography contains 23 entries, matching the report's scope. A broader source
scan initially found five additional equipment-manual keys in inactive or
supplementary table sources; they are not part of `main.bbl`.

## Pre-edit archive

- `paper_archives/2026-07-22-citation-strict-revision/paper_pre_citation_revision_20260722_023550.tar.gz`
  - SHA-256: `7b7c36cea6c3e4bbbdf292b589df4248b8d22f0e30741fcd4329658082626b67`
- `paper_archives/2026-07-22-citation-strict-revision/submission_pre_citation_revision_20260722_023550.tar.gz`
  - SHA-256: `7e14f998a4b5a6f785ec97c8b0ff6078b67f587ae4833a5aa27cef2d8ed0f07c`

## Metadata corrections

- Corrected the Fernández-Bes et al. DOI to
  `10.1109/JSAC.2015.2391792`.
- Added the arXiv DOI `10.48550/arXiv.2601.21482` to Tran et al. and retained
  its explicit preprint status.
- Recast Monrad-Krohn et al. as a PANGAEA dataset with the complete ten-author
  list and DOI `10.1594/PANGAEA.992701`.
- Added pages 864--873 and the complete IEEE INFOCOM venue to Wei and Zheng.
- Added the complete book title, LNCS volume 14950, pages 150--165, and
  publisher to Pendyala et al.
- Added pages 1--8 and the complete ACM proceedings title to Murad et al.
- Expanded the Proceedings of the Royal Society A journal title for Ogbodo et
  al.; its 2026 year and article number remain unchanged.

The high-risk DOI and proceedings metadata were checked against Crossref,
Springer Nature, IEEE, Royal Society, and PANGAEA records. Historical BibTeX
keys were retained so no citation-key churn entered the manuscript.

## Claim-source alignment

- Split the former long objective citation cluster into estimation error or
  covariance, censoring utility, AoI or AoII, tracking or belief-state RL, and
  delay-aware remote-estimation statements.
- Rephrased the active-perception connection as sequential observation
  selection under partial observability.
- Separated DRL precedents for adaptive sensing and sensor steering,
  informative path planning, and delayed-reward constrained optimization.
- Identified TFT, TCN, and iTransformer as candidate forecasting backbones. The
  cited papers are no longer presented as schedule-evaluation methods.
- Replaced the unsupported claim that blowing-snow variables change predictive
  value with source-supported statements about transport, atmosphere--snow
  interactions, and particle microphysics.
- Separated Antarctic AWS statistics from Antarctic drifting-snow evidence,
  numerical modelling, and cold-region particle measurements in the simulator
  description.
- Distinguished Bai et al.'s TCN architecture precedent from this paper's
  three-level, 64-channel, kernel-3, dropout-0.05 implementation.
- Framed the eight generator checks as this paper's acceptance checks, informed
  by statistics-preserving time-series generation, rather than checks taken
  from Aloni et al.

## Validation

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: passed,
  51 pages.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error anonymous_manuscript.tex`:
  passed, 51 pages.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex`:
  passed, 10 pages.
- Citation-key audit: 23 rendered references; no missing keys, undefined
  citations, or undefined references.
- The only BibTeX metadata warning is the absence of conventional page numbers
  for the official ICLR 2024 iTransformer record; no page number was invented.
- Visual review covered manuscript pages 5--7, 21, 27, and 48--51. No clipping,
  overlap, broken glyph, or unreadable reference was found.
- The packaged anonymous PDF is byte-identical to the canonical anonymous PDF.
  The anonymous source bundle compiles independently and has identical extracted
  text.
- All 15 files in `submission/eswa_pdppo_20260719/checksums.sha256` pass.
- Anonymous text and PDF-string scans contain no author names, institutional
  affiliation, or author email addresses.

