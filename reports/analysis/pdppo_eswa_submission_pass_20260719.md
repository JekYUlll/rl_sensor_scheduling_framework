# PD-PPO ESWA submission consistency pass

Date: 2026-07-19

## Binding manuscript and evidence identity

- Canonical scientific source: `paper/main.tex` and `paper/sections/*.tex`.
- Named build: `paper/main.pdf`.
- Double-anonymized build: `paper/anonymous_manuscript.tex` and
  `paper/anonymous_manuscript.pdf`.
- Separate title page: `paper/title_page.tex` and `paper/title_page.pdf`.
- Current clean evidence uses seeds 117 and 118 for the bounded actor
  architecture choice and seeds 119--140 for the unchanged post-pilot
  expansion.
- The earlier 117/122 pair belongs to the superseded July 10 evidence contract.
  It remains in historical archives but is not a seed boundary for the active
  manuscript.

The frozen architecture decision is
`reports/aggregate/pdppo_clean_method_gate_20260718/clean_candidate_decision.json`.
Both candidates passed both pilot seeds. The plain actor was selected because
the context encoder's mean macro improvement was 0.002891, below the predeclared
0.005 materiality threshold. The decision records the candidate tags, seed set,
truth/evaluator hashes, validation starts, and final starts.

## Manuscript corrections made

1. Added one ordered estimator-update sentence without duplicating the complete
   sample-and-hold specification.
2. Replaced broad `rule-based` wording with AoI-priority, round-robin, and random
   wherever those are the intended conventional comparators.
3. Clarified that matched reward controls substitute the scalar objective inside
   a shared guide/auxiliary training scaffold rather than testing reward-only
   learning from scratch.
4. Added the metric boundary: the claim concerns the prespecified ordinary loss
   and validation-normalized equal-regime macro score and is not asserted to be
   invariant to other target or regime weightings.
5. Added one-source conditional anonymization and a separate title page. The
   full affiliation is the School of Mechanical Engineering's published postal
   address: No. 2 Southeast University Road, Jiangning District, Nanjing 211189,
   China.

## ESWA submission requirements checked

The official ESWA Guide for Authors was checked on 2026-07-19:

- double-anonymized peer review is required;
- title page and anonymized manuscript must be separate;
- acknowledgements must appear only on the title page for review;
- the anonymized manuscript must not contain author names, affiliations, or
  acknowledgements;
- the abstract maximum is 250 words;
- highlights are a separate editable file with 3--5 bullets and at most 85
  characters per bullet.

Source:
`https://www.sciencedirect.com/journal/expert-systems-with-applications/publish/guide-for-authors`

## Verification

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: passed,
  50 pages.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error anonymous_manuscript.tex`:
  passed, 49 pages.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex`:
  passed, 9 pages.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error title_page.tex`: passed,
  2 pages.
- Anonymous PDF text scan: no Yongzhe, Zhuyu, email address, Southeast
  University, CRediT heading, or acknowledgements.
- Anonymous PDF metadata author field: empty.
- Anonymous build dependency audit (`anonymous_manuscript.fls`): none of
  `author_pdf_metadata.tex`, `author_frontmatter.tex`, or
  `author_identity_statements.tex` is loaded. Author identity is therefore
  isolated from both the rendered review PDF and its compiled source path.
- Abstract word count: 230.
- Highlights: five editable lines; every line is at most 85 characters.
- Undefined reference/citation scan: none.
- Active submission-source scan for the historical 117/122 boundary: none.
- `git diff --check` in the framework and paper repositories: passed.

The existing nonfatal BibTeX warnings concern missing page fields for four
conference/preprint references; they do not produce undefined citations.

## Compression and supplementary split

The bounded split has now been applied. It reduces the named manuscript from 67
to 50 pages and the anonymous manuscript from 66 to 49 pages without changing
font size, spacing, or the frozen scientific evidence. The Conclusion starts on
page 41 and the theoretical appendix starts on page 42.

The 9-page `paper/supplementary_material.pdf` now contains the detailed history
indicators and actor fusion, forecaster-fitting quantities, reward-control
recursions, physical-platform rendering, simulator and observation tables,
executable baseline definitions, full-partition replay table, ridge table,
generator diagnostic figure, fixed-mask ledger, and information-access audit.
Exact headline numbers and interpretive anchors remain in the main manuscript.

All affected pages and all supplement pages were rendered and visually checked.
No clipping, overlap, misleading blank space, or separated caption was found.

## Submission package

The local upload-ready package is
`submission/eswa_pdppo_20260719/`. It contains:

- anonymous manuscript, title page, supplement, cover letter, declarations,
  highlights, and four individual vector figures;
- independently compilable anonymous and title-page source archives;
- an anonymous code/evidence archive with core PD-PPO, Double-DQN, simulator,
  collector, aggregate, and focused-test material;
- `checksums.sha256` covering all upload artifacts.

The anonymous source compiles from inside the package. Its PDF author metadata
is empty; PDF text, source dependencies, source archive, evidence archive, and
absolute-path scans contain no author identity. The evidence verifier recovers
the 24-seed primary, continuous-replay, Double-DQN, and ridge result directions.
The complete focused `tests/v2` suite passes in the `darts` environment with one
existing skip.

Figure 1 is generated by the tracked Matplotlib script
`paper/figures/gen_fig_framework_and_support.py`. The active vector PDF contains
no embedded raster image object and is not a generative-image asset.

## Remaining external gate

`anonymous_repository_upload.zip` is prepared but not yet hosted. The current
Data Availability wording remains provisional because a local archive is not an
accessible repository. After uploading it to an anonymous service, insert the
actual URL in the manuscript and declarations, rebuild both manuscript PDFs,
and regenerate the checksums. No further training experiment is required.
