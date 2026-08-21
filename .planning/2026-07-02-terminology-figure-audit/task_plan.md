# 2026-07-02 Terminology, Figure, and Title Audit

Goal: audit the active ESWA manuscript for chart/figure/table/title terminology that reads as internal experiment jargon, suspected AI-coined wording, unexplained abbreviations, inconsistent variants, or awkward proprietary/instrument naming; write a new report; then apply claim-preserving refinements and verify the compiled PDF.

Canonical manuscript: `paper/main.tex`
Canonical PDF: `paper/main.pdf`
Deep research framework: `docs/07-01-02-deep-research-report.md`

## Phases
- [x] Phase 1 — Extract active included sources, headings, captions, table titles, table text, figure text, and PDF page/figure ordering.
- [x] Phase 2 — Build a term evidence store and write `docs/07-02-01-terminology-figure-title-audit.md`.
- [ ] Phase 3 — Back up the current PDF/source and apply the report's high-priority terminology/title/caption fixes.
- [ ] Phase 4 — Rebuild `paper/main.pdf`, run source/PDF residual scans, and inspect log/citation status.
- [ ] Phase 5 — Update planning/progress/findings and report exact outputs.

## Guardrails
- Preserve scientific claims, numbers, table meanings, and benchmark boundaries.
- Edit only active included manuscript sources, captions, tables, and figure-generation scripts if necessary.
- Do not count historical `raw.tex`, `rewrite_sections/`, or `_archive/` as active residue unless included by `main.tex`.
- Keep official names such as AntAWS and FC4 only when they are needed and first-use rules are clear.
