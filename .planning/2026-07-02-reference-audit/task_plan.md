# 2026-07-02 Reference Audit Plan

Goal: 对 active ESWA manuscript 的所有 active references 做联网校对，输出逐条中文报告：真实文档、引用上下文、原文证据、支持性判断、格式/作者校对、ESWA 适配性、增删建议。

Scope:
- Canonical entry point: `paper/main.tex`.
- Included sources only: recursively resolve `\\input`/`\\include` from `paper/main.tex`.
- Bibliography: `paper/references.bib`.
- Exclude historical drafts, raw translations, archives, and non-included files unless a key is active in `main.tex`.

Phases:
1. [complete] Extract active citation keys, citation contexts, and BibTeX metadata.
2. [complete] Query online metadata sources for every active key: DOI/Crossref, Semantic Scholar/OpenAlex, arXiv/OpenReview/PMLR/official manuals where appropriate.
3. [complete] Gather source evidence quotes that can support or contradict the local citation context. Prefer official abstract/full-text lines; record if only metadata/abstract is available.
4. [complete] Judge each reference: supports claim? metadata correct? author format correct? ESWA relevance? keep/fix/remove/add?
5. [complete] Write detailed Chinese report under `docs/07-02-02-reference-audit.md` plus machine-readable evidence under this planning directory.

Outputs:
- Final report: `docs/07-02-02-reference-audit.md`.
- Machine-readable citation extraction: `.planning/2026-07-02-reference-audit/citation_extraction.json`.
- Online metadata: `.planning/2026-07-02-reference-audit/online_metadata_raw_24.json` and `online_metadata_summary_24.json`.

Quality rules satisfied:
- No invented source quotes; official abstracts/pages used where available, with limitations noted.
- Severe metadata errors separated from style-level ESWA formatting issues.
- No bibliography edits were applied; report is advisory only.
