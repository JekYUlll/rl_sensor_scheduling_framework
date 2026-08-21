# 2026-07-02 Reference Audit Findings

## Extraction summary

- Active included TeX files resolved from `paper/main.tex`: 26.
- BibTeX entries in `paper/references.bib`: 38.
- Corrected active citation parser to handle multi-line citation clusters.
- Active citation keys used by the compiled manuscript: 24.
- Unused BibTeX entries: 14. These are not printed by `elsarticle-harv` unless cited or `\\nocite` is added; treat separately from active reference-list audit.
- Missing active citation keys: 0.

Active keys:
`AlAhdab2025`, `Alali2024`, `Aloni2024`, `Amory2020`, `AntAWS2023`, `Bai2018`, `Bajcsy2018`, `FernandezBes2015`, `Golovin2011`, `Jonah2026`, `Kaul_2012`, `Lauri_2023`, `Lim2021`, `Liu2024`, `Monrad2026`, `Murad2020`, `Ogbodo2025`, `Pendyala2024`, `Qu2022`, `Schulman2017`, `Sharma2023`, `Shi_2011`, `Tran2026`, `Wei2020`.

Unused BibTeX keys:
`VanHasselt2016`, `Wang2016`, `DeLaFuente2024`, `Ibrahim2024`, `Ying2022`, `Chen2026`, `Wang2021`, `Ding2025`, `Wang2025`, `OTT2022`, `IAV2024`, `GillGMX500`, `SensecaLPS10`, `ApogeeSI111`.

## Confirmed issues

- `FernandezBes2015`: current DOI `10.1109/JSAC.2015.2430512` returned DOI Not Found. IEEE Xplore record exists at `https://ieeexplore.ieee.org/document/7009961`. Fix/remove DOI before ESWA submission.
- `Golovin2011`: DOI.org lookup failed for `10.1613/jair.3278`, although JAIR page displays that DOI. Add JAIR/arXiv URL; omit DOI if resolver remains broken.
- `Murad2020`, `Wei2020`, `Pendyala2024`: pages missing in local BibTeX; official pages provide `1--8`, `864--873`, `150--165` respectively.
- `Monrad2026`: PANGAEA dataset is Arctic/Ny-Ålesund, not Antarctic; use only for particle size/flux measurement evidence and format as dataset citation.
- `Pendyala2024`: real PPO optimization paper but weak support for sensor scheduling; remove or reword local sentence if retained.
- `Schulman2017`: supports PPO, not action masking. Recommended addition: Huang & Ontañón 2022 invalid-action masking paper, DOI `10.32473/flairs.v35i.130584`.

## ESWA-specific findings

- ESWA uses APA 7 style, requires citation/reference-list consistency, encourages DOI links, warns that incorrect surnames/titles/years/pagination can prevent link creation, and requires data references to include repository/year/persistent identifier plus `[dataset]` marking.
- ESWA preprint rule: mark preprints clearly and include preprint DOI; if a peer-reviewed version exists, cite the formal publication.
- ESWA research-data policy Option C requires deposit, citation/linking, or a statement explaining why data cannot be shared.

## Final output

Final Chinese report written to `docs/07-02-02-reference-audit.md`.
