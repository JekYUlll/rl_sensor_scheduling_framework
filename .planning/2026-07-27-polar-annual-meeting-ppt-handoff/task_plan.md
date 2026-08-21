# Task Plan: Polar Annual Meeting PPT Handoff Package

## Goal
Create a self-contained upload package for a web-based ChatGPT to prepare a
15-minute, 22--23-slide Chinese academic talk on polar sensor scheduling,
grounded in the current PD-PPO manuscript and verified experiment artifacts.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements and evidence discovery
- [x] Audit the canonical manuscript, current figures, tables, aggregate reports,
  sensor-system context, and available user-owned visual material.
- [x] Separate verified headline evidence from diagnostics, limitations, and
  historical experiments that should not enter the talk.
- [x] Record the audience, speaker, timing, and deliverable constraints.
- **Status:** completed

### Phase 2: Talk structure and asset selection
- [x] Define a 22--23-slide Chinese story suitable for a polar-science annual
  meeting and a 15-minute delivery by the user's supervisor.
- [x] Select a minimal set of high-resolution figures, diagrams, photographs,
  and compact CSV evidence objects.
- [x] Define consistent Chinese terminology and claim boundaries.
- **Status:** completed

### Phase 3: Build the handoff package
- [x] Create concise Markdown files for the talk brief, slide storyboard,
  scientific facts, method explanation, speaker notes, Q&A, and web-ChatGPT
  instructions.
- [x] Export or copy required figures in upload-friendly PNG/PDF formats.
- [x] Create compact data tables that support every numerical statement.
- [x] Add an asset manifest and source/provenance map.
- **Status:** completed

### Phase 4: Verification
- [x] Check every claim and number against the manuscript or aggregate results.
- [x] Inspect all selected images for legibility, clipping, and duplication.
- [x] Verify that the package is self-contained and excludes irrelevant,
  internal, or superseded artifacts.
- **Status:** completed

### Phase 5: Delivery
- [x] Produce a compact archive and a recommended upload order.
- [x] Report package path, contents, and any optional material.
- **Status:** completed

## Decisions Made
| Decision | Rationale |
|---|---|
| Create a new isolated plan | The talk handoff is separate from manuscript submission work. |
| Use the current English manuscript and frozen aggregate evidence as authoritative | Historical branches and older report names contain superseded claims. |
| Provide source assets and structured context instead of generating the final PPT | The user has already supplied the paper and style reference to web ChatGPT. |
| Use 20 main slides plus 3 backup slides | This fits a 15-minute talk without rushing 23 narrated pages. |
| Regenerate standalone Chinese charts from aggregate CSVs | Manuscript composites are too dense for presentation use. |

## Errors Encountered
| Error | Resolution |
|---|---|
| Matplotlib selected an inconsistent Chinese font face | Registered the Microsoft YaHei font file explicitly before rendering. |
| Validation and test labels overlapped in the proportional timeline | Reworked the figure into a proportional numbered bar with four evenly spaced description blocks. |
| Initial event CSV mixed raw loss and normalized confidence intervals | Recomputed every event loss with its validation normalizer so values, margins, and confidence intervals use one scale. |
