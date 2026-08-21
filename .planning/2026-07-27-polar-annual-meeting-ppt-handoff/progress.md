# Progress Log

## Session: 2026-07-27

### Current Status
- **Phase:** 5 - Delivery complete
- **Started:** 2026-07-27

### Actions Taken
- Audited the canonical paper, current 24-seed aggregate reports, paper figures,
  sensor-system context, and earlier PPT assets.
- Defined a 23-page structure with 20 main pages and 3 backup pages.
- Generated 8 compact CSV files and 24 PNG/PDF assets.
- Created 9 Chinese Markdown handoff documents and a root upload guide.
- Rebuilt the time-split graphic after visual inspection found overlapping
  labels.
- Corrected the event table so losses and confidence intervals share the same
  validation-normalized scale.
- Added a reproducible asset-generation script, checksums, and archive.

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Frozen evidence audit | 24 seeds 117--140 and all headline values match | Assertions passed | pass |
| Event metric audit | Particle/flux/thermal normalized means and wins match paper assets | Assertions passed | pass |
| Package completeness | 9 Markdown, 24 image/vector, 8 CSV files | Present and non-empty | pass |
| Visual audit | No clipping or label overlap in generated Chinese charts | Final contact sheet inspected | pass |
| Font rendering | Chinese plots render without missing-glyph warnings | Clean build log | pass |

### Errors
| Error | Resolution |
|-------|------------|
| Chinese font fallback produced missing-glyph warnings | Registered `/usr/share/fonts/truetype/winfonts/msyh.ttf` explicitly. |
| Narrow timeline segments caused label overlap | Used numbered proportional segments and evenly spaced detail blocks. |
| Event CSV scale mismatch | Divided raw event losses by seed-specific validation normalizers. |
