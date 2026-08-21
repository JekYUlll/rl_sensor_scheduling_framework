# Findings

## Audit framework takeaways
- Use source-native extraction first for LaTeX headings, captions, table cells, and figure-generation scripts.
- Use rendered PDF extraction as a second layer to catch generated figure text, float order, and visible labels that are not obvious from TeX source.
- Treat suspected AI-coined terms as review findings, not facts: flag by awkward morphology, internal-only usage, high-visibility placement, and inconsistent variants.
- Produce a report before editing, then use the report as the normalization plan.
