# Findings & Decisions

## Requirements
- Deliverable is supporting material for a web-based ChatGPT, not the final PPT.
- Talk language is Chinese; duration is 15 minutes; target length is 22--23
  slides; presenter is the user's supervisor.
- The user has already uploaded the current paper PDF and a style-reference PPT.
- The package must add missing context, editable facts/data, and high-resolution
  visuals without duplicating the full paper.

## Research Findings
- The current headline evidence is the fixed six-channel, one-optional-channel
  simulation over seeds 117--140. PD-PPO reduces mean forecast loss by 10.1%
  and macro-averaged normalized loss by 7.9% versus the validation-selected
  fixed schedule, with lower loss in 24/24 seeds.
- The six logical channels are one mandatory core weather channel and five
  optional channels. At most one optional channel is active; event-specific
  channels are the infrared surface-temperature sensor, laser disdrometer, and
  FC4 blowing-snow flux sensor.
- Event-specific channel selection is a strong talk-level mechanism result:
  laser 99.3% in particle windows, FC4 97.8% in flux windows, and infrared
  temperature 99.1% in thermal windows.
- The user-owned AWS rendering is well suited to the polar-domain motivation
  section and clearly labels the principal physical instruments.
- The paper's combined main-evidence PNG is unsuitable as a slide asset without
  editing: the panel headings and summary annotations overlap heavily at its
  native layout. The PPT package should contain simpler presentation-specific
  charts generated from the verified aggregate data.
- The combined mechanism PNG is also manuscript-dense: long English sensor
  names collide below the heatmap and the three panels compete for space. A
  Chinese event-to-channel heatmap should be regenerated as a standalone slide
  asset.
- The current framework PNG is stale relative to the repaired vector PDF.
  Presentation assets must be rasterized from
  `figure_pdppo_framework_drawio.pdf`, not copied from the existing PNG.
- The synthetic-statistics figure is legible and scientifically useful for one
  optional validation slide, but it should not occupy main-story time unless the
  audience asks how the simulation was calibrated.
- The generic Chinese sensor-system block diagram in
  `/home/horeb/Documents/Academic/传感器图示.drawio-new.png` is low resolution,
  predates the current six-channel model, and should not be included.
- The most reliable physical-system visual remains the Blender AWS rendering;
  it already labels the weather station, radiometer, infrared sensor, laser
  disdrometer, and FC4 sensor and is directly aligned with the paper.
- Earlier internal PPTs provide useful institutional context but contain
  superseded forecasting experiments and generic sensor procurement material.
  They should inform the talk opening only; their old LSTM/TFT results must not
  be mixed with the current PD-PPO evidence.
- The supervisor's 2024 polar public-talk deck establishes a useful broad
  context: Southeast University has a longer polar research line that includes
  energy-support and sensing platforms. This can support one optional opening
  sentence, but the new talk should remain centered on sensor scheduling.
- A 23-page file is best organized as 19 substantive slides, one Q&A slide, and
  three backup slides. This keeps the spoken sequence within roughly 14 minutes
  and preserves technical details for questions.
- The first compact event table combined raw event losses with normalized
  confidence intervals. The final package recomputes particle, flux, and thermal
  losses with the seed-specific validation normalizers; the corrected means are
  1.0394/1.0678, 0.7535/0.8706, and 1.0285/1.1233 for PD-PPO/fixed.
- The proportional time-split graphic required a separate four-column
  description row because the validation and test segments are too narrow for
  full labels inside the bar.

## Technical Decisions
| Decision | Rationale |
|---|---|
| Use 18--20 main slides plus 3--5 backup slides within a 22--23 page file | A literal 23-slide narration is too dense for 15 minutes; backup pages preserve requested length without forcing rushed delivery. |
| Create presentation-specific charts from frozen aggregate CSVs | Paper figures are optimized for manuscript width and some contain dense labels. |
| Lead with polar observation needs and hardware, then introduce RL | The annual-meeting audience is domain-first, unlike the ESWA manuscript audience. |
| Keep warning-rule, true-label, reward-proxy, and metric-boundary results in backup/Q&A | They are important scientific boundaries but interrupt the main 15-minute story. |
| Treat the English framework figure as a redraw reference | It is current and accurate but too dense and English-heavy for the Chinese annual-meeting slides. |

## Issues Encountered
| Issue | Resolution |
|---|---|
| Repository contains many superseded report and figure variants | Use current manuscript inputs and `pdppo_*_20260718` aggregate directories as the authoritative evidence set. |

## Resources
- `paper/main.pdf`
- `paper/figures/aws_deployment.png`
- `paper/figures/figure_pdppo_framework_drawio.pdf`
- `paper/figures/proposition_dynamic_value_standalone.pdf`
- `reports/aggregate/pdppo_clean_validation_frozen_24seed_20260718/`
- `reports/aggregate/pdppo_clean_mechanism_24seed_20260718/`
- `reports/aggregate/pdppo_framework_baselines_clean_24seed_20260718/`
