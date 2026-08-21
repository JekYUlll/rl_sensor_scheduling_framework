# ESWA Full Rewrite Strategy

Date: 2026-05-25

Note: the filename is historical. The target journal has been updated to
*Expert Systems with Applications* (ESWA), and the rewrite strategy below should
be read as ESWA-facing.

## Submission Target

Journal: *Expert Systems with Applications* (ESWA)  
Article type: Research Article  
Official Guide for Authors:
<https://www.sciencedirect.com/journal/expert-systems-with-applications/publish/guide-for-authors>

The rewritten paper will be an applied intelligent sensing-system study: a
prediction-driven RL scheduler for power-budgeted microclimate monitoring, tested
in a blowing-snow digital-twin benchmark. The Antarctic/cold-region case remains
the motivating application, but the submission framing should emphasize expert
systems, constrained scheduling, forecast-oriented decision support, and
engineering validation rather than cold-region science as the primary
contribution.

## Core Thesis

Corrected chronological final-test experiments establish that an instantaneous
power limit alone does not yield a demonstrated dynamic-over-static advantage:
PD-PPO significantly outperforms dynamic heuristic baselines but is statistically
indistinguishable from a validation-selected static allocation. The remaining
regime-dependent thesis is limited to a mechanism diagnostic: the existing
energy-account oracle reference illustrates a dynamic opportunity in selected
storm windows, while its learned-policy curriculum outputs failed the independence
audit and require corrected retraining before supporting comparative claims.

## Candidate Title

**Prediction-Driven Reinforcement Learning for Power-Budgeted Sensor Scheduling
in Microclimate Digital Twins**

Title rationale:

- Places the intelligent scheduling system and prediction-driven decision problem
  before the application setting.
- Clearly states that the evidence is simulation-based.
- Avoids suggesting field deployment or universal RL superiority.

## Contribution Set

1. A warm-up-aware, forecast-evaluated simulation framework for heterogeneous
   blowing-snow monitoring channels under normalized power constraints, grounded in
   Antarctic statistical anchors and sensor acquisition considerations.
2. A protocol-controlled fixed-budget analysis showing PD-PPO improvement over
   dynamic heuristics but no significant advantage over a validation-selected
   static allocation, followed by a clearly labeled energy-account mechanism
   diagnostic.
3. A reproducible protocol audit demonstrating why adaptive-policy claims must be
   withheld when curriculum, oracle or evaluation windows are not independent.

Excluded contribution language:

- No claim of field validation or operational deployment.
- No claim that normalized energy costs reproduce watt-level consumption.
- No claim that PD-PPO reliably learns event-triggered laser control.
- No claim of robust superiority over AoI.

## New Manuscript Architecture

The existing section prose is not to be edited paragraph-by-paragraph. The new
manuscript will be written afresh around the evidence ledger; verified tables and
data-derived figures may be reused with revised captions.

### Front Matter

- Title and authorship fields.
- Stand-alone abstract, maximum 250 words.
- Up to seven indexing keywords.
- Separate `highlights.txt` or `.tex` file with 3--5 bullets, each <=85 characters.

### 1. Introduction: Intelligent Sensing-System Scheduling Problem (900--1100 words)

- Why power-budgeted heterogeneous sensing creates a practical scheduling problem
  for intelligent monitoring systems.
- Why short-horizon prediction quality, not only instantaneous estimation error,
  should drive scheduling decisions in microclimate monitoring.
- Gap: scheduling studies often report a learned policy comparison without first
  establishing when adaptation is necessary rather than a fixed allocation.
- State research question, hypothesis and three contributions.

### 2. Monitoring System and Simulation Design (1400--1800 words)

#### 2.1 Observation task and sensor channels

- Five motivating instrument families and their logical observation channels.
- Warm-up and availability model.
- Normalized deployment-cost caveat.

#### 2.2 Synthetic Antarctic blowing-snow environment

- Statistical anchors and V3.1 semi-Markov storm/event generator.
- Validation criteria and the minimum credible generator figure/table.
- Explicitly state simulation status and lack of external field validation.

#### 2.3 Forecast-based evaluation signal

- Frozen forecast oracle, target variables, horizon and FW-MAE definition.
- Chronological oracle-pretrain/RL-train/validation/final-test split and no online
  oracle updates; report only outputs generated under this declared protocol.

### 3. Scheduling Under Two Energy Regimes (1100--1500 words)

#### 3.1 Instantaneous budget benchmark

- Feasible actions, startup peaks, warm-up aborts.
- Baselines: static projection, round-robin, AoI, random, full observation as
  unconstrained diagnostic.

#### 3.2 Calibrated energy-account diagnostic

- State-of-charge bookkeeping as a simplified normalized analysis, not a complete
  hardware model.
- Calibration values and storm-window selection.
- Event labels are evaluation/oracle information; operational event inference is
  untested.

#### 3.3 Prediction-driven learned scheduler

- Brief PD-PPO description sufficient to reproduce evaluation.
- Move detailed neural architecture and extended ablations to supporting material
  unless space permits.

### 4. Experimental Results (1500--1900 words)

#### 4.1 Fixed budget: effective allocation is close to static

- Present the completed split-protocol table (`3` budgets by `10` seeds), including
  the validation-selected static comparison and corrected significance conclusion.
- Report mean improvements over heuristic baselines and proximity to the
  validation-selected static comparator.
- Interpret as diagnosis, not as dynamic superiority.

#### 4.2 Energy account: a dynamic opportunity appears in storm windows

- Present oracle diagnostic table.
- Focus on dynamic reference versus static snow core and clipped static laser.

#### 4.3 Learned scheduling: opportunity is partly exploitable

- Do not present the existing five-seed curriculum table as comparative evidence:
  its storm windows are the PPO training windows and its full-distribution
  rollouts overlap training/oracle windows.
- Include a learned energy-account comparison only if a replacement chronological
  split-protocol run completes; otherwise present the dynamic opportunity as an
  oracle/reference-policy mechanism diagnostic and identify learning as future work.

#### 4.4 Algorithm diagnostic evidence

- Compact A1 conclusion only: coupled AWBC/prior evidence; no individual component
  claims unsupported by significance.

### 5. Discussion (900--1200 words)

- Engineering implication: characterize energy regime before selecting adaptive
  scheduling complexity.
- Relation to Antarctic monitoring and event-focused observation.
- Limitations: simulation; normalized costs; oracle/event-label assumptions;
  storm-window selection; small learned-policy sample; simplified energy account;
  data sharing and no field validation.
- Path to deployment: measured power/SOC profiles, always-on event detector,
  independent field sequences.

### 6. Conclusions (200--300 words)

- Directly answer when adaptation helps.
- Repeat only supported numerical findings.
- End with field-validation and physical energy-account requirement.

### End Matter

- CRediT author contribution statement.
- Funding statement.
- Declaration of competing interest.
- Data availability statement satisfying current ESWA/Elsevier requirements or
  stating justified limits.
- Declaration of generative AI use naming tools and human verification.
- Acknowledgements immediately before references.

## Figures and Tables Policy

Retain only figures generated from verified data or manually/typeset diagrams; do not
use AI-generated or AI-edited artwork in the submitted manuscript.

Essential candidate tables:

1. Sensor-channel and normalized energy assumptions.
2. V3.1 environment validation.
3. Fixed-budget `n=10` results.
4. Energy-account storm-window reference diagnostic.
5. Energy-account learned-policy results.

Essential candidate figures:

1. Study workflow and the two evaluated energy regimes, created as LaTeX/vector
   artwork from the study design.
2. Generator/statistical validation figure, subject to provenance confirmation.
3. Power-loss or policy comparison figure derived from locked result CSVs.

Remove or relegate:

- Decorative deployment rendering unless its provenance and rights are clear.
- Long architecture figure unless it materially improves reproducibility.
- Development-probe figures or results that do not support the final question.

## ESWA Compliance Checklist

| Requirement | Planned handling |
| --- | --- |
| Expert/intelligent-system scope | Lead with prediction-driven constrained scheduling for sensing systems; use blowing-snow monitoring as the benchmark application. |
| Editable sources | Deliver LaTeX sources, editable tables and separate figure files. |
| Abstract and highlights | Validate against current ESWA/Elsevier author instructions before final packaging. |
| 1--7 keywords | Include six or fewer plain-English terms. |
| Highlights required | Create separate file; validate 3--5 bullets <=85 characters. |
| Figure policy | Data/vector/manual figures only; no generative-AI-created artwork. |
| Data availability | Deposit/cite generated/result data or state justified sharing restriction. |
| CRediT | Add contribution statement after author confirmation. |
| AI disclosure | Add official-style disclosure before references. |
| References | Use consistent author-year entries with DOI checks where available. |

## Quality Gates

1. Evidence gate: every number in abstract, results and conclusions maps to the
   evidence ledger and a result artifact generated under a stated valid protocol.
   The old V3.1 S2 table is diagnostic only because its window audit fails; main
   fixed-budget claims require split-protocol final-test outputs with a
   validation-selected static comparator. The existing energy-account learned-policy
   table is also diagnostic only because its 2026-05-26 audit fails.
2. Scope gate: independent reviewer agrees the ESWA-facing intelligent scheduling
   problem, not a cold-region science claim alone, drives the manuscript.
3. Compliance gate: abstract/highlights/end matter/data statement/artwork all meet
   official guide requirements.
4. Rendering gate: compiled PDF passes visual review for tables, figures and
   references.
5. Review gate: at least two post-draft independent subagent review cycles have no
   unresolved major findings.

## Review Round 1: Structural Audit Disposition

Independent reviewer `Zeno` returned a "rewrite before review" recommendation. Its
high-priority findings are accepted into the rewrite plan:

- PD-PPO will not be the organizing subject of the title or first argument.
- The protocol and energy-regime definitions must be written before results and
  reconciled directly with implementation artifacts.
- Truth-event use, simplified energy accounting, single-oracle evaluation, absence of
  field validation and the weakness of the current static comparator will be made
  explicit; the static comparator will be strengthened if feasible from saved runs.
- Package-level requirements (short abstract, highlights, CRediT, data and AI
  statements) become mandatory deliverables.
