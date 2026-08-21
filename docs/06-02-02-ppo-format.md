# Writing and Format Patches for the PD-PPO Paper

This document provides concrete, directly applicable patches organized by category. Each patch is formatted as a `FIND → REPLACE` block suitable for Codex or manual substitution. Issues are grouped by type: (A) de-AI-ification / register, (B) sentence-level clarity, (C) structural/format, (D) terminology consistency.

---

## A. De-AI-ification and Register

AI-generated prose tends to produce (i) hollow meta-commentary ("This study asks…", "The central question of this paper is…"), (ii) over-hedged throat-clearing before the actual claim, (iii) symmetrical sentence pairs that mirror each other without adding content, and (iv) the word "therefore" used as a filler connective. The patches below replace these patterns with direct, author-voiced academic prose.

---

### A-1 Abstract — opening sentence

**Problem:** "This study asks when adaptive scheduling is useful for such stations." — This is a meta-description of the paper's intent, not a statement of the problem. It reads as AI-generated framing.

```
FIND:
Energy-limited Antarctic automatic weather stations cannot always power every
sensor needed for blowing-snow monitoring. This study asks when adaptive
scheduling is useful for such stations.

REPLACE:
Energy-limited Antarctic automatic weather stations must allocate sensing
capacity across heterogeneous instruments whose power demands, warm-up
latencies, and forecast relevance differ substantially. Whether adaptive
scheduling improves on a well-chosen fixed allocation depends on the energy
model, and that dependence has not been characterised for blowing-snow
monitoring applications.
```

---

### A-2 Abstract — "The results provide a regime map"

**Problem:** "The results provide a regime map for cold-region sensor scheduling" — "provide a regime map" is a hollow AI phrase. The actual finding should be stated directly.

```
FIND:
The results provide a regime map for cold-region sensor scheduling:
instantaneous power limits favor compact fixed allocations, whereas energy
storage can create adaptive value around blowing-snow events.

REPLACE:
These findings delineate the conditions under which adaptive scheduling
adds value: instantaneous power limits favour compact fixed allocations,
whereas intertemporal energy storage can make event-triggered activation
of high-cost particle and flux channels worthwhile.
```

---

### A-3 §1 Introduction — "This creates a sensor-scheduling question"

**Problem:** The paragraph beginning "This creates a sensor-scheduling question rather than only a forecasting question" is well-structured but contains two AI-typical mirror sentences at the end ("Conversely, a static subset can be operationally simple and may be close to optimal if the same low-cost channels explain most forecast targets.") that add no new information after the preceding sentence already made the point.

```
FIND:
Conversely, a static subset can be operationally simple and may be close
to optimal if the same low-cost channels explain most forecast targets.

REPLACE:
Whether a static subset is competitive depends on how much of the forecast
variance is explained by the low-cost channels that remain permanently
active — a question this paper answers empirically.
```

---

### A-4 §1 — Research question block

**Problem:** The indented research question is well-placed, but the sentence immediately following it ("We answer this question with a prediction-driven scheduling framework.") is a hollow AI transition. The paragraph should move directly into the methodological description.

```
FIND:
We answer this question with a prediction-driven scheduling framework. A
frozen forecast oracle is trained before policy learning and is then used to score
the forecast consequence of a scheduler's estimated state.

REPLACE:
A frozen forecast oracle is trained before policy learning and scores the
forecast consequence of each scheduler's estimated state.
```

---

### A-5 §1 — "The resulting conclusion is a regime map"

**Problem:** "The resulting conclusion is a regime map for monitoring-system design." — Again the "regime map" phrase used as a meta-label before the actual content. The content that follows is good; the label sentence should be cut.

```
FIND:
The resulting conclusion is a regime map for monitoring-system design. Under
instantaneous normalized power budgets, compact fixed allocations capture
much of the available forecast value, while the learned scheduler mainly improves
over switching heuristics.

REPLACE:
Under instantaneous normalized power budgets, compact fixed allocations
capture much of the available forecast value, while the learned scheduler
improves primarily over switching heuristics.
```

---

### A-6 §1 — Contributions list preamble

**Problem:** "This study makes three contributions." — Hollow AI preamble. The numbered list should follow a more informative lead sentence.

```
FIND:
This study makes three contributions.

REPLACE:
Three contributions follow from this framing.
```

---

### A-7 §1 — Post-contributions paragraph

**Problem:** "The study therefore identifies design conditions rather than presenting adaptive control as universally beneficial:" — This sentence restates what the contributions already said. It is AI-typical redundant summary.

```
FIND:
The study therefore identifies design conditions rather than presenting adaptive
control as universally beneficial: instantaneous budgets favor compact fixed
allocations, whereas energy storage can create adaptive value around blowing-snow
events.

REPLACE:
[DELETE — content already conveyed by contribution 3 and the preceding paragraph.]
```

---

### A-8 §2.3 — "Our scheduler uses PPO as a tested policy optimizer"

**Problem:** "Our scheduler uses PPO as a tested policy optimizer rather than as the main scientific novelty." — The phrase "tested policy optimizer" is awkward and AI-typical. The intended meaning is that PPO is used as a standard algorithmic component, not as a contribution.

```
FIND:
Our scheduler uses PPO as a tested policy optimizer rather than as the main
scientific novelty. The manuscript focuses on what the learned policy reveals
about the monitoring regime.

REPLACE:
PPO serves here as a standard policy-gradient algorithm; the scientific
contribution lies in what the learned policy reveals about the monitoring
regime, not in the algorithm itself.
```

---

### A-9 §2.4 — "The same design also creates a responsibility"

**Problem:** "The same design also creates a responsibility." — Anthropomorphising the design. This is AI-typical rhetorical framing.

```
FIND:
The same design also creates a responsibility. Oracle pretraining, policy
training, validation, and final testing must be separated.

REPLACE:
This design imposes a strict data-separation requirement: oracle pretraining,
policy training, validation, and final testing must use disjoint temporal
partitions.
```

---

### A-10 §4.3 — "This separation is the main reason"

**Problem:** "This separation is the main reason the reported fixed-budget results differ from earlier development diagnostics." — Vague. Should state what the difference is.

```
FIND:
This separation is the main reason the reported fixed-budget results differ from
earlier development diagnostics. The result table in Table 3 uses only the final
segment shown in Figure 3.

REPLACE:
Because earlier development diagnostics sampled evaluation windows from
the same sequence used for training, the corrected protocol yields more
conservative estimates of PD-PPO advantage. Table 3 reports only final-test
results from the held-out segment shown in Figure 3.
```

---

### A-11 §6.3 — "This audit is not a side issue"

**Problem:** "This audit is not a side issue. It changes the scientific claim." — Emphatic AI-style assertion. Should be integrated into the preceding argument rather than standing as a separate declaration.

```
FIND:
This audit is not a side issue. It changes the scientific claim. Under the
corrected protocol, PD-PPO remains useful against dynamic heuristics, but the
strongest static baseline absorbs the apparent dynamic advantage.

REPLACE:
The protocol correction is consequential: under the corrected evaluation,
PD-PPO retains its advantage over dynamic heuristics, but the strongest
static baseline absorbs the apparent dynamic advantage that earlier
development diagnostics had suggested.
```

---

### A-12 §7.1 — "The main scientific outcome is regime dependence"

**Problem:** "The main scientific outcome is regime dependence." — AI-typical label sentence before the actual explanation. Should be merged with the explanation.

```
FIND:
The main scientific outcome is regime dependence. In the instantaneous-budget
regime, the best explanation of the results is not that PPO has solved a hard
event-triggered control problem.

REPLACE:
In the instantaneous-budget regime, the results are best explained not by
PPO solving a hard event-triggered control problem, but by a compact sensor
subset capturing most of the forecast value.
```

---

### A-13 §7.1 — "This is still useful: it tells a station designer"

**Problem:** "This is still useful:" — AI-typical defensive qualifier. Rephrase to make the positive claim directly.

```
FIND:
This is still useful: it tells a station designer that adaptive control
complexity is not justified by the fixed-budget simulation alone.

REPLACE:
For station designers, this result is informative: adaptive control
complexity is not warranted when the energy model permits a strong compact
subset.
```

---

### A-14 §8 Conclusion — "The fixed-budget engineering lesson is therefore direct"

**Problem:** "The fixed-budget engineering lesson is therefore direct:" — "therefore direct" is AI-typical filler. The lesson should be stated without the meta-label.

```
FIND:
The fixed-budget engineering lesson is therefore direct: adaptive control is not
automatically valuable when the energy regime permits a strong compact subset.

REPLACE:
When the energy regime permits a strong compact subset, adaptive control
adds no measurable value over a well-chosen fixed allocation.
```

---

## B. Sentence-Level Clarity and Precision

---

### B-1 Abstract — "A separate energy-storage experiment showed that saving energy for storm windows can make high-cost particle and flux sensors valuable"

**Problem:** The sentence is cut off at the page break in the PDF (page 1 ends mid-sentence). The full sentence on page 2 reads: "…reducing error relative to a static snow-core schedule." The sentence is grammatically complete but the construction "can make … valuable, reducing error" is a dangling participial. Rewrite for clarity.

```
FIND:
A separate energy-storage experiment showed that saving energy for storm
windows can make high-cost particle and flux sensors valuable, reducing error
relative to a static snow-core schedule.

REPLACE:
A separate energy-storage experiment showed that reserving energy for
storm windows enables high-cost particle and flux channels to reduce
forecast error relative to a static snow-core allocation.
```

---

### B-2 §1 — "weather windows are rare"

**Problem:** "weather windows are rare" is ambiguous — it could mean windows of good weather (for access) or windows of bad weather (blowing-snow events). The intended meaning is access windows for maintenance.

```
FIND:
a region where manual inspection is expensive and weather windows are rare

REPLACE:
a region where manual inspection is costly and logistical access windows
are infrequent
```

---

### B-3 §2.2 — "Static allocation remains a strong and necessary comparator"

**Problem:** "necessary" is doing two different jobs here — it means both "required for methodological completeness" and "likely to be competitive." Separate the two meanings.

```
FIND:
Static allocation remains a strong and necessary comparator.

REPLACE:
Static allocation is both a methodologically required comparator and,
in many regimes, a competitive one.
```

---

### B-4 §3.2 — "This is why a static comparator must be selected carefully and treated as a serious baseline."

**Problem:** "treated as a serious baseline" is informal. In a journal paper, the point should be made in terms of experimental design.

```
FIND:
This is why a static comparator must be selected carefully and treated as a
serious baseline.

REPLACE:
This property motivates selecting the static comparator on held-out
validation data rather than by hand-coded priority, so that it represents
the best achievable fixed allocation rather than an arbitrary one.
```

---

### B-5 §3.6 — "There is no reason for these objectives to induce the same policy ordering"

**Problem:** "There is no reason" is informal and slightly imprecise. The correct statement is that the objectives are not generally equivalent.

```
FIND:
There is no reason for these objectives to induce the same policy ordering
when some delayed sensors are valuable mainly around rare blowing-snow events.

REPLACE:
These objectives are not generally equivalent when some delayed sensors
carry value concentrated around rare blowing-snow events.
```

---

### B-6 §4.1 — "A learned policy that repeatedly selects the same projected subset should be interpreted as a compact allocation strategy, not as evidence of rich event-triggered control."

**Problem:** "rich event-triggered control" is informal. Replace with precise language.

```
FIND:
A learned policy that repeatedly selects the same projected subset should be
interpreted as a compact allocation strategy, not as evidence of rich
event-triggered control.

REPLACE:
A learned policy that repeatedly selects the same projected subset is
best interpreted as having converged to a compact fixed allocation, not
as having learned event-conditioned switching behaviour.
```

---

### B-7 §6.1 — "The same table also prevents overclaiming."

**Problem:** "prevents overclaiming" is informal and slightly awkward. The table does not prevent anything; the authors' interpretation does.

```
FIND:
The same table also prevents overclaiming. The validation-selected static
policy is as good as, or slightly better than, PD-PPO in the central budgets.

REPLACE:
The same table also constrains the interpretation. The validation-selected
static policy matches or slightly outperforms PD-PPO at the central budgets.
```

---

### B-8 §6.2 — "This is the regime signal that is absent from the instantaneous-budget benchmark"

**Problem:** "regime signal" is jargon that has not been defined. Replace with a plain description.

```
FIND:
This is the regime signal that is absent from the instantaneous-budget
benchmark: intertemporal energy bookkeeping can make high-value event
sensing feasible as a burst but not as a permanent allocation.

REPLACE:
This contrast is absent from the instantaneous-budget benchmark:
intertemporal energy bookkeeping makes high-cost event sensing feasible
as a timed burst, whereas a permanent static laser allocation is clipped
by the energy guard.
```

---

### B-9 §7.3 Limitations — serial "First … Second … Third … Fourth … Fifth"

**Problem:** Five consecutive sentences each beginning with an ordinal adverb is a known AI writing pattern. The limitations should be presented as a coherent paragraph with logical connectives, not as a numbered list in prose disguise.

```
FIND:
First, the evidence is simulation-based. The generator is statistically checked,
but the scheduler has not yet been validated on a deployed Antarctic station.
Second, the power model uses normalized scheduling costs rather than measured
watt-level profiles and physical battery dynamics. Third, the event flag is
available inside the simulator; an operational system would need a separate
always-on event detector or event forecast. Fourth, the frozen forecast oracle
is a surrogate endpoint. It keeps the comparison controlled, but it is not
equivalent to end-to-end joint training of a scheduler and downstream
forecaster. Fifth, the energy-account result is a storm-window reference-policy
result, not a full-distribution learned-policy result.

REPLACE:
All evidence is simulation-based: the generator passes statistical checks
against Antarctic station anchors, but the scheduler has not been validated
on a deployed station. The power model uses normalised scheduling costs
rather than measured watt-level profiles and physical battery dynamics,
so the energy-account analysis is an abstraction of, not a substitute for,
a physical power-budget study. The event flag is available as ground truth
inside the simulator; a deployable system would require a separate always-on
event proxy or event-forecast module. The frozen forecast oracle is a
surrogate endpoint that keeps comparisons controlled but is not equivalent
to end-to-end joint optimisation of scheduler and forecaster. Finally, the
energy-account result characterises a hand-crafted reference policy over
selected storm windows; it does not demonstrate that a learned policy can
reliably exploit the same opportunity under an independent evaluation protocol.
```

---

### B-10 §7.4 — "These changes are outside the evidence scope of the present manuscript, but they follow directly from its diagnosis."

**Problem:** "outside the evidence scope" is awkward. "follow directly from its diagnosis" is AI-typical self-congratulatory framing.

```
FIND:
These changes are outside the evidence scope of the present manuscript, but
they follow directly from its diagnosis.

REPLACE:
These extensions are beyond the scope of the present study; the current
results establish the conditions under which they would be necessary.
```

---

## C. Structural and Format Issues

---

### C-1 Figure 1 caption — disclaimer placement

**Problem:** The disclaimer "The rendering is not a field photograph or validation record" is important but buried in the middle of the caption. It should come first or be set off clearly.

```
FIND:
Figure 1: Conceptual three-dimensional rendering of the Antarctic AWS platform
motivating the simulated sensor suite. The rendering is not a field photograph
or validation record. Five physical sensor families are represented in the
simulator as logical scheduling channels with heterogeneous normalized costs,
startup peaks and warm-up abstractions.

REPLACE:
Figure 1: Conceptual rendering of the Antarctic AWS platform that motivated
the simulated sensor suite. \textit{Note: this is an illustrative rendering,
not a field photograph or instrument validation record.} Five physical sensor
families are represented in the simulator as logical scheduling channels with
heterogeneous normalised costs, startup peaks, and warm-up abstractions.
```

---

### C-2 §4.3 — Numbered list inside prose section

**Problem:** The four-item numbered list (oracle pretraining, RL training, validation, final testing) is formatted as a LaTeX enumerate inside a prose section. This is appropriate, but the list items are grammatically inconsistent: items 1 and 2 use subordinate clauses ("where the frozen forecast model is fitted"), while items 3 and 4 use the same pattern. The issue is that items 3 and 4 are shorter and feel truncated. Expand for parallelism.

```
FIND:
1. oracle pretraining, where the frozen forecast model is fitted;
2. RL training, where PD-PPO and candidate-prior information are learned;
3. validation, where the static comparator is selected;
4. final testing, where all reported fixed-budget results are evaluated.

REPLACE:
1. oracle pretraining, where the frozen forecast model is fitted on the
   first 35\% of each generated sequence;
2. policy training, where PD-PPO and candidate-prior information are
   learned on the subsequent 50\%;
3. validation, where the static comparator is selected on the next 7.5\%;
4. final testing, where all reported fixed-budget results are evaluated
   on the remaining 7.5\%, using non-overlapping 1024-step windows.
```

---

### C-3 Table 3 caption — "regenerated from final-test outputs"

**Problem:** "regenerated from final-test outputs" is ambiguous — it could mean the table was regenerated from stored artifacts, or that the policies were retrained. Clarify.

```
FIND:
Entries are mean ± standard deviation across $n = 10$ random seeds,
regenerated from final-test outputs with disjoint oracle-pretrain, RL-train,
validation, and test partitions.

REPLACE:
Entries are mean $\pm$ standard deviation across $n = 10$ random seeds
(seeds 41--50), evaluated on held-out final-test windows from sequences
with disjoint oracle-pretrain, policy-train, validation, and test
partitions (see Figure~3 and Table~B.5).
```

---

### C-4 Table 4 caption — epoch configuration note

**Problem:** "under a separate three-hour logical-epoch configuration" is mentioned only in the caption. This is a critical methodological difference from the one-hour fixed-budget experiment and should be flagged more prominently in the §6.2 body text, not only in the caption.

Add the following sentence to §6.2 immediately before the Table 4 reference:

```
FIND:
Table 4 reports a separate storm-window reference-policy analysis with
normalized harvest $h = 0.92$, capacity $C = 180$, reserve 20, and budget
$B = 1.20$.

REPLACE:
Table 4 reports a separate storm-window reference-policy analysis with
normalised harvest $h = 0.92$, capacity $C = 180$, reserve 20, and budget
$B = 1.20$. This experiment uses a three-hour logical epoch (rather than
the one-hour epoch of the fixed-budget benchmark), so its oracle-loss
values are not directly comparable with those in Table~3.
```

---

### C-5 §3 — Proposition/Remark formatting inconsistency

**Problem:** Propositions 1, 2, and 3 use `\textbf{Proposition N}` in the source, but Remark 1 also uses `\textbf{Remark 1}`. In standard LaTeX theorem environments these would be set in the same style. More importantly, Proposition 3 and Remark 1 are currently in the same section (§3.6) but logically belong in §3.2 (instantaneous budget), since they concern the fixed-budget regime. This is a structural issue that cannot be patched with a simple find/replace; it requires moving the blocks. Flag for manual relocation:

> **ACTION REQUIRED (manual):** Move Proposition 3 and Remark 1 from §3.6 to the end of §3.2, immediately after the sentence "This is why a static comparator must be selected carefully…" (after patch B-4 above). Update cross-references accordingly.

---

### C-6 §5.2.5 — Sentence fragment

**Problem:** "They combine practical considerations that affect scheduling decisions: steady operation, startup or heating peaks, warm-up latency, channel scarcity, and the importance of the observed variable for the forecast task." — The colon introduces a list that is grammatically a fragment (no verb). Rewrite.

```
FIND:
They combine practical considerations that affect scheduling decisions:
steady operation, startup or heating peaks, warm-up latency, channel
scarcity, and the importance of the observed variable for the forecast task.

REPLACE:
They reflect the practical considerations that govern scheduling decisions:
steady operating draw, startup or heating peaks, warm-up latency, channel
scarcity, and the forecast relevance of the observed variable.
```

---

### C-7 Appendix A.2 — "This is the theoretical interpretation of the corrected fixed-budget experiment: the result is not inconsistent with the prediction-driven formulation"

**Problem:** Double negation ("not inconsistent") is weak and AI-typical. State the positive claim.

```
FIND:
This is the theoretical interpretation of the corrected fixed-budget
experiment: the result is not inconsistent with the prediction-driven
formulation; it shows that the tested instantaneous regime does not create
enough time-varying opportunity for dynamic switching to dominate the
strongest static comparator.

REPLACE:
This is the theoretical interpretation of the corrected fixed-budget
experiment: the instantaneous-budget regime does not generate sufficient
time-varying opportunity for dynamic switching to outperform the strongest
static comparator, which is precisely what the prediction-driven formulation
predicts when the energy constraint is non-binding.
```

---

## D. Terminology Consistency

---

### D-1 "normalised" vs "normalized"

**Problem:** The paper mixes British ("normalised", "favour") and American ("normalized", "favor") spelling. The journal *Cold Regions Science and Technology* (Elsevier) accepts either but requires internal consistency. The current draft uses "normalized" in most places but "normalised" in a few (e.g., Table 1 caption, §5.2.5 patch above). Choose one and apply globally.

> **ACTION REQUIRED:** Run `grep -n "normalised\|normaliz" paper.tex` and standardise to "normalised" (British, consistent with Elsevier house style for this journal) throughout. Similarly check "favour/favor", "behaviour/behavior", "colour/color".

Current mixed instances identified in the 36-page excerpt:
- "normalized deployment costs" (Table 1 body) → "normalised deployment costs"
- "normalized steady cost" (§3.1) → "normalised steady cost"  
- "normalized scheduling costs" (§5.2.5) → "normalised scheduling costs"
- "normalized harvest" (§6.2, Table 4) → "normalised harvest"
- "normalized power constraint" (§3.2) → "normalised power constraint"

Patch (apply with `replace_all`):

```
FIND (replace_all): normalized
REPLACE: normalised
```

```
FIND (replace_all): favor
REPLACE: favour
```

```
FIND (replace_all): behavior
REPLACE: behaviour
```

---

### D-2 "AoI-based" vs "AoI based" vs "age-of-information"

**Problem:** The paper uses "AoI-based scheduling" (§4.4, §6.1), "AoI based scheduling" (no hyphen, if present), and "age-of-information" (§2.2). The abbreviation is defined in the Abbreviations section. After first use, "AoI" should be used consistently without re-spelling.

> **ACTION REQUIRED:** Verify that "age-of-information" is not spelled out after the Abbreviations section. In §2.2 the phrase "Age-of-Information" appears with capital letters in a list context — this is acceptable as a proper noun in that context but should be lowercase "age of information" in running prose.

```
FIND: Age-of-Information,
REPLACE: age of information (AoI),
```
(Apply only to the first occurrence in §2.2 body text, not the Abbreviations definition.)

---

### D-3 "PD-PPO" capitalisation in running text

**Problem:** The paper correctly uses "PD-PPO" as the algorithm name. However, in two places it appears as "prediction-driven PPO" (§1, §4.2) without the acronym. After the acronym is defined, "PD-PPO" should be used exclusively.

```
FIND: A prediction-driven PPO policy is trained under this reward

REPLACE: A PD-PPO policy is trained under this reward
```

```
FIND: The dominant reward term is $-\mathcal{L}_{\mathrm{FW}}(t)$.
Additional terms discourage excessive switching,
coverage failure, and hard-constraint violation. Direct state-tracking error is not
the organizing objective of the corrected manuscript; the paper is about preserving
future forecast quality under partial sensing.

REPLACE: The dominant reward term is $-\mathcal{L}_{\mathrm{FW}}(t)$;
additional terms penalise excessive switching, coverage failure, and
hard-constraint violation. State-tracking error is not the primary
objective; the scheduler is trained to preserve future forecast quality
under partial sensing.
```
(The second part of this patch also fixes the AI-typical "the paper is about" construction.)

---

### D-4 "corrected manuscript" / "rewritten manuscript" — self-referential labels

**Problem:** The paper refers to itself as "the corrected manuscript" (§4.2) and "this rewritten manuscript" (§4.3). These are revision-process labels that should not appear in the final submission.

```
FIND: Direct state-tracking error is not
the organizing objective of the corrected manuscript; the paper is about preserving
future forecast quality under partial sensing.

REPLACE: State-tracking error is not the primary objective; the scheduler
is trained to preserve future forecast quality under partial sensing.
```

```
FIND: Those earlier diagnostics remain useful for development, but they are not used
as primary evidence in this rewritten manuscript.

REPLACE: Those earlier diagnostics remain useful for development but are
not used as primary evidence in this paper.
```

---

### D-5 "truth sequence" / "generated truth sequence" / "generated sequence"

**Problem:** Three different phrases refer to the same object. Standardise to "generated truth sequence" throughout.

```
FIND (replace_all): generated sequence
REPLACE: generated truth sequence
```

(Check that this does not inadvertently alter "generated truth sequence" itself — it will not, since the pattern is a strict substring match only when "generated sequence" appears without "truth".)

---

## E. Summary Execution Order

Apply patches in the following order to avoid conflicts:

1. **D-1** (normalised/normalized global replace) — do first, as it affects many locations.
2. **D-5** (generated truth sequence) — global replace, no conflicts.
3. **D-4** (remove "corrected manuscript" / "rewritten manuscript") — targeted.
4. **D-3** (PD-PPO consistency) — targeted.
5. **D-2** (AoI capitalisation) — targeted, single location.
6. **A-1 through A-14** (de-AI-ification) — apply in document order (Abstract → §1 → §2 → §4 → §6 → §7 → §8).
7. **B-1 through B-10** (sentence-level clarity) — apply in document order.
8. **C-1 through C-6** (structural/format) — apply in document order; C-5 requires manual block relocation.
9. **Full recompile** and cross-reference check after all patches.

---

## F. Items Requiring Author Judgment (Not Patchable by Codex)

The following issues require a decision by the authors before a patch can be written:

**F-1 §3 Proposition/Remark relocation (C-5 above).** Moving Proposition 3 and Remark 1 to §3.2 improves logical flow but changes section numbering and all downstream cross-references. Confirm before executing.

**F-2 §6.4 Figure 6 reference.** The text says "Figure 6 illustrates the behavioral interpretation behind the main claim" but Figure 6 in the PDF is the behaviour timeline (sensor modes + rolling oracle loss). The caption correctly describes this figure. However, the body text says "PD-PPO is smoother than frequent-switching heuristics, but the figure does not establish a robust event-triggered laser-control mechanism." This is accurate but the figure as rendered (seed 41 only) shows the laser disdrometer active for much of the window, which could be misread as event-triggered. Authors should either (a) add a panel showing a seed where the laser is NOT event-triggered, or (b) add a sentence noting that seed 41 is not representative of the aggregate laser event/non-event ratio of 1.03×.

**F-3 §7 Future Work section.** The current §7.4 ("Next algorithmic step") is a single paragraph. Per the locked TOC (§8.1–8.5), this should expand to five subsections. The content exists across turns 76–79 but has not been drafted into the paper. This is a writing task, not a patch.

**F-4 Data Availability statement.** "A versioned archive … will be deposited before submission." This placeholder must be replaced with an actual repository DOI or a confirmed data-availability statement before final submission.

**F-5 Funding statement.** "Funding information will be inserted after author confirmation." Must be completed before submission.
