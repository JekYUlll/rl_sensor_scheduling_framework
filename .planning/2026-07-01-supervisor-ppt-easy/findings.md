# Findings

## Source assessment
- The 2026-06-29 PDF states the correct result but uses too many internal terms: operational step, event macro, strict fixed replay, gate, behaviour audit.
- The 2026-07-01 HTML is easier but still reads like a report, not a 2-3 page supervisor PPT.
- Better supervisor framing: the experiment changed because the old setting allowed a fixed sensor combination to explain much of the result; the new setting forces a meaningful choice among specialist sensors under one remaining slot.

## Plain-language story
1. Old line: multiple sensors competed together; fixed combinations could be strong, so RL improvement was hard to explain.
2. New line: always keep a basic weather channel, then choose only one specialist sensor depending on the weather event. This matches a low-power station intuition and makes dynamic scheduling necessary.
3. New results: across 24 random repeats, PD-PPO consistently improves prediction; event-specific behavior matches domain logic: particle -> laser, flux -> FC4, thermal -> surface IR.
4. Caution: present as a result for this forecast-oriented expert-sensor scheduling task, not a universal statement for all sensor scheduling.

## Figure choices
- Use 02_main_24seed_evidence.png for stability across seeds.
- Use 03_event_type_behavior.png for intuitive learned behavior.
- Use 04_mechanism_ablation.png only if space allows; otherwise summarize in text.

## Final deck decisions
- Use three slides, not two, to avoid crowding: design change, stable result, learned behavior.
- Use cropped key panels instead of full multi-panel figures because full paper figures have small English labels that are hard to read in a supervisor PPT.
- Keep the language natural and parenthetical: old line vs new line, fixed combination vs dynamic expert choice, not universal RL superiority.
- Keep the limitation sentence on the final slide to prevent overclaiming while staying positive.
