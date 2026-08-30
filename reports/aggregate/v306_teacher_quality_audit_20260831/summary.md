# V306 teacher-quality audit (lookahead=4)

The receding teacher was replayed on the same final-test starts used by the V294 corrected scene. It uses the frozen TCN oracle and the executable action geometry; it is an information diagnostic, not a deployable policy.

| seed | teacher loss | validation static | feasible static | AoI | random | round-robin |
|---:|---:|---:|---:|---:|---:|---:|
| 6811 | 0.313391 | 0.358921 | 0.407193 | 0.387442 | 0.414225 | 0.414651 |
| 6812 | 0.410015 | 0.506724 | 0.506724 | 0.524336 | 0.532527 | 0.542790 |

Conclusion: lookahead-4 teacher wins every listed baseline on both seeds. The V306 pretraining-only failure is therefore not evidence that the teacher target is weak. It isolates the unresolved issue to policy transfer, return/credit assignment, or state-distribution mismatch. No teacher-label or baseline-dependent patch is justified by this audit.

Source traces:
- `reports/v306_teacher_audit/seed6811/receding_oracle_l4_teacher_audit/`
- `reports/v306_teacher_audit/seed6812/receding_oracle_l4_teacher_audit/`
