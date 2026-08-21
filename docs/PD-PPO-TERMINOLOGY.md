---
document: PD-PPO terminology and claim contract
status: canonical pre-edit reference
scope: active main manuscript, anonymous manuscript, supplementary material, figures, tables, highlights, cover letter, and submission metadata
editorial_source: docs/08-02-01-word.md
stable_path: docs/PD-PPO-TERMINOLOGY.md
---

# PD-PPO terminology and claim contract

## 1. How to use this file

Read this file before every manuscript-polish iteration. It is the stable terminology baseline for the PD-PPO paper. Do not infer that two expressions are interchangeable merely because they sound similar.

Use it in this order:

1. Read Sections 2--4 before editing anything.
2. Consult the relevant concept table in Sections 5--14 while editing.
3. Apply the claim and evidence rules in Sections 15--16 before strengthening any result sentence.
4. Run the checklist in Section 22 before accepting a revision.

Interpret the labels as follows:

- **Canonical**: use this expression unless a mathematical sentence requires the listed symbol.
- **Allowed short form**: allowed only after the first-use form has appeared in the independently readable document or component.
- **Avoid**: ambiguous, inaccurate, nonstandard, or unsupported in this paper.
- **Conditional**: use only when the stated evidential condition is satisfied.

This contract resolves terminology. It does not authorize new scientific claims, new numerical results, or changes to frozen evidence. If a proposed edit conflicts with the implementation or frozen artifacts, verify the conflict rather than rewriting it away.

### Quick lookup map

| Editing topic | Read first |
|---|---|
| Any revision | Sections 2--4 |
| Method name, scheduler, policy, forecaster | Sections 4--5 |
| Conditions, labels, warnings, privilege | Section 6 |
| Data split and training/evaluation sequence | Section 7 |
| Policy inputs and unavailable information | Section 8 |
| Candidate, feasible, proposed, and executed actions | Section 9 |
| Measurement age, AoI, and uncertainty | Section 10 |
| Forecast loss, reward, and endpoints | Section 11 |
| BC, AWBC, PPO, GAE, and Double DQN | Section 12 |
| Baselines and references | Section 13 |
| Behavior, warning, switching, and simulation diagnostics | Section 14 |
| Seed strata, CIs, and statistical wording | Section 15 |
| Result sentences and claim boundaries | Section 16 |
| Sensor/channel names and units | Section 17 |
| Mathematical symbols | Section 18 |
| Spelling, capitalization, and hyphenation | Section 19 |
| Direct replacement of a risky expression | Section 20 |
| Figures, generated assets, release, and anonymity | Section 21 |
| Final review | Section 22 |

## 2. Evidence hierarchy

When sources disagree, use this order:

1. Current executable implementation and configuration.
2. Frozen evidence artifacts and their verified manifests.
3. Current active manuscript equations and protocol definitions.
4. This terminology contract.
5. `docs/08-02-01-word.md` as an editorial preparation document.
6. Historical manuscript wording only as a deprecated-alias source.

A style recommendation cannot override an implemented fact. A code identifier is not automatically good reader-facing terminology.

## 3. Non-negotiable distinctions

1. **Method versus trainable object**: the **PD-PPO scheduler** is the complete method; the **PD-PPO policy** is the trainable categorical neural policy.
2. **Method versus attribution unit**: the main empirical result belongs to the **complete PD-PPO training configuration**, not to reward design, PPO, warnings, behavior cloning, advantage-weighted behavior cloning, or the auxiliary classifier in isolation.
3. **Proposal versus execution**: the policy samples a **proposed action index** $u_t$, which identifies the **proposed activation vector** $\bar a_t=a^{(u_t)}$; the scheduler applies the **executed activation vector** $a_t$ after execution-rule enforcement.
4. **Likelihood versus transition credit**: the PPO buffer stores $u_t$ and $\log\pi_\theta(u_t\mid o_t,\mathcal I_t)$; the environment transition, reward, and generalized advantage estimate correspond to executed $a_t$. Existing logs do not support a post hoc switch to executed-action likelihood.
5. **Action mask versus dwell rule**: the **feasible-action mask** enforces instantaneous power feasibility over the candidate set; the **minimum dwell-time constraint** is a post-sampling execution rule.
6. **Scheduler input versus training/evaluation information**: condition labels, future-condition supervision labels, frozen-forecaster loss, and test targets are not online policy inputs.
7. **Condition versus event**: calm is an operating condition but not an event type. Particle, flux, and thermal are the three event types.
8. **Warning versus label**: warning signals are noisy simulated decision-time inputs; $c_t$ and $c_t^{\mathrm{sup}}$ are simulator-derived labels used only for training, calibration/validation, or offline evaluation as specified.
9. **Partition chronology versus computational order**: the chronological order is forecaster-training, policy-training, calibration/validation, test. The computational workflow may first fit constants on calibration/validation data and then construct labels on policy-training indices.
10. **Forecast targets versus AoI variables**: $\mathcal Y$ is the forecast-target set; $\mathcal V_{\mathrm{AoI}}$ is the observed-variable age set. Never use one as the index domain of the other.
11. **Primary evidence versus descriptive aggregate**: seeds 119--140 are the 22 post-selection evaluation seeds. Seeds 117--118 are pilot/model-selection seeds. The 24-seed aggregate is descriptive only.
12. **Endpoint versus metric**: mean forecast loss and macro-averaged normalized forecast loss are the two co-primary endpoints. Other reported quantities are metrics or diagnostics unless explicitly promoted by the protocol.
13. **Comparison versus ablation**: reward variants are matched PPO training configurations with different scalar objectives; Double DQN is a training-configuration comparison. Neither is an isolated component ablation.
14. **CI containing zero**: do not claim significance, equivalence, no difference, or a population-level directional advantage.
15. **Logical channel versus physical product**: the experiment instantiates logical sensing channels. Do not silently convert them into product-specific instruments or claim product-level transfer functions.

## 4. Mandatory first-use forms

| Concept | First-use form | Allowed short form | Do not use |
|---|---|---|---|
| Method | **Prediction-Driven Proximal Policy Optimization (PD-PPO)** | PD-PPO | Predictive PPO; prediction-oriented PPO; forecast PPO |
| Complete method | **PD-PPO scheduler** | scheduler, when unambiguous | trained scheduler, when referring to the neural object |
| Trainable decision rule | **categorical neural policy** or **PD-PPO policy** | policy | scheduler network; scheduling model |
| Frozen prediction model | **frozen forecaster** | forecaster | forecasting backbone; reward oracle; learned reward model |
| Proximal policy optimization | **proximal policy optimization (PPO)** | PPO | Proximal Policy Optimization after first definition is unnecessary |
| Deep reinforcement learning | **deep reinforcement learning (DRL)** | DRL | deep RL unless the short form is explicitly defined |
| Behavior cloning | **behavior cloning (BC)** | BC | behavioural cloning; imitation loss when the specific BC procedure is intended |
| Advantage-weighted behavior cloning | **advantage-weighted behavior cloning (AWBC)** | AWBC, only if repeatedly used | AWBC auxiliary loss when the continuing loss is not merely auxiliary |
| Primary forecaster architecture | **temporal convolutional network (TCN)** | TCN | TCN-style before TCN is defined; forecasting backbone |
| Generalized advantage estimation | **generalized advantage estimation (GAE)** | GAE | generalized-advantage estimator |
| Age metric | **age of information (AoI)** | AoI | Age of Information after first definition; information age |
| Energy state, only in the energy-account extension | **state of charge (SOC)** | SOC | SOC in the main abbreviation list if the active text never uses the energy-account extension |
| Confidence interval | **95% percentile bootstrap confidence interval (CI)**, when that is the implemented interval | 95% bootstrap CI; CI | confidence limits; error range |
| Warning discrimination | **receiver operating characteristic area under the curve (ROC AUC)** | ROC AUC | ROC-AUC; AUC-ROC |
| Warning precision--recall | **precision--recall area under the curve (PR AUC)** | PR AUC | PR-AUC; AUPRC unless separately defined |
| Value-based comparator | **Double DQN** | Double DQN | Double Deep Q-Network; Double-DQN; double DQN; DDQN without definition |
| Main held-out tuning split | **calibration/validation partition** | calibration/validation | validation partition when referring to the combined partition |
| Action roles | **proposed action index** $u_t$, **proposed activation vector** $\bar a_t$, and **executed activation vector** $a_t$ | proposal and execution, only after the objects are clear | proposed action mask $u_t$; executed $u_t$ |
| Operating-condition metadata | **operating condition** and **condition label** $c_t$ | condition; label, when unambiguous | warning label; regime label without definition |
| Supervision metadata | **future-condition supervision label** $c_t^{\mathrm{sup}}$ | supervision label | look-ahead supervision label; future event label |
| Warning input | **simulated warning signal** $q_{t,c}$ | warning signal | condition label; calibrated probability; field warning |
| Sensing abstraction | **logical sensing channel** | channel | physical sensor product unless substantiated |
| Label-privileged comparator | **privileged true-label baseline** | true-label baseline, after definition | deployable true-label policy |
| Target-privileged reference | **privileged one-step look-ahead reference using next-step test targets** | one-step look-ahead reference, after definition | greedy baseline; deployable look-ahead controller |
| Primary seed scope | **22 post-selection evaluation seeds (119--140)** | 22 post-selection seeds | 24 evaluation seeds |
| Disclosure-only seed scope | **descriptive 24-seed aggregate (117--140)** | 24-seed descriptive aggregate | confirmatory 24-seed analysis |

Define abbreviations independently in the abstract, main text, supplementary material, highlights, and any standalone figure/table document when those components may be read separately. Do not create an acronym for a term used only a few times.

## 5. Method identity and role terms

| Canonical term | Definition and use | Allowed short form | Avoid or distinguish from |
|---|---|---|---|
| Prediction-Driven Proximal Policy Optimization (PD-PPO) | Formal method name. Hyphenate **Prediction-Driven** because it is a compound modifier in the proper name. | PD-PPO | Predictive PPO; forecasting PPO |
| PD-PPO scheduler | Complete online scheduling method: input construction, categorical neural policy, feasibility check, and execution rule. | scheduler | PD-PPO policy; complete training configuration |
| PD-PPO policy | Trainable categorical neural policy that maps scheduler observations to a masked distribution over candidate actions. | policy | scheduler as a whole |
| complete PD-PPO training configuration | Empirical attribution unit for the reported main result. It includes the fixed warning-conditioned input design, forecast-loss reward, behavior cloning pretraining, continuing advantage-weighted behavior cloning, auxiliary future-condition classification, and PPO updates. | complete configuration, after explicit definition | reward alone; PPO alone; warning mechanism alone; BC, AWBC, or classifier alone |
| reported PD-PPO implementation | Code-specific realization of the scheduler and training configuration, including its proposal/execution credit convention. | implementation | generic PPO theory |
| frozen forecaster | Forecaster trained on the disjoint forecaster-training partition, then held fixed during policy training and all scheduler comparisons. It supplies forecast loss for training and offline evaluation but is absent from online action selection. | forecaster | oracle, reward model, forecasting backbone |
| primary forecaster | The frozen forecaster used for the main endpoint calculations. | primary forecaster | ridge forecaster |
| ridge forecaster | Secondary frozen linear forecaster used for a sensitivity analysis with its own validation-selected fixed baseline. | ridge model, only after definition | replacement for the primary endpoint model |
| sensor scheduler | Generic decision system that selects which logical sensing channels are requested. | scheduler | policy network |
| logical sensing channel | One selectable simulation-level sensing unit with observations, power demand, startup demand, warm-up behavior, and availability. | channel | verified physical product; individual forecast variable |
| core channel | Mandatory meteorological channel included in every candidate action. | core | optional channel |
| optional channel | Any of the five non-core logical channels that may be added subject to optional-channel capacity. | optional channel | specialist channel when the radiometer or other generic option is intended |
| specialist channel | Optional channel whose measurements are especially informative for one operating condition or target family. Use only where that role is intended. | specialist | every optional channel indiscriminately |
| forecast-oriented scheduling | Scheduling whose objective is tied to downstream forecast loss rather than instantaneous state estimation alone. | forecast-oriented | prediction-driven when the formal method name is intended |
| projected or masked action selection | Selection from a candidate set after infeasible actions have zero probability and the remaining probabilities are renormalized. | masked action selection | differentiable projection; continuous projection |

Use **the PD-PPO policy is trained**. Use **the PD-PPO scheduler executes**, **receives**, **applies**, or **requests**. Do not write **the scheduler is trained** unless the sentence explicitly refers to fitting the complete configured system and cannot be misread as identifying the trainable object.

## 6. Operating conditions, labels, warnings, and information availability

| Canonical term | Symbol | Definition and use | Avoid or distinguish from |
|---|---:|---|---|
| operating condition | $c_t$ as a realized label | One of calm, particle, flux, or thermal at time $t$. | state; mode, unless simulator mode is meant; event type when calm is included |
| condition label | $c_t$ | Simulator-derived current operating-condition label. It can select training weights and offline evaluation strata but is not an online policy input. | event label; state label; regime label without definition |
| calm period | $c_t=\mathrm{calm}$ over one or more steps | Time interval outside the three non-calm event types. | no-event regime if the paper has not defined regime; normal state |
| event | $c_t\ne\mathrm{calm}$ | A non-calm interval. | condition when calm is also possible |
| event type | $c\in\{\mathrm{particle},\mathrm{flux},\mathrm{thermal}\}$ | One of the three non-calm operating conditions. | calm event; regime unless the semi-Markov regime is specifically intended |
| particle event | $c_t=\mathrm{particle}$ | Simulated event subtype emphasizing particle-related variables. | particle microstructure event unless microstructure is explicitly defined |
| flux event | $c_t=\mathrm{flux}$ | Simulated event subtype emphasizing snow mass flux. | FC4 event |
| thermal event | $c_t=\mathrm{thermal}$ | Simulated event subtype emphasizing snow-surface thermal contrast. | temperature anomaly unless defined |
| future-condition supervision label | $c_t^{\mathrm{sup}}$ | First non-calm condition label from $t$ through $t+16$, or calm if none occurs. Used to construct the supervised target action index and auxiliary classification target. Not an online input. | look-ahead supervision label; future event label; current condition label |
| supervised target action index | $u_t^{\mathrm{target}}\in\mathcal I_t$ | Feasible candidate index obtained by applying the fixed condition-to-action mapping to $c_t^{\mathrm{sup}}$ at policy-training time indices. | oracle action without qualification; validation action; activation vector |
| condition-to-action mapping | no dedicated active-manuscript symbol | Mapping fixed from calibration/validation data and later applied to policy-training labels. | $S_c^\star$ from the theoretical formulation; labels computed from test data; online optimization |
| simulated warning signal | $q_{t,c}$ | Noisy score for one event type, with a 16-step pre-onset ramp in the simulator. Assumed available at decision time; not validated as a field detector. | condition label; calibrated probability; real sensor alert |
| aggregate warning signal | $q_t^{\max}=\max_c q_{t,c}$ | Maximum of the three event-specific warning scores. | event probability; condition label |
| thresholded warning condition | $\widehat c_t$ | Argmax event warning if the maximum is at least 0.5; calm otherwise. | true condition label |
| simulated warning-information assumption | none | Assumption that the four warning features are available online. Always disclose that the warnings are simulator-generated and highly informative. | field-validated warning system; label-free evidence |
| deployment-available information | none | Information explicitly included in $o_t$ at the decision time under the simulation protocol. | future target values; $c_t$; $c_t^{\mathrm{sup}}$ |
| training-only information | none | Labels, target action indices, condition-selected loss weights/normalizers, rewards, returns, and auxiliary targets used for policy optimization but unavailable to the decision-time policy. | deployment-available input |
| calibration/validation-only information | none | Frozen normalizers, condition-to-action mapping, and fixed-schedule selection decisions computed from the calibration/validation partition and transferred into later stages only through the declared protocol. | online policy input; test-selected information |
| evaluation-only information | none | Frozen future targets, condition labels used for grouping or loss computation, and diagnostic quantities applied after action execution. | online scheduler observation |
| privileged information | none | Exact current/future condition labels or next-step test targets unavailable to a deployed scheduler. | warning scores, which are assumed available in the simulation but require a separate realism caveat |
| offline evaluation label | $c_t$ when used after rollout | Condition label used to stratify frozen test outputs after actions have been executed. | online scheduler input |

Use **operating condition** as the generic four-class concept. Use **event type** only for particle, flux, and thermal. Reserve **regime** for the simulator's calm/storm semi-Markov process or for a separately defined regime construct.

The reported supervised condition-to-action mapping retains the meteorological core channel and pairs it with:

- the shielded temperature--humidity channel for calm;
- the laser disdrometer channel for particle;
- the FC4 snow mass-flux channel for flux;
- the infrared snow-surface-temperature channel for thermal.

## 7. Chronological partitions and protocol stages

The canonical chronological order is:

**forecaster-training $\rightarrow$ policy-training $\rightarrow$ calibration/validation $\rightarrow$ test**.

| Canonical term | Role | Must not be described as |
|---|---|---|
| forecaster-training partition | Trains the primary and secondary forecasters. Disjoint from policy training and test evaluation. | policy-training data; calibration data |
| policy-training partition | Supplies trajectories for behavior cloning pretraining and PPO. The fixed condition-to-action mapping is applied here to construct per-step supervised target actions. | validation data; test data |
| calibration/validation partition | Computes the frozen loss normalizers $s_c$ and $s_{\mathrm{calm}}$, defines the condition-to-action mapping used for supervised targets, and selects the fixed-schedule baseline before test evaluation. | validation partition, when all combined roles are intended; policy-training partition |
| test partition | Held-out temporal partition used only after model selection and final checkpointing. | validation set; training set |
| simulated ground-truth time series | Fully observed simulator trajectory to which the four-part chronological split is applied. | field ground truth; observed sensor stream |
| event-balanced evaluation window | Concatenated test windows selected to balance calm and the three event types for primary evaluation. | challenge set; external benchmark; independent episodes unless that is established |
| continuous scoreable test partition | All scoreable steps in the final test partition, excluding the final forecast-horizon tail. | event-balanced window; training partition |
| scoreable step | Time step for which all required future forecast targets exist. | every raw time step; evaluable step unless separately defined |
| excluded forecast-horizon tail | Last $H$ steps of a partition, omitted because full future targets are unavailable. | missing data; failed predictions |
| pilot/model-selection seed | Seed 117 or 118, used during model/protocol selection. | post-selection evaluation seed; independent replication seed |
| post-selection evaluation seed | One of seeds 119--140, evaluated after the configuration was fixed. | validation seed; pilot seed |
| descriptive 24-seed aggregate | Summary over seeds 117--140 retained for disclosure only. | confirmatory analysis; primary evidence |

Do not confuse chronological placement with computational dependency. Calibration/validation data may be processed before policy training to freeze constants and mappings, while remaining later in the time series.

The canonical computational sequence is:

1. train and freeze the forecaster on the forecaster-training partition;
2. compute $s_c$, $s_{\mathrm{calm}}$, the condition-to-action mapping, and the validation-selected fixed schedule from the calibration/validation partition;
3. apply the frozen mapping to policy-training labels and construct supervised targets;
4. perform behavior cloning pretraining;
5. perform PPO training with continuing supervised and auxiliary losses;
6. save the final policy checkpoint;
7. evaluate the frozen checkpoint on the test partition.

This computational order does not alter the four-part chronological split.

## 8. Scheduler observation and information flow

| Canonical term | Symbol | Definition | Avoid or distinguish from |
|---|---:|---|---|
| scheduler observation | $o_t$ | Concatenation of recent sample-and-hold values, recent validity masks, per-channel runtime features, previous executed action, current steady-state power ratio, time-of-day features, and four simulated warning features. | environment state if full latent truth is implied |
| recent sample-and-hold history | part of $o_t$ | Most recent 20 sample-and-hold value vectors and corresponding validity masks. | raw full-observation history; ten-step history |
| validity mask | $m_t$ or recent mask history | Indicates which variables produced valid observations at a time step. | action mask; sensor availability mask |
| recent-history features | part of $o_t$ | Ten deterministic features computed from the most recent 16 sample-and-hold value vectors and validity masks. | the 20-step value/mask history itself; latent-condition features |
| previous executed action | $a_{t-1}$ | Activation mask actually executed at the previous step. | previous proposal |
| sensor mode | runtime feature | Off, warming, or ready state represented in the per-channel runtime features. | selected action alone |
| warm-up remaining | runtime feature | Remaining warm-up steps normalized by the maximum configured warm-up. | remaining activation time |
| channel sampling age | runtime feature | Time since the channel's latest ready sampling opportunity. **Sampling opportunity age** is an allowed explanatory form. | variable-level AoI; generic measurement freshness without reset semantics |
| current steady-state power ratio | part of $o_t$ | Current executed steady-state power divided by the per-step steady-state power budget. | remaining energy fraction; startup-power ratio; generic power ratio |
| time-of-day features | $\sin\theta_t,\cos\theta_t$ | Cyclic features derived from simulation time. | calendar date; seasonal phase |
| remaining activation time | $r_t$ | Environment-side counter controlling the dwell-rule override. It is not included in the reported policy observation. | warm-up remaining; policy input; $\ell_t$ |
| latent truth | none | Fully simulated physical state used to generate observations, labels, and offline diagnostics. | scheduler observation |

The policy does not receive $c_t$, $c_t^{\mathrm{sup}}$, future forecast targets, frozen-forecaster loss, test reward, or remaining activation time.

## 9. Actions, feasibility, and execution

| Canonical term | Symbol | Definition and use | Avoid or distinguish from |
|---|---:|---|---|
| activation mask | $a_t\in\{0,1\}^m$ | Binary vector indicating active logical channels. | observation-validity mask |
| candidate action | $a\in\mathcal A$ | One pre-enumerated activation mask satisfying mandatory-core and optional-channel-capacity rules. | currently feasible action |
| candidate action set | $\mathcal A$ | Static set of all candidate activation vectors. In the reported $q=1$ instance it contains six vectors. | feasible action set $\mathcal F_t$; feasible action index set $\mathcal I_t$ |
| mandatory core constraint | $a_{\mathrm{met}}=1$ | Requires the meteorological core channel in every candidate. | power feasibility |
| optional-channel limit | $q$ | Maximum number of optional channels active with the core. The reported instance uses $q=1$. The broader concept is optional-channel capacity. | total channel capacity; number of candidates |
| optional-channel set | $\mathcal S$ | Index set of optional logical channels; $n_{\mathrm{opt}}=\lvert\mathcal S\rvert$. | operating-condition set $\mathcal C$ |
| optional-channel cardinality constraint | $\sum_{i\in\mathcal S}a_i\le1$ in the reported instance; $\le q$ in the broader formulation | Structural restriction defining $\mathcal A$. | candidate-set cardinality; steady-state power constraint |
| feasible action set | $\mathcal F_t$ | Candidate activation vectors in $\mathcal A$ that meet steady-state and startup-peak power limits relative to the previous executed vector. | candidate action set; feasible index set |
| feasible action index set | $\mathcal I_t=\{k:a^{(k)}\in\mathcal F_t\}$ | Indices of currently feasible candidate vectors. | $\mathcal F_t$ itself |
| feasible-action mask | implementation vector over candidate indices | Binary representation that removes indices outside $\mathcal I_t$ before the categorical distribution is normalized. | $\mathcal F_t$ as a mathematical set; dwell-time mask; observation mask; learned gate |
| masked categorical distribution | $\pi_\theta(\cdot\mid o_t,\mathcal I_t)$ | Categorical policy normalized over feasible indices only. | continuous relaxation; projection operator with gradients through hardware |
| proposed action index | $u_t$ | Candidate index sampled from the masked categorical distribution during training or selected as the highest-scoring feasible index during evaluation. Stored with its log-probability for PPO. | proposed activation vector; executed action |
| proposed activation vector | $\bar a_t=a^{(u_t)}$ | Candidate activation vector identified by $u_t$ before the dwell rule. | proposed action index; executed activation vector |
| executed activation vector | $a_t$ | Activation vector applied to the environment after dwell-rule enforcement. It determines measurements, power, state transition, forecast loss, reward, and evaluation traces. | proposed activation vector |
| proposed-action credit convention | none | Buffer stores $u_t$ and $\log\pi(u_t\mid o_t,\mathcal I_t)$, while transitions, rewards, GAE, and diagnostics follow executed $a_t$. | executed-action likelihood convention |
| executed-action PPO likelihood | none | Not available from the frozen logs when dwell retention changes proposal into execution. | quantity that can be reconstructed offline from the current rollout logs |
| previous executed action | $a_{t-1}$ | Basis for switching and startup-power calculations. | previous proposal |
| minimum dwell-time constraint | $D_{\min}$ | After an action change, retain the executed activation mask for at least $D_{\min}$ steps. | minimum activation duration; per-channel minimum on-time; warm-up |
| dwell-rule override | none | Replacement of proposed $\bar a_t$ by $a_{t-1}$ while $r_t>0$. | invalid-action masking; power projection |
| action change | $a_t\ne a_{t-1}$ | Any difference between successive executed masks. | proposal change |
| changed-channel fraction | $S_t=m^{-1}\lVert a_t-a_{t-1}\rVert_1$ | Normalized Hamming distance between consecutive executed activation masks; used as the switching-cost term. | binary switch indicator; count without normalization |
| steady-state power demand | $\sum_i p_i a_{t,i}$ | Demand of channels active after execution. | startup-peak demand |
| startup-peak demand | $P_t^{\mathrm{peak}}$ | Demand that substitutes peak startup power for channels turned on relative to $a_{t-1}$. | steady-state demand |
| steady-state power limit | $B$ | Instantaneous steady-state budget. | average-energy budget |
| startup power limit | $B^{\mathrm{peak}}$ | Instantaneous startup-peak budget. | steady-state limit |
| warm-up | none | Delay between requesting an off channel and receiving valid measurements. Hyphenate the noun and modifier. | startup power; dwell time |
| warm-up interruption | none | Aborted warming sequence. None occurred in the reported instance. | action override in general |

In the reported $q=1$ experiment, all six candidate actions satisfy both power limits at every evaluated step. State that the limits were enforced but did not exclude candidates. Do not claim that the power mask improved performance or frequently filtered actions without a dedicated binding-constraint experiment.

## 10. Measurement, sample-and-hold, age, and uncertainty

| Canonical term | Symbol | Definition and use | Avoid or distinguish from |
|---|---:|---|---|
| valid measurement | $y_{t,i,j}$ when channel and variable indices matter | Measurement of variable $j$ produced by ready channel $i$ and accepted as available at time $t$. | measurement attempt; sampling opportunity |
| observation mask | $m_t$ | Variable-level indicator of valid observations at time $t$. | activation mask |
| sample-and-hold value | none | Most recent retained valid measurement used when no new valid measurement arrives. | imputed ground truth; forecast |
| channel sampling age | sensor freshness feature | Time since the channel's most recent ready sampling opportunity; may reset even when a measurement attempt yields no valid measurement. | variable-level AoI |
| age of information (AoI) | $d_t^{(v)}$ | Time since the most recent valid observation of observed variable $v$. Resets only on a valid observation. | channel freshness; forecast target age |
| forecast-target set | $\mathcal Y$ | Variables whose future values contribute to forecast loss. | $\mathcal V_{\mathrm{AoI}}$ |
| observed-variable age set | $\mathcal V_{\mathrm{AoI}}$ | Variables indexed by the theoretical AoI cost. | forecast-target set |
| measurement attempt | none | A sampling attempt by a ready channel; it may fail to produce a valid measurement. | valid measurement |
| observation set for variable $j$ | $\mathcal O_{t,j}$ | Ready channels that provide valid measurements of variable $j$ at time $t$. | all active channels |
| measurement-noise variance | $R_{i,j}$ | Configured error variance for channel $i$ measuring scalar variable $j$. | full measurement-error covariance matrix |
| inverse-variance-weighted mean | observation-fusion equation | Scalar mean with weights $R_{i,j}^{-1}$ under conditionally independent errors. **Inverse-variance-weighted fusion** or **precision-weighted fusion** is acceptable only as an explanatory umbrella term. | minimum-variance fusion under correlated errors |
| weighted circular mean | none | Separate fusion rule for wind-direction measurements. | arithmetic mean of angles |
| measurement-error covariance matrix | none | Full cross-channel error covariance required for a correlated-error minimum-variance linear estimate. It is not supplied to the main policy. | diagonal measurement-noise variances |
| minimum-variance linear estimate | none | Counterfactual correlated-error fusion requiring the full measurement-error covariance matrix; not the implemented scalar fusion. | implemented state estimator |
| bounded diagonal uncertainty proxy | none | Empirical scalar objective constructed from per-variable uncertainty surrogates in the matched PPO objective comparison. | full covariance matrix; Kalman covariance |
| posterior covariance trace | $\operatorname{tr}(P_t)$ | Theoretical state-uncertainty quantity for a separate model-based estimator. | bounded diagonal proxy unless the correspondence is explicitly qualified |

AoI and uncertainty objectives are alternative scalar training objectives in matched PPO configurations. They are not components removed one at a time from the complete forecast-loss configuration.

## 11. Forecast loss, reward, and endpoint terms

| Canonical term | Symbol | Definition and use | Avoid or distinguish from |
|---|---:|---|---|
| forecast target | $j\in\mathcal Y$ | Variable predicted over future horizons by the frozen forecaster. | observed variable indexed by AoI |
| forecast horizon | $k=1,\ldots,H$ | Future prediction offset. The reported forecaster uses a fixed multi-horizon output. | episode horizon |
| horizon weight | $w_k$ | Fixed weight applied to the forecast error at horizon $k$; uniform in the reported experiment. | condition weight |
| condition-specific target weight | $\alpha_{j,c}$ | Predeclared weight for target $j$ under operating condition $c$. | training condition weight $\omega_c$; validation normalizer |
| target scale | $\sigma_j>0$ | Fixed scale used to normalize the absolute error of target $j$. | seed-level paired difference $d_s$; validation normalizer |
| capped scale-normalized absolute error | $\min\{10,\lvert\widehat y_{t+k,j}-y_{t+k,j}\rvert/\sigma_j\}$ | Componentwise error entering the per-step loss. | squared error; uncapped absolute error |
| per-step forecast loss | $L_{\mathrm{step}}(t)$ | Weighted average of capped scale-normalized absolute errors across targets and horizons, evaluated after the executed transition. | training-normalized loss; reward; squared error |
| condition-specific validation normalizer | $s_c$ | Median event-type-specific forecast loss across feasible fixed channel subsets on the calibration/validation partition. | mean loss; target scale; standard deviation |
| calm validation normalizer | $s_{\mathrm{calm}}$ | Median overall forecast loss across the same feasible fixed channel subsets on the calibration/validation partition; used for calm training steps. | median calm-only loss unless separately computed |
| condition-specific training weight | $\omega_c$ | Frozen scalar multiplying the normalized training loss for current condition $c$. | target weight $\alpha_{j,c}$ |
| normalized training loss | $L_{\mathrm{train}}(t)=\omega_{c_t}L_{\mathrm{step}}(t)/s_{c_t}$ | Training loss using the frozen condition weight and validation normalizer selected by current $c_t$. | macro endpoint |
| switching-cost term | $\lambda_{\mathrm{sw}}S_t$ | Reward contribution based on executed changed-channel fraction. | logged cumulative switching energy; binary switching penalty |
| warm-up-abort term | $\lambda_{\mathrm{warm}}W_t$ | Reward term for interrupted warm-up. Structurally zero in the reported setup because $D_{\min}$ exceeds maximum warm-up. | startup-power penalty |
| policy-training reward | $R_t$ | Negative sum of normalized forecast loss and configured executed-action penalties, computed after execution and environment update. | online policy input; learned reward |
| mean forecast loss | none | Mean per-step forecast loss over the specified test support. One of the two co-primary endpoints. | macro-averaged normalized forecast loss |
| condition-specific test loss | $L_c$ | Mean per-step forecast loss over test steps with event type $c$. | normalized condition score unless divided by $s_c$ |
| macro-averaged normalized forecast loss | $L_{\mathrm{macro}}$ | Equal average of $L_c/s_c$ over particle, flux, and thermal event types. Calm periods are excluded. One of the two co-primary endpoints. | mean forecast loss; event-frequency-weighted average |
| co-primary endpoint | none | Either mean forecast loss or macro-averaged normalized forecast loss. Use the plural when naming both. The label gives both endpoints primary reporting status; it does not by itself define multiplicity control or a joint success rule. | secondary metric; dual primary metric; automatic joint significance claim |
| normalized condition-specific loss | $\widetilde L_c(\pi)$ | Condition-specific loss used in Proposition 1; in the empirical macro definition it is $L_c(\pi)/s_c$. | raw $L_c$ |
| weighted normalized forecast loss | $L_\rho(\pi)=\sum_c\rho_c\widetilde L_c(\pi)$ | Theoretical weighted condition objective in Proposition 1. | empirical equal-weight event-only macro unless $\rho_c$ is set accordingly |
| discounted forecast objective | $J_{\mathrm{fcst}}(\pi)$ | Negative expected finite-horizon discounted sum of $L_{\mathrm{step}}(t;\pi)$. Higher objective value means lower discounted forecast loss. | empirical mean forecast loss without qualification |
| discounted AoI objective | $J_{\mathrm{AoI}}(\pi)$ | Negative expected discounted sum of ages over $\mathcal V_{\mathrm{AoI}}$. | empirical AoI-objective PPO training loss without qualification |
| discounted covariance objective | $J_{\mathrm{cov}}(\pi)$ | Negative expected discounted sum of $\operatorname{tr}(P_t)$ for the separate model-based estimator. | empirical bounded diagonal uncertainty proxy |

Use **forecast loss** as a noun. A hyphen is acceptable in a compact attributive compound such as **forecast-loss reward**, but prefer clearer constructions such as **reward based on forecast loss** in prose.

## 12. Training and optimization terms

| Canonical term | Symbol | Definition and use | Avoid or distinguish from |
|---|---:|---|---|
| behavior cloning | BC, after definition | Supervised fitting of the policy to target actions before PPO. | imitation learning as a broader field, unless intended |
| behavior cloning pretraining | none | Initial supervised phase on policy-training examples. Keep the established term **behavior cloning** open. | offline pretraining on calibration/validation rows; behavior-cloning pretraining |
| advantage-weighted behavior cloning loss | $\mathcal L_{\mathrm{BC}}$ | Continuing supervised loss weighted by positive advantage during PPO updates. `AWBC` may be introduced only if repeatedly needed. | ordinary BC; isolated component effect; advantage-weighted behavior-cloning loss |
| auxiliary future-condition classification loss | $\mathcal L_{\mathrm{cond}}$ | Auxiliary loss predicting the future-condition supervision label from policy features. | look-ahead reference; online true-label input; reward term |
| PPO clipped surrogate loss | $\mathcal L_{\mathrm{clip}}$ | Negative clipped PPO surrogate used in minimization. If discussing the maximized form, call it the clipped surrogate objective and state the sign convention. | policy reward |
| critic or value estimate | no dedicated active-manuscript symbol | Estimate used to form return targets and GAE. Introduce a symbol only if the edited text defines it. | frozen-forecaster prediction |
| value-function loss | $\mathcal L_V$ | Squared error between the value estimate and return target. | forecast loss |
| policy entropy | $\mathcal H(\pi_\theta(\cdot\mid o_t,\mathcal I_t))$ | Entropy regularizer of the masked policy distribution during training. | empirical executed-action entropy |
| composite PPO loss | $\mathcal L_{\mathrm{PPO}}$ | Sum of clipped policy, value, entropy, continuing behavior cloning, and auxiliary terms with configured coefficients. | environment reward |
| generalized advantage estimation | $\widehat A_t$ via GAE | Advantage estimator computed from rewards and value estimates along executed transitions. | executed-action likelihood |
| probability ratio | $\rho_t(\theta)$ | Ratio using the stored proposed-action-index likelihood under new and old policy parameters. | ratio for executed action unless explicitly logged and recomputed |
| positive-advantage weight | $[\widehat A_t]_+$ | Nonnegative weight in the continuing behavior cloning term. | proof that the supervised target caused the advantage |
| valid supervised target-action indicator | $\ell_t$ | Gates the behavior cloning loss when a feasible supervised target action exists. | remaining activation time $r_t$; event indicator |
| future-condition label-availability indicator | $\eta_t$ | Gates the auxiliary classification loss when a future-condition supervision label exists. | valid target-action indicator $\ell_t$; event indicator |
| final checkpoint | none | Policy parameters saved after the configured training sequence and used for test evaluation. | best test checkpoint; online-selected checkpoint |
| matched PPO training configurations with different scalar objectives | none | Canonical description of forecast-loss, AoI, and uncertainty objective variants. | reward-only ablation; reward proves improvement |
| Double DQN training configuration | none | Complete value-based comparison using a dueling network, feasible-action mask, online and target Q-networks, three-step returns, and experience replay under the shared environmental setup. | pure algorithm ablation; isolated PPO comparison |
| online Q-network | none | Q-network updated directly by gradient optimization. | target Q-network |
| target Q-network | none | Lagged Q-network used in the Double DQN target calculation. | online Q-network |
| Double DQN target | none | Target using the implemented online/target-network split over feasible next actions. | ordinary DQN target unless the selection/evaluation rule is ordinary DQN |
| three-step return | none | Three-transition return used by the reported Double DQN configuration. | one-step temporal-difference target |
| experience replay | none | Off-policy reuse of stored transitions for Q-network updates. | behavior cloning |
| replay buffer | none | Data structure storing transitions for experience replay. | PPO rollout buffer |

Do not write that behavior cloning, AWBC, the auxiliary classifier, warnings, forecast reward, or PPO **caused** the complete configuration's improvement unless a dedicated controlled experiment isolates that component.

## 13. Baselines, references, and comparison objects

| Canonical term | Definition and information access | Avoid or distinguish from |
|---|---|---|
| validation-selected fixed-schedule baseline | Primary structural comparator. A single fixed channel subset is selected on calibration/validation data before test evaluation using the predeclared lexicographic rule. | best fixed action chosen on test; static oracle |
| fixed schedule | Same activation mask at every step. | periodic schedule; dwell-constrained adaptive schedule |
| fixed-channel policy that always selects one optional channel | Theoretical policy class covered by Proposition 1. | core-only fixed policy; every fixed-channel policy |
| core-only fixed policy | Fixed policy $\pi_{\emptyset}$ that selects no optional channel. Not covered by the current Proposition 1 comparison. | fixed policy covered by Proposition 1 |
| warning-rule baseline | Chooses a calibration/validation-selected condition-specific subset using the thresholded simulated warning condition $\widehat c_t$. Uses warning scores assumed available online in the simulation. | true-label baseline; field-validated warning controller |
| privileged true-label baseline | Receives the exact sequence $c_t,\ldots,c_{t+16}$ at each decision and is unavailable at deployment. | warning-rule baseline; deployable policy |
| privileged one-step look-ahead reference using next-step test targets | Snapshots the complete test environment, evaluates each candidate using next-step target values, restores the state, and chooses lexicographically. The shorter **one-step look-ahead reference** is allowed only after this disclosure. | deployable greedy baseline; frozen-forecaster policy |
| AoI-priority baseline | Dynamic rule that favors channels according to age-based priority under feasibility and execution rules. | AoI training objective |
| round-robin baseline | Dynamic cyclic rule over eligible channels. Hyphenate **round-robin**. | random feasible baseline |
| random-feasible baseline | Randomly selects among currently feasible candidates. | unrestricted random action |
| post hoc best rule-based dynamic comparator | Best-performing rule-based dynamic comparator identified after inspecting test results. Report as post hoc, not preselected. | prospective baseline; validation-selected comparator |
| same-duty comparator | Comparator matched on an explicitly defined duty quantity. Use only with the exact matching definition. | fixed-schedule baseline |
| theoretical label-aware policy | $\pi_{\mathrm{label}}$ in Proposition 1, which receives condition labels by construction. | empirical true-label baseline unless their procedures are shown equivalent |
| rule-based dynamic baseline | Deterministic or stochastic adaptive scheduling rule without learned policy parameters. | fixed baseline; learned comparator |
| training-configuration comparison | Comparison whose training algorithm or objective changes along with associated targets, advantages, or losses. | component ablation |

Use **baseline** for an explicitly specified comparator procedure. Use **reference** for an informative but privileged or nondeployable comparison. Use **comparator** as the neutral umbrella term. Do not rename objects solely to make results sound stronger or weaker.

## 14. Diagnostics and operational metrics

### 14.1 Channel-use acceptance criterion

Use **channel-use acceptance criterion** as the transparent canonical name. If the historical phrase **common channel-use diagnostic** must be retained for traceability, define it immediately with the following exact criterion:

- warm-up abort count $=0$;
- exactly one always-on channel, where activation frequency is at least $0.99$;
- exactly one always-off channel, where activation frequency is at most $0.01$;
- three or four intermediate-duty channels, where activation frequency lies in $[0.05,0.95]$.

This criterion is implemented in `scripts/95_v31_build_clean_paper_assets.py` using thresholds from `src/v2/rollout.py`. It is not the same as the behavior-complexity gate.

### 14.2 Behavior-complexity gate

The **behavior-complexity gate** rejects schedules that are fixed-like, simple-cycle-like, low-complexity, or weakly condition-dependent. Its implemented diagnostics include:

| Canonical diagnostic | Implemented quantity | Terminology caution |
|---|---|---|
| unique executed-action count | Number of distinct executed activation masks. | Not number of policy logits or feasible actions. |
| executed-action entropy | Shannon entropy in bits of empirical executed-mask frequencies. | Not policy entropy. |
| consecutive-action-pair entropy | Shannon entropy in bits of categorical ordered pairs $a_{t-1}\rightarrow a_t$. | Not conditional entropy unless normalized as such. |
| dominant-action fraction | Fraction occupied by the most frequent executed mask. | Old: fixed top-1 fraction. |
| top-three action coverage | Combined fraction of the three most frequent executed masks. | Do not call diversity. |
| best periodic-match fraction | Highest exact mask-match fraction over lags 1--64. | Old: periodicity through lag 64. |
| condition--executed-action mutual information | Mutual information in bits between current condition label and categorical executed activation mask. | `condition--channel mutual information` is only shorthand in the $q=1$ instance. |
| condition-specific activation shift | Maximum pairwise $L_1$ difference between per-channel activation-frequency vectors across condition labels. | Old: sufficient observation dependence. |
| event-specific activation shift | $L_1$ or $L_\infty$ difference between event and non-event activation frequencies, with the norm stated. | Do not omit the norm. |

The frozen gate thresholds are part of the analysis procedure, not universal definitions. When threshold values matter, report them explicitly rather than using adjectives such as *sufficient*, *complex*, or *nontrivial* alone.

For the frozen mechanism analysis, the collector passes these values explicitly:

- maximum periodicity lag: 64 steps;
- fixed-like dominant-action threshold: $0.95$;
- simple-cycle top-three coverage threshold: $0.85$;
- simple-cycle periodic-match threshold: $0.90$;
- minimum unique executed actions: 5;
- minimum executed-action entropy: $1.50$ bits;
- minimum consecutive-action-pair entropy: $1.25$ bits;
- event-dependence thresholds: activation-vector $L_1$ shift at least $0.50$ or event/action mutual information at least $0.10$ bits;
- subtype-dependence thresholds: maximum pairwise activation-vector $L_1$ shift at least $1.00$ or condition/action mutual information at least $0.25$ bits.

The gate passes only when the sequence is not fixed-like, not simple-cycle-like, not low-complexity, and not weakly condition-dependent under those implemented rules. These thresholds are analysis settings, not physical constants or generally validated standards.

### 14.3 Switching and power diagnostics

| Canonical diagnostic | Definition | Avoid |
|---|---|---|
| mean normalized Hamming distance per transition | Mean of $m^{-1}\lVert a_t-a_{t-1}\rVert_1$ over adjacent executed masks. | changed-activation fraction without definition; switch rate if binary switching is implied |
| concatenated-window boundary contribution | Contribution from the seven artificial boundaries created when eight evaluation windows are concatenated. | treating all transitions as within-window decisions |
| current steady-state power ratio | Current steady-state demand divided by the per-step steady-state budget. | current power ratio without denominator |
| maximum steady-state demand | Maximum executed steady-state power over the stated support. | average power |
| maximum startup-peak demand | Maximum startup-aware demand over the stated support. | steady-state demand |
| warm-up abort count | Number of warming sequences interrupted before readiness. | dwell override count |
| proposal override rate | Fraction of policy proposals replaced by the dwell rule. Not recorded for the frozen run. | inferring it from executed switches |

The reported mean changed-channel fraction of $0.00369$ over concatenated evaluation windows is a descriptive execution metric. Multiplying it by $\lambda_{\mathrm{sw}}$ gives only a nominal quantity, not a logged training-reward contribution.

### 14.4 Warning diagnostics

| Canonical diagnostic | Definition | Avoid |
|---|---|---|
| subtype-wise one-vs-rest ROC AUC | ROC AUC for one event-warning score against the corresponding current condition label versus all other labels. | event-wise ROC-AUC without class construction |
| subtype-wise one-vs-rest PR AUC | Average precision for the same one-vs-rest task. | PR-AUC without positive-class definition |
| thresholded four-class warning accuracy | Accuracy of argmax warning label above 0.5, calm otherwise. | warning accuracy without threshold rule |
| macro-averaged F1 score | Unweighted mean of the four classwise F1 scores for calm, particle, flux, and thermal. `macro-F1` is allowed after definition. | F1 without averaging scheme |
| warning--condition mutual information | Mutual information in bits between the thresholded four-class warning output and current condition label. | causal information; calibrated probability |
| event-onset warning lead | Number of steps from first threshold crossing in the preceding 16-step window to event onset. | prediction horizon; guaranteed field lead time |

These metrics quantify information in simulator-generated warnings. They do not validate a field detector and do not make the warning-rule baseline equivalent to the privileged true-label baseline.

### 14.5 Simulation-validation diagnostics

| Canonical diagnostic | Definition in the reported validation | Avoid or qualify |
|---|---|---|
| wind-speed autocorrelation-function deviation | Maximum absolute difference between simulated and Antarctic-reference wind-speed autocorrelation functions over lags 1--12 h. Define **autocorrelation function (ACF)** on first use if the abbreviation is repeated. | autocorrelation error without lag range or norm |
| maximum Kolmogorov--Smirnov statistic | Maximum two-sample Kolmogorov--Smirnov (KS) statistic across the scalar meteorological variables compared with Antarctic-reference distributions. | KS distance without variables; KS test result if no p-value-based test is intended |
| event-heavy-window fraction | Fraction of 512-step windows with event fraction greater than 0.75. | event frequency without window and threshold |
| calm-window fraction | Fraction of 512-step windows with event fraction below 0.25. | calm-period prevalence without window and threshold |
| median blowing-snow event duration | Median duration of simulated non-calm event runs, reported in hours. | storm duration; warning duration |
| flux--wind log--log slope | Fitted slope between positive snow mass flux and wind speed on logarithmic axes over active event samples. | correlation coefficient; causal exponent |
| particle-size--wind Spearman rank correlation | Spearman rank correlation between particle diameter and wind speed over the defined active particle-event samples. Use $\rho$ only after defining it. | Pearson correlation; particle-velocity correlation |
| wind-speed power-spectrum log-MSE | Mean squared difference between log power spectra over 0.1--4.0 cycles per day. Define **mean squared error (MSE)** or **power spectral density (PSD)** only if those abbreviations are reused. | spectral error without frequency band or log scale |

These are generator-acceptance diagnostics for the simulated environment. Passing them does not establish field equivalence, complete distributional fidelity, or validity outside the measured variables, lags, windows, and frequency band.

## 15. Statistical and evidence terms

| Canonical term | Definition and use | Avoid or qualify |
|---|---|---|
| seed | One simulation/training realization indexed by its random seed. | replicate, unless independence and experimental replication are defined |
| seed-level paired difference | $d_s=L_s(\text{comparator})-L_s(\text{PD-PPO})$. Positive values favor PD-PPO. | improvement with unspecified sign |
| mean paired difference | Mean of $d_s$ across the stated seed set. Describe it as an observed paired improvement only when the sign is positive and the comparator is explicit. | mean paired improvement as an unconditional estimand; percent reduction unless denominator is stated |
| relative reduction | $(L_{\mathrm{comp}}-L_{\mathrm{PD-PPO}})/L_{\mathrm{comp}}$. State the comparator and support. | improvement rate without denominator |
| 95% percentile bootstrap confidence interval | Percentile interval produced by the frozen seed-level bootstrap procedure. | confidence limits; Bayesian credible interval |
| exact one-sided sign test | Exact test of the predeclared direction using signs of the 22 post-selection paired differences. | two-sided test; 24-seed confirmatory test |
| post-selection evaluation | Evaluation on seeds 119--140 after configuration selection. | pristine external validation; preregistered trial |
| descriptive aggregate | Summary retained for disclosure but not used as primary inferential evidence. | confirmatory aggregate |
| claim--evidence attribution boundary | Rule that a result may be attributed only to the configuration or comparison actually supported by the frozen evidence. | component-level causal attribution |
| evidence support | Exact seed set, temporal support, endpoint, comparator, and forecaster underlying a value. | result context inferred from a nearby table |
| statistical significance | Use only with an explicitly named test and prespecified threshold. Prefer reporting estimate, CI, and test result directly. | `significant` as a synonym for large or important |
| equivalence | Requires an equivalence margin and equivalence test. | CI includes zero, therefore equivalent |
| no detected directional difference | Safe description when the paired CI includes zero and no test supports a direction. | no difference; identical; the same |

### CI rule

If a paired 95% bootstrap CI includes zero, the allowed conclusion is limited to the observed estimate and uncertainty. Do not write:

- significantly better or worse;
- equivalent;
- no difference;
- similar performance, unless explicitly framed as descriptive and not inferential;
- an average directional advantage in the population.

A safe form is:

> The paired point estimate was [value], and its 95% bootstrap confidence interval included zero; these data do not establish a population-level directional difference.

## 16. Frozen claim and attribution templates

### 16.1 Core result sentences

For the macro endpoint, use this sentence unchanged unless frozen evidence is formally updated:

> Across the 22 post-selection evaluation seeds, the complete PD-PPO training configuration achieved lower macro-averaged normalized forecast loss than the validation-selected fixed-schedule baseline in every seed.

When both co-primary endpoints must be stated, use:

> Across the 22 post-selection evaluation seeds, the complete PD-PPO training configuration achieved lower mean forecast loss and lower macro-averaged normalized forecast loss than the validation-selected fixed-schedule baseline in every seed.

This sentence does **not** establish that:

- PD-PPO outperforms every rule-based or context-aware policy;
- forecast-loss reward alone causes the improvement;
- PPO is superior to Double DQN as an isolated algorithmic factor;
- warning inputs alone cause the improvement;
- behavior cloning, AWBC, or auxiliary classification is individually necessary;
- the power mask was behaviorally active;
- the warnings are field-realistic.

### 16.2 Reward-objective comparison

Use:

> matched PPO training configurations with different scalar objectives

Also disclose that each scalar objective changes the PPO reward, value target, advantage, and advantage weights used in the continuing behavior cloning term. Do not call these rows **reward-only ablations**.

### 16.3 Double DQN comparison

Use:

> Double DQN training-configuration comparison

Do not write **PPO versus DQN ablation** or attribute the difference only to the value-based versus policy-gradient update.

### 16.4 Warning-rule comparison

Use language such as:

> The warning-rule baseline used the same simulator-generated warning scores assumed available to PD-PPO. The paired estimate was small and its confidence interval included zero, so the comparison did not establish a population-level directional difference.

Do not claim that the complete configuration consistently outperformed the warning-rule baseline.

### 16.5 Privileged references

Use:

> The privileged true-label baseline uses exact current and future condition labels, and the privileged one-step look-ahead reference uses next-step test targets; neither information source is available to a deployed scheduler.

Do not describe either as deployable, online-feasible, or label-free.

### 16.6 Constraint activity

Use:

> Both instantaneous power limits were enforced, but all six candidate actions were feasible in the reported $q=1$ instance; the active restriction was primarily the mandatory-core, one-optional-channel action design together with the minimum dwell-time rule.

Do not claim that action masking or power projection improved forecast performance without a binding-constraint experiment.

### 16.7 Proposition 1

Use:

> Proposition 1 gives sufficient conditions under which condition-dependent channel selection improves on fixed policies that always select one optional channel.

Do not generalize the result to the core-only fixed policy.

### 16.8 Proposition 2

Any restatement must preserve:

- the capped, target-weighted, scale-normalized absolute-error loss defined by `eq:step_forecast_loss` (currently Equation (5));
- a common finite horizon $T$;
- common discount factors $\gamma^t$;
- $\mathcal Y$ as the forecast-target set;
- $\mathcal V_{\mathrm{AoI}}$ as the AoI-variable set.

Do not substitute the empirical macro endpoint for the theoretical loss without an explicit derivation.

### 16.9 Claims that require new experiments

Ordinary terminology and prose polishing uses the frozen scientific evidence and does not rerun training, rollout, or simulation. New evidence is required before adding any of the following claims:

- **the power mask is behaviorally effective**: requires an experiment in which the power constraints bind and exclude otherwise admissible candidate actions;
- **the forecast-loss reward is superior by itself**: requires an isolated reward comparison that does not also change value targets, advantages, or supervised-loss weights;
- **the method is robust to warning quality**: requires a warning-quality or warning-noise sweep over predeclared perturbations.

Do not convert a prose edit into one of these claims by changing only an adjective, heading, or caption.

## 17. Scenario and sensing-channel nomenclature

The reported experiment uses six **logical sensing channels**. The safe reader-facing names are:

| Canonical logical-channel name | Safe description | Conditional or prohibited replacement |
|---|---|---|
| meteorological core channel | Mandatory channel supplying core meteorological measurements. | AWS alone if the acronym has not been defined |
| basic radiometer channel | Optional channel supplying the simulated solar-radiation measurement. | pyranometer, unless hemispherical broadband irradiance and instrument assumptions are explicitly established |
| shielded temperature--humidity channel | Optional channel supplying shielded air-temperature and relative-humidity measurements. | thermo-hygrometer product class without support |
| infrared snow-surface-temperature channel | Optional channel supplying snow-surface temperature. | infrared thermometer if a product-level instrument is intended but not modeled |
| laser disdrometer channel | Optional channel supplying simulated particle diameter and particle velocity. | particle fall velocity unless the velocity direction is explicitly vertical; particle microstructure unless defined |
| FC4 snow mass-flux channel | Optional channel supplying simulated snow mass flux and wind speed. | FlowCapt FC4 acoustic snow-flux sensor unless the manuscript intentionally makes and supports the product-level identification |

The code and paper represent logical channel behavior, not detailed transfer functions of named commercial instruments. Product names may be used as motivation only after verifying the product identity, measurand, unit, and modeled response.

Use **snow mass flux** for `snow_mass_flux_kg_m2_s`, with unit $\mathrm{kg\,m^{-2}\,s^{-1}}$ unless a documented conversion is performed. Do not silently replace it with grams per square metre per second.

Use **particle velocity** for `snow_particle_mean_velocity_ms`. Do not call it **fall velocity** without a defined vertical component.

Use **wind-driven snow** as a safe umbrella expression when no meteorological height distinction is intended. Keep **drifting snow** and **blowing snow** distinct when invoking formal meteorological definitions; do not interchange them merely for variety.

For automatic weather stations:

- first use when singular: **Antarctic automatic weather station (AWS)**;
- first use when plural is needed: **Antarctic automatic weather stations (AWSs)**;
- as a modifier: **Antarctic AWS network** or **station network**;
- avoid **AWS station** and **AWSs network**.

### Scientific variables and phenomenon names

| Canonical term | Scope and caution |
|---|---|
| cold-region environmental monitoring | Broad application context. Use **Antarctic monitoring** when the scope is specifically Antarctic. |
| wind-driven snow transport | Safe umbrella term for snow transport caused by wind. It does not erase the meteorological height distinction between drifting snow and blowing snow. |
| blowing-snow condition or event | Use for the simulated phenomenon only under the manuscript's operational definition. Do not silently replace with drifting snow. |
| drifting-snow field observations | Use for AntAWS provenance when that is the dataset description. Do not relabel as blowing-snow observations without checking the measurement definition. |
| wind speed; wind direction | Modeled meteorological variables. **Wind-direction measurements** is the attributive form in the fusion rule. |
| air temperature; relative humidity; air pressure | Modeled meteorological variables. Prefer **air pressure** to an unsupported product-specific barometer label. |
| solar radiation | Name of the modeled quantity. Use **solar irradiance** only after confirming the physical measurand and geometry. |
| snow surface temperature | Modeled surface-temperature variable. Do not call it air temperature. |
| snow particle mean diameter | Modeled particle-size summary. Do not call it a particle size distribution unless a distribution is actually represented or analyzed. |
| snow particle mean velocity | Modeled particle-velocity summary. Do not call it fall velocity unless direction and vertical fall are explicitly defined. |
| snow mass flux | Forecast target with manuscript units $\mathrm{kg\,m^{-2}\,s^{-1}}$. Do not silently convert to $\mathrm{g\,m^{-2}\,s^{-1}}$. |
| snow particle properties | Safe collective term for modeled mean diameter and mean velocity. **Particle microstructure** is too strong for the current variables. |
| atmosphere--snow interaction | Use an en dash in rendered text because the phrase relates two coordinate concepts. |

## 18. Mathematical symbol registry

| Symbol | Canonical meaning | Critical distinction |
|---:|---|---|
| $x_t$ | Simulated physical state at time $t$. | Not scheduler observation $o_t$. |
| $m$ | Number of logical sensing channels. | Not number of actions. |
| $a_{\mathrm{met}}$ | Meteorological-core entry of a candidate activation vector; fixed to 1. | Not a separate condition label. |
| $\mathcal S$ | Optional-channel index set. | Do not confuse with the operating-condition set $\mathcal C$. |
| $n_{\mathrm{opt}}=\lvert\mathcal S\rvert$ | Number of optional channels. | Not optional-channel limit $q$. |
| $q$ | Optional-channel limit; capacity parameter in the broader formulation. | Reported value is 1. |
| $a_t$ | Executed activation vector. | Not the proposal. |
| $u_t$ | Proposed action index sampled or selected from $\mathcal I_t$. | Stored for PPO likelihood; not an activation vector. |
| $\bar a_t=a^{(u_t)}$ | Proposed activation vector identified by $u_t$. | Not the executed vector. |
| $\mathcal A$ | Static candidate action set. | Not $\mathcal I_t$. |
| $K$ | Number of candidate activation vectors in $\mathcal A$. | Not number of logical channels $m$. |
| $a^{(k)}$ | Candidate activation vector indexed by $k$. | Not necessarily feasible at time $t$. |
| $\mathcal F_t$ | Set of candidate activation vectors satisfying both instantaneous power limits. | Does not encode dwell retention. |
| $\mathcal I_t$ | Indices $\{k:a^{(k)}\in\mathcal F_t\}$ of currently feasible candidates. | Contains all six indices in the reported instance. |
| $p_i$ | Steady-state power of channel $i$. | Not startup peak. |
| $p_i^{\mathrm{peak}}$ | Startup-peak power of channel $i$. | Applied on turn-on relative to $a_{t-1}$. |
| $B$ | Steady-state power limit. | Reported limit 0.75. |
| $B^{\mathrm{peak}}$ | Startup power limit. | Reported limit 0.95. |
| $D_{\min}$ | Minimum dwell time in steps. | Reported value 6. |
| $r_t$ | Remaining required activation time for the previously executed subset. | Environment guard; not in the policy input. |
| $o_t$ | Scheduler observation. | Not the latent simulator state. |
| $m_t$ | Valid-observation mask. | Not the action mask. |
| $c_t$ | Current operating-condition label. | Training/evaluation metadata, not online policy input. |
| $c_t^{\mathrm{sup}}$ | Future-condition supervision label. | Not the warning estimate or current condition label. |
| $q_{t,c}$ | Warning score for event type $c$. | Not a calibrated probability unless demonstrated. |
| $\widehat c_t$ | Thresholded warning-rule condition. | Not true $c_t$. |
| $\mathcal C$ | Set of four operating conditions. | Do not use for all channels. |
| $S_c^\star$ | Optimal optional-channel subset for condition $c$ in the broader formulation. | Not the supervised condition-to-action mapping. |
| $\rho_c$ | Positive condition weight in $L_\rho$, with $\sum_c\rho_c=1$. | Not PPO ratio $\rho_t(\theta)$. |
| $\mathcal C_{\mathrm{mis}}(s)$ | Conditions whose optimal optional channel differs from fixed channel $s$. | Not all operating conditions automatically. |
| $\Delta_c$ | Positive normalized excess-loss lower bound for a mismatched condition in Proposition 1. | Empirical paired difference $d_s$. |
| $\mathcal Y$ | Forecast-target set. | Separate from $\mathcal V_{\mathrm{AoI}}$. |
| $\mathcal V_{\mathrm{AoI}}$ | Observed-variable age set. | Separate from forecast targets. |
| $w_k$ | Forecast-horizon weight. | Uniform in the reported experiment; not a condition weight. |
| $\alpha_{j,c}$ | Condition-specific target weight. | Not training condition weight $\omega_c$. |
| $\sigma_j$ | Positive target scale used in capped absolute-error normalization. | Not validation normalizer or seed-level difference. |
| $\omega_c$ | Condition-specific scalar weight in $L_{\mathrm{train}}$. | Not target weight $\alpha_{j,c}$. |
| $s_c$ | Median event-type loss across feasible fixed subsets on calibration/validation data. | Not target scale $\sigma_j$. |
| $s_{\mathrm{calm}}$ | Median overall loss across the same fixed subsets; calm training normalizer. | Not a calm-only median unless separately computed. |
| $L_{\mathrm{step}}(t)$ | Per-step forecast loss. | Pre-normalization. |
| $L_{\mathrm{train}}(t)$ | Condition-normalized training loss. | Not the test macro endpoint. |
| $R_t$ | Post-execution policy-training reward. | Not scheduler input. |
| $S_t$ | Executed changed-channel fraction. | Normalized Hamming distance. |
| $W_t$ | Warm-up-abort indicator or penalty quantity as defined. | Structurally zero in the reported setup. |
| $\pi_\theta$ | Categorical neural policy. | Not the complete scheduler. |
| $\pi_s$ | Fixed-channel policy that always selects optional channel $s\in\mathcal S$. | Does not include the core-only policy. |
| $\pi_{\mathrm{label}}$ | Theoretical privileged policy with condition-label access. | Not automatically identical to the empirical true-label baseline. |
| $\widehat A_t$ | GAE advantage estimate. | Based on executed transitions. |
| $\rho_t(\theta)$ | PPO ratio for the stored proposed action index. | Not executed-action likelihood. |
| $u_t^{\mathrm{target}}$ | Feasible supervised target action index. | Not proposed $u_t$ or executed $a_t$. |
| $\ell_t$ | Indicator of a valid supervised target action. | Not remaining activation time. |
| $\eta_t$ | Indicator of an available future-condition supervision label. | Not $\ell_t$ or an event flag. |
| $\mathcal L_{\mathrm{clip}}$ | Negative PPO clipped surrogate used in minimization. | Not policy reward. |
| $\mathcal L_V$ | Value-function loss. | Not forecast loss. |
| $\mathcal L_{\mathrm{BC}}$ | Advantage-weighted behavior cloning loss. | Gated by $\ell_t$. |
| $h_t$ | Policy representation used by the auxiliary classifier. | Not scheduler observation $o_t$ itself. |
| $q_\theta(c\mid h_t)$ | Auxiliary probability assigned to operating condition $c$. | Not warning score $q_{t,c}$. |
| $\mathcal L_{\mathrm{cond}}$ | Auxiliary future-condition classification loss. | Not policy reward or online condition input. |
| $\mathcal L_{\mathrm{PPO}}$ | Composite training loss. | Not forecast loss. |
| $L_c$ | Condition-specific test forecast loss. | Must name the condition and support. |
| $\mathcal T_c$ | Evaluated test time steps labeled as event type $c$. | Not the full test partition. |
| $L_{\mathrm{macro}}$ | Macro-averaged normalized event forecast loss. | Excludes calm. |
| $\widetilde L_c(\pi)$ | Normalized condition-specific loss in Proposition 1. | Not raw $L_c$. |
| $L_\rho(\pi)$ | Weighted sum $\sum_c\rho_c\widetilde L_c(\pi)$. | Not automatically the empirical equal-weight macro endpoint. |
| $d_s$ | Seed-level comparator-minus-PD-PPO difference. | Positive favors PD-PPO. |
| $J_{\mathrm{fcst}}(\pi)$ | Theoretical discounted forecast objective. | Not empirical mean loss. |
| $T$ | Common finite horizon used by all three theoretical objectives. | Not forecast horizon $H$. |
| $\gamma$ | Common discount factor for the three theoretical objectives. | Not GAE parameter $\lambda$. |
| $J_{\mathrm{AoI}}(\pi)$ | Negative discounted AoI objective. | Indexed by $\mathcal V_{\mathrm{AoI}}$. |
| $\mathrm{age}_{t,j}$ | Age of observed variable $j$ at time $t$ in the theoretical AoI objective. | Channel sampling opportunity age. |
| $P_t$ | Posterior covariance of the separate theoretical estimator. | Not a main-policy input. |
| $J_{\mathrm{cov}}(\pi)$ | Negative discounted covariance-trace objective. | Not empirical diagonal proxy without qualification. |

## 19. House style, capitalization, and typography

### 19.1 Language and voice

- Use American English: **behavior**, **normalization**, **modeling**, **analyze**.
- Prefer impersonal active constructions: **The scheduler executes...**, **The analysis uses...**, **This study evaluates...**.
- Avoid **we**, **our**, and reader-steering phrases such as *It is important to note that*.
- Do not vary technical terms merely to avoid repetition. Precision outranks lexical variety.
- Use causal verbs only for experimentally isolated causal claims. Prefer **is associated with**, **the configuration achieved**, or **the comparison showed** when causality is not identified.

### 19.2 Hyphenation

Use these forms consistently:

| Hyphenated or en-dash form | Open or closed form |
|---|---|
| Prediction-Driven | behavior cloning |
| minimum dwell-time constraint | dwell time |
| calibration/validation | sensor scheduling |
| forecaster-training partition | time series forecasting |
| policy-training partition | forecast target |
| validation-selected fixed-schedule baseline | forecast loss, as a noun |
| condition-specific | measurement error |
| event-balanced | observation history |
| post-selection | startup |
| co-primary | warm-up |
| one-sided | look-ahead |
| one-vs-rest | sample-and-hold |
| multi-horizon | steady-state |
| seed-level | round-robin |
| rule-based | Double DQN |
| value-based | ROC AUC; PR AUC |
| condition--action, in LaTeX for a rendered en dash | Spearman rank correlation |
| Kolmogorov--Smirnov, in LaTeX | power spectral density |
| actor--critic, in LaTeX | online; offline |
| continuous--discrete, in LaTeX | componentwise; nonempty |
| particle-size--wind, in LaTeX | pretraining; postprocessing; fine-tuning |
| power--performance, in LaTeX | startup; trade-off |

Use an en dash for relationships and ranges in rendered text: **condition–action**, **power–performance**, **119–140**. In LaTeX source, use `--`. Use a hyphen for lexical compounds: **fixed-schedule**, **warm-up**.

This table intentionally resolves two generic recommendations in `docs/08-02-01-word.md`: keep the fixed protocol labels **forecaster-training partition** and **policy-training partition** hyphenated, but keep the established term **behavior cloning** open in **behavior cloning pretraining**, **advantage-weighted behavior cloning loss**, and **continuing behavior cloning term**. The rules in this contract supersede conflicting style-only suggestions in the editorial source.

### 19.3 Capitalization and pluralization

- Capitalize the proper method name: **Prediction-Driven Proximal Policy Optimization**.
- Lowercase generic concepts: **scheduler**, **policy**, **frozen forecaster**, **behavior cloning**.
- Write **Figure**, **Table**, **Equation**, **Section**, and **Appendix** when a numbered reference follows; use `\Cref` or an equivalent typed reference.
- Write **time step** as a noun and **time-step** as an attributive modifier.
- Write **seed 119**, **22 seeds**, and **seeds 119--140**.
- Write **channel subset** when referring to selected logical channels; use **activation mask** when the binary vector representation matters.

### 19.4 Numbers and units

- Use numerals with units: **16 time steps**, **0.75 power units** if units are normalized, **95% CI**.
- State whether power is normalized or physical before attaching units.
- Keep a leading zero for decimals: **0.05**, not **.05**.
- Define the denominator for percentages and relative reductions.
- Use consistent mathematical units, for example $\mathrm{kg\,m^{-2}\,s^{-1}}$ and $\mathrm{m\,s^{-1}}$.

## 20. High-risk substitutions

| Avoid | Use instead | Reason |
|---|---|---|
| PD-PPO scheduler is trained | PD-PPO policy is trained; the complete configuration is evaluated | Separates method from trainable object |
| the policy executes $u_t$ | the policy selects/proposes index $u_t$, which identifies $\bar a_t$; the scheduler executes $a_t$ | Preserves index/vector and proposal/execution distinctions |
| validation partition | calibration/validation partition | Combined partition has calibration and selection roles |
| supervised actions are computed on calibration/validation rows | calibration/validation defines the mapping; the mapping constructs targets on policy-training indices | Correct chronological and computational roles |
| look-ahead supervision label | future-condition supervision label | Separates training supervision from the privileged one-step look-ahead reference |
| candidate-set cardinality constraint | optional-channel cardinality constraint | The constraint limits active optional channels, not $\lvert\mathcal A\rvert$ |
| mean paired improvement, as the estimand | mean paired difference; describe a positive estimate as an observed improvement only with its comparator | The signed difference may be negative |
| behavior-cloning pretraining | behavior cloning pretraining | Keeps the established compound **behavior cloning** open |
| forecasting backbone | frozen forecaster | States the operational property that matters |
| reward oracle | frozen forecaster, or frozen-forecaster loss | Avoids implying truth access |
| minimum activation duration | minimum dwell time; minimum dwell-time constraint | Standard switched-system term and correct action-level semantics |
| minimum on-time | minimum dwell time | Current rule retains the entire action, not individual on channels |
| current power ratio | current steady-state power ratio | Names the numerator and denominator class |
| changed-activation fraction | mean normalized Hamming distance between adjacent executed masks | Gives the actual computation |
| transition entropy | entropy of consecutive executed-action pairs | Avoids implying conditional entropy |
| action entropy | executed-action entropy or policy entropy | Distinguishes empirical actions from policy distribution |
| condition--channel MI | condition--executed-action mutual information | Matches the implemented mask-label MI; shorthand allowed only for $q=1$ |
| sufficient observation dependence | report the exact dependence criterion and threshold | Removes undefined adjective |
| common channel-use diagnostic | channel-use acceptance criterion, followed by its four thresholds | Makes the frozen criterion reproducible |
| intermediate overall selection frequency | intermediate channel activation frequency | Replaces vague wording |
| challenge set | event-balanced evaluation window | Avoids implying an external benchmark |
| reward ablation | matched PPO training configurations with different scalar objectives | Components besides the scalar reward pathway also change |
| PPO outperformed DQN | the complete PPO configuration achieved lower loss than the matched Double DQN configuration | Prevents isolated algorithm attribution |
| no difference / equivalent | CI included zero; no population-level directional difference was established | Correct inferential scope |
| field-deployable warning system | simulator-generated warnings assumed available at decision time | Preserves simulation boundary |
| true-label policy | privileged true-label baseline, on first use | Discloses unavailable information |
| one-step greedy baseline | privileged one-step look-ahead reference using next-step test targets, on first use | Discloses unavailable information in the label |
| every fixed-channel policy | fixed policies that always select one optional channel | Matches Proposition 1 |
| power mask was effective | all six candidates remained feasible; the limits were enforced but non-excluding | Matches frozen diagnostics |
| particle fall velocity | particle velocity | Direction is not established |
| particle microstructure | particle diameter and particle velocity, or the explicitly defined simulated particle latent variable | Uses modeled quantities |
| pyranometer | basic radiometer channel | Product/measurand subclass is unverified |
| FlowCapt FC4 sensor | FC4 snow mass-flux channel | Keeps the logical-channel abstraction unless product transfer is supported |
| Double-DQN | Double DQN | Standard method name |
| ROC-AUC / PR-AUC | ROC AUC / PR AUC | Canonical metric style |
| time-series forecasting | time series forecasting | House style for the noun phrase |
| FW-MAE | mean forecast loss or the exact current endpoint name | FW-MAE belongs to an earlier manuscript branch and is not a current co-primary endpoint |
| SCENEBAL-2 and internal run tags | reader-facing experiment or scenario name | Internal development identifiers are not manuscript terminology |

## 21. Figure, table, and anonymous-submission terminology

### Figure 1

- Authoritative source: `paper/figures/figure_pdppo_framework.drawio`.
- Exported PDF/PNG files are derived assets, not editing sources.
- Panel A Stage 3: **Calibration/validation**.
- Full prose form: **calibration/validation partition**.
- Non-text topology gate: 82 cells unless a separately authorized structural revision is made.
- Keep proposed and executed actions visually distinct where both are shown.

### Supplementary split timeline

- Narrow-box abbreviation: **Cal./val.**
- Caption and surrounding prose: **calibration/validation** or **calibration/validation partition**.

### Figure 4

- Panel B title: **Validation-selected fixed channels**.
- Panel C title: **Executed-action diagnostics**.
- Do not shorten these labels if shortening obscures the baseline-selection or execution distinction.

### Generated assets

- Authoritative generator: `scripts/95_v31_build_clean_paper_assets.py`.
- Change generated scientific tables and figures through the generator and frozen inputs, not by editing exported PDF, PNG, or generated TeX alone.
- Preserve frozen bootstrap settings unless scientific evidence is intentionally regenerated:
  - `--bootstrap-samples 100000`
  - `--bootstrap-seed 20260718`

### Release and repository verification

- Canonical repository test command: `conda run -n darts pytest -q tests`.
- A synchronized staging directory is not, by itself, a verified submission package.
- Any rebuilt submission package must pass fresh extraction, path-safety and duplicate-member checks, archive CRC checks, member/hash comparison, independent compilation from the extracted source, repository tests, anonymity checks, and checksum-sidecar verification.
- Do not commit or push a manuscript-polish round unless the user explicitly requests it.

### Anonymous submission

- Do not introduce author names, affiliations, acknowledgments, usernames, absolute local paths, private repository identifiers, or credentials into anonymous artifacts.
- Replace any credential accidentally encountered with `[REDACTED]`; do not reproduce it in review notes or this glossary.
- Internal run tags and code identifiers may appear only when needed for reproducibility and when they do not compromise anonymity; prefer reader-facing experiment names in prose.

## 22. Per-round pre-edit checklist

Before editing:

- [ ] Read Sections 2--4 and the relevant concept tables in this file.
- [ ] Identify whether each edited sentence concerns the scheduler, policy, complete training configuration, frozen forecaster, or a baseline.
- [ ] Identify whether each action is proposed or executed.
- [ ] Identify the information available online, training-only, calibration/validation-only, offline, or privileged.
- [ ] Identify the exact temporal support and seed stratum behind every result.
- [ ] Identify the endpoint, comparator, and sign convention behind every number.
- [ ] Confirm that forecast-target and AoI-variable index sets remain separate.

During editing:

- [ ] Use canonical first-use forms and allowed short forms.
- [ ] Do not use synonyms for variation when they blur concept identity.
- [ ] Preserve the calibration/validation versus policy-training role distinction.
- [ ] Preserve the proposed-action likelihood versus executed-transition convention.
- [ ] Do not attribute the complete result to an isolated training component.
- [ ] Disclose privileged information for true-label and one-step look-ahead references.
- [ ] Disclose the simulation assumption and information content of warning signals.
- [ ] Avoid product-level instrument claims not represented by the implementation.

Before accepting the revision:

- [ ] Search active TeX for deprecated or high-risk terms in Section 20.
- [ ] Check first-use acronym definitions in every independently readable component.
- [ ] Check capitalization, hyphenation, en dashes, units, and seed ranges.
- [ ] Check that CIs containing zero are not described as significant, equivalent, or directionally conclusive.
- [ ] Check that 22-seed primary and 24-seed descriptive results are not conflated.
- [ ] Check that Figure 1 edits, if any, originate in the Draw.io source and preserve the topology gate.
- [ ] Check that generated-asset edits, if any, originate in Script 95 and verified frozen inputs.
- [ ] Check anonymous artifacts for identifying information and credentials.
- [ ] If a submission package changed, verify it from a fresh extraction rather than from the source worktree or staging directory alone.
- [ ] Run `conda run -n darts pytest -q tests` when repository or generated-asset code is affected.
- [ ] Build and visually inspect all affected PDF pages before declaring the round complete.

## 23. Maintenance rule

This file has a stable path so every future polish round can import it directly. Update it only when at least one of the following changes:

- an implemented method role or information flow;
- a mathematical definition or symbol domain;
- the partition protocol;
- the frozen claim/evidence boundary;
- a baseline's information access;
- an endpoint definition;
- a verified reader-facing nomenclature decision.

Do not update it merely because a reviewer suggests stylistic variation. Record any new canonical term here before propagating it through the manuscript. When a new revision memo conflicts with this contract, resolve the conflict against the evidence hierarchy in Section 2 and document the decision.