# Framework Architecture Diagram

![Windblown framework architecture](assets/windblown_framework_architecture.svg)

## What this figure shows

This figure is the current engineering view of the `rl_sensor_scheduling_framework` pipeline.

- One shared truth environment is generated first.
- Multiple schedulers act on the same sensor bank under power constraints.
- Their actions are projected into feasible sensor subsets by the online projector.
- The resulting observations are fused by the Kalman estimator.
- Each scheduler produces its own forecast dataset.
- The same predictor family is then trained on each scheduler-specific dataset.
- Final comparison is done against the `full_open` oracle baseline.

## Paper-style design notes

If this figure is later redrawn in a presentation or by a design model, keep these constraints:

- Use a clean white or very light background.
- Use one color per stage family:
  - blue: data / environment
  - green: sensing / estimation
  - orange: scheduling / control
  - purple: optional auxiliary reward path
  - cyan: evaluation
- Keep the main pipeline left-to-right, then top-to-bottom.
- The online projector must be visually emphasized, because it is the key bridge between learned scores and physically feasible sensor subsets.
- Keep the optional reward-oracle path dashed, because it exists in code but is disabled by default in the current experiments.
- Make the distinction between:
  - primary reward targets
  - forecast targets
  explicit in the figure.

## If you want a prettier redraw

A cleaner publication redraw should preserve exactly these blocks:

1. Shared truth generation
2. Sensor bank + power constraints
3. Scheduler family
4. Online subset projector
5. Sensor replay + noisy observations
6. Kalman state estimation
7. Scheduler-specific dataset export
8. Forecasting model zoo
9. Primary targets / forecast targets
10. Evaluation and reporting

The most important semantic message is:

> the framework does not directly compare forecasting models on the same raw data only; it compares forecasting models after each scheduler has changed what information is available to the estimator and predictor.
