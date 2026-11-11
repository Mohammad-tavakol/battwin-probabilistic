# BATTWIN Probabilistic: practical tutorial

This tutorial starts with the frozen calibrated models, then covers new training, chain propagation, interpretation and process design. Examples assume that the current directory is the package directory and that the environment described in the README is active. All paths below are relative to that directory.

## 1. Load and inspect the saved models

```python
from pathlib import Path
from battwin_probabilistic import Project, SimulationConfig
project = Project(Path.cwd())
print(project.catalog.registry().to_string(index=False))
model = project.model("mixing")
print(model.features)
print(model.targets)
```

Valid process keys are `mixing`, `coating_drying`, `calendaring` and `electrochemical`. `project.schema(stage)` returns exact feature names, categorical fields, targets, split groups, bootstrap groups and selected configurations. Supply these feature names exactly, including their units. The fitted preprocessing handles categoricals; do not independently encode them or apply a second target transform.

Each `artifacts/models/<stage>.joblib` contains the fitted process ensemble and its Kuleshov calibration maps. Calibration is applied once to the ensemble CDF. Passing `calibration="uncalibrated"` selects the same fitted weights without applying the maps. Kuleshov predictions use CDF transport without an additional spread multiplier.

## 2. Individual predictions and uncertainty

```python
import pandas as pd
x = pd.read_csv("examples/data/mixing_inputs.csv", float_precision="round_trip")
prediction = model.predict(x)
quantiles = model.quantiles(x, levels=(.05, .5, .95))
particles = model.sample(x, n_particles=400, seed=42)
prediction.to_csv("mixing_predictions.csv", index=False)
```

`predict` returns one row per input and target with `q05`, `median` and `q95`, in the original target units. `quantiles` returns arrays of shape input rows by requested probabilities. `sample` returns paired multi-target particles with `input_row`, `particle_id` and `member_id`. Use a particle count divisible by 20. The target distributions share a member assignment and have separate residual draws; this is the established ensemble dependence, not an independently fitted residual copula.

For a finite Monte Carlo mean, group the particles by `input_row`. The median and Monte Carlo mean are different point estimates. In particular, an exponentiated StudentT target has no finite theoretical mean. Its finite-particle average may vary substantially with particle count or seed; it must not be described as a converged population expectation.

```python
mean_prediction = particles.groupby("input_row")[model.targets].mean()
p_below = model.cdf(x, model.targets[0], values=10.0)
from battwin_probabilistic.uncertainty import decompose
decomposition = decompose(model, x, particles=1000, seed=42)
```

The decomposition computes the finite-particle law of total variance using equally represented members. Its aleatoric and epistemic components sum to the variance of those same sampled particles. It does not assert that every chosen distribution has finite theoretical moments.

### Formulation inputs

All component amounts are wt% of total slurry. The runtime calculates **solid content = active material + conductive additive + binder 1 + binder 2/thickener**. Binder 2/thickener is counted once; dispersant is separate. Complete compositions override a stale solid-content value. Invalid or nonfinite fractions, totals above 100%, and positive fractions with an absent material are rejected without rescaling.

```python
from battwin_probabilistic import COMPONENTS, prepare_formulation
x[COMPONENTS[0]] = x[COMPONENTS[0]] - 0.1
x = prepare_formulation(x)  # returns a validated copy with calculated solid content
prediction = model.predict(x)
```

The chain requires all four fractions and calculates solid content even if its column is omitted. Individual models also calculate it when all four fractions are supplied, including auxiliary columns outside their fitted feature vector. An individual coating/drying or calendaring table with only its native feature columns keeps its explicit solid-content input: the omitted fractions cannot be inferred. Solid density and target-specific engineered or conditional features are constructed by the supplied model recipes.

The current coating/drying export includes a conditional OOF-stack porosity model with predicted parents. Loading and inference apply this recipe automatically. The generic training examples below fit new direct ensembles; they do not reproduce the conditional architecture or replace its saved weights. Reproducing the conditional stack requires its grouped OOF-parent development protocol and target-specific recipes. The saved configuration documents the selected predictor; fitting the same hyperparameters on a different feature matrix is a different model.

## 3. Use and evaluate the frozen partitions

```python
data = project.data("mixing")
train, valid, test = data["train"], data["validation"], data["test"]
development = data["development"]
from battwin_probabilistic.training import evaluate_artifact
test_metrics = evaluate_artifact(model.artifact, test, point="median")
print(test_metrics.to_string(index=False))
```

The loader checks that no `condition_id` or schema-defined split group crosses partitions. `condition_id` uses the full feature vector. Repetitions remain individual observations. The mixing partition permits different shear conditions of a formulation in training and validation while keeping identical formulation/shear repetitions together. Test membership is fixed by the supplied snapshots. Other process group definitions follow their frozen schemas.

Metrics include R2, MAE, RMSE, MAPE, MdAPE, PICP90, MPIW90, NMPIW90, Winkler90 and WIS90. PICP90 is a fraction; MAPE and MdAPE are percentages. Percentage errors divide by observed target magnitude, so small viscosity observations can dominate MAPE. Always state which point definition and split are reported.

## 4. Train a new model without overwriting the frozen one

To refit using the already selected distribution, transformation and hyperparameters:

```python
new_artifact = project.train("mixing", "runs/mixing/refit_seed42")
```

This uses the frozen training rows to fit a 20-member group-bootstrap ensemble and the frozen validation rows to fit new CDF maps. Test rows are not fitted. The output goes to the explicit run directory.

For a new configuration search:

```python
from battwin_probabilistic import TrainingConfig, TuningConfig
artifact = project.train(
    "mixing", "runs/mixing/tuned",
    training=TrainingConfig(seed=42),
    tuning=TuningConfig(trials=50, top_candidates=2,
        distributions=("Gaussian", "Gamma"),
        transformations=("raw", "log1p")),
    use_selected=False,
)
```

The new-development path first reserves grouped validation rows for calibration. It splits the remaining development rows again for candidate selection. Preliminary fits compare the requested distribution/transform combinations and two stabilization settings. The top two distinct alternatives receive Optuna TPE hyperparameter searches, 50 trial attempts each by default. Hyperparameters include learning rate, depth, regularisation, row/column subsampling, minimum child weight, stabilization and boosting rounds. R2 is the default objective. Among alternatives within 0.01 of the best R2, lower Winkler90 wins. A separate `objective="width"` option minimises interval width without a coverage constraint, matching the earlier exploratory experiment; report its coverage alongside the width.

Selected configurations are frozen before the final bootstrap ensemble is fitted. The calibration holdout is not used in Optuna selection. SQLite studies in a run directory resume to the requested trial budget. Trial failures are recorded as pruned attempts. Use a new output directory for a different tuning protocol; the frozen artifacts remain untouched.

For external data, call `training.fit_process(development, schema, ...)`, or `train_process(train, validation, schema, configurations, ...)` with explicit frames. Use `data.annotate` to create condition/group identifiers. `partition="test"` rows are rejected by development APIs, but truthful partition labels and appropriate source provenance remain the caller's responsibility.

## 5. Grouped CV and repeated outer splits

```python
from battwin_probabilistic.training import cross_validate, repeated_splits
schema = data["schema"]
configs = schema["target_configurations"]
folds, mean_std = cross_validate(development, schema, configs,
    TrainingConfig(cv_folds=5))
outer = repeated_splits(development, schema, configs,
    seeds=(1, 42, 100))
```

CV scores belong to the held-out fold, not its training rows or the external test set. Each fold's fitting portion is internally divided into model fitting and spread/CDF calibration. No configuration selection occurs inside CV. The returned fold table and mean/std table use median-based point metrics. These scores use predictive medians; numerical means and Monte Carlo means are separate point definitions.

Outer splits reserve a grouped 10% holdout by default and a grouped calibration holdout within the remainder. Change `TrainingConfig(outer_holdout_fraction=.2)` if a different protocol is needed. These robustness runs refit ensembles and maps and can be expensive. They are not executed by importing the package or opening the notebook.

## 6. Recalibrate an existing artifact

```python
from battwin_probabilistic.training import recalibrate
from battwin_probabilistic.artifacts import save_artifact
updated = recalibrate(model.artifact, valid, project.schema("mixing"))
save_artifact(updated, Path("runs/mixing/recalibrated.joblib"))
```

For each held-out validation observation, the fitted mixture CDF produces a PIT value. Upper empirical ranks are fitted with monotone isotonic regression and anchored at 0 and 1. The calibrated CDF is G(y|x)=R(F(y|x)); calibrated quantiles use Q_G(p|x)=Q_F(R_inverse(p)|x). The generalised inverse uses the left edge of a plateau. Recalibration changes the predicted distribution and can change its median. It does not refit tree weights. Test observations must never determine the map.

Training-condition overlap is rejected when the artifact contains training-condition provenance. Use the supplied validation snapshot when training-condition identifiers are not recorded in an artifact.

## 7. Propagate a chain with a fixed particle population

```python
inputs = pd.read_csv("examples/data/chain_inputs.csv", float_precision="round_trip")
chain = project.chain(SimulationConfig(particles=400, seed=42))
summary, chain_particles = chain.simulate(inputs)
raw_chain = project.chain(SimulationConfig(particles=400,
    seed=42, calibration="uncalibrated"))
raw_summary, raw_particles = raw_chain.simulate(inputs)
```

Every input receives exactly 400 particles throughout the process chain. The code assigns a balanced set of ensemble members to those particles and draws one output per target per particle. Downstream stages consume the same particle states. It does not expand 20 branches at every stage. Multi-target results remain paired within particle rows.

The order is mixing, coating-drying, calendaring and electrochemical. Generated intermediate values replace their measured counterparts as the state advances. A high individual-model R2 obtained with measured inputs therefore does not guarantee the same chain R2: uncertain, biased upstream states pass through nonlinear downstream models. Calibration can change these upstream states and downstream point accuracy as well as coverage.

The process models use target-specific density and conditional feature recipes with held-out validation CDF maps. The default chain propagates learned porosities. Dried porosity conditions on predicted mass/thickness particles, and calendaring receives upstream dried porosity. `SimulationConfig(physical_porosity=True)` selects equation-based porosity for a separate physical-model comparison. Feature construction is embedded in each process contract and applied by the loader. In a thickness-only individual diagnostic, use measured mean mass loading and predicted thickness with `physics.porosity_from_thickness`; this is distinct from paired-particle equation propagation in the chain.

Particle count, seed and batch size all form part of the reproducibility protocol. The standard chain seeds each batch with `seed + batch_start`. Keep batch size fixed when comparing saved runs. No sampled output is silently clipped merely to improve a point score.

```python
from battwin_probabilistic.uncertainty import particle_summary
targets = project.model("electrochemical").targets
points = particle_summary(chain_particles, targets)
```

## 8. Compare individual and chain permutation importance

```python
from battwin_probabilistic.pfi import process_importance, chain_importance
ec_train = project.data("electrochemical")["train"]
individual_raw, individual_pfi = process_importance(
    project.model("electrochemical"), ec_train, repeats=5)
chain_raw, chain_pfi = chain_importance(chain, ec_train,
    repeats=5, particles=100)
```

Importance is baseline R2 minus R2 after one feature is permuted, using predictive medians. Chain evaluations reuse the same Monte Carlo seed schedule for baseline and permutations; permutation seeds change across five repetitions. Report mean and standard deviation. Negative importance is retained. Measured intermediate columns that are overwritten by the chain can have zero chain importance even when important to the standalone electrochemical model. Marginal permutations can also break correlations, so interpret rankings in the context of the process definitions.

The separate `sensitivity.compute_chain_pfi` preserves the original simulation-only analysis of shifts in distribution summaries. Its R2 compares permuted predictions against baseline predictions, rather than against measured targets. Label it separately from supervised PFI.

## 9. Capped and uncapped optimisation at 0.2C

```python
from battwin_probabilistic.design_space import SearchConfig
from battwin_probabilistic.optimisation import optimise_capacity
development_ec = project.data("electrochemical")["development"]
search = optimise_capacity(development_ec,
    SearchConfig(chemistry="NMC622", discharge_rate=.2,
        apply_capacity_caps=True),
    project.catalog, "runs/design/0.2C_capped")
```

The objectives simultaneously maximise finite Monte Carlo mean gravimetric and volumetric capacity. Development profiles supply compatible material identities. Four machine settings and the four component fractions vary within observed development bounds; roll speed and temperature use observed discrete levels. Every optimisation and inverse proposal recalculates solid content from its actual fractions, without rescaling them. Totals outside (0,100] are rejected; calculated solids remain inside observed support unless formulation extrapolation is explicitly enabled. Shear is derived using each source anchor's convention.

```python
config = SearchConfig(discharge_rate=.2,
    formulation_bounds={COMPONENTS[0]: (64.0,64.2),
                        COMPONENTS[1]: (1.331,1.339)})
# Pass config to optimise_capacity or Experiment for inverse design.
```

Omitted component bounds use the observed range. `vary_formulation=False` retains source recipe proportions. `allow_formulation_extrapolation=True` permits explicit bounds outside observed support, while percentage and total limits still apply. These fields also work in CLI `--config` JSON. Solid content is derived and cannot be an independent bound. PFI similarly uses independent components when the full composition is available; their importance includes their solid-content effect. Use a fresh run directory: caches include the accounting implementation and incompatible completed inverse results are rejected rather than reused.

All candidates must satisfy intermediate-output support, physical-pass and tail-stability gates. In capped mode, each predicted mean capacity must also be no larger than the corresponding observed development maximum for the chosen chemistry/rate. These are constraints on means, not clipping of particle values. Uncapped mode removes those two empirical capacity caps and retains the remaining gates.

The standard search uses 128 initial random specifications plus observed paths, three generations of 64 offspring, 80 particles for screening, 1000 for refinement and two independent 2000-particle confirmation runs. The default physical-pass threshold is 80%; maximum single-particle capacity shares are 10% during screening and 2% during refinement/verification. A normalised Pareto knee or closest-to-utopia fallback selects a balanced target. Additional feasibility search can run if the first pass finds no confirmed target. A failed budget is reported honestly; it is not proof that no feasible design exists and no acceptance gate is relaxed.

## 10. Inverse design and independent verification

```python
from battwin_probabilistic.inverse_design import inverse_design
if search.target is not None:
    alternatives = inverse_design(search, min_alternatives=10,
        max_rounds=6)
```

An explicit positive `[gravimetric, volumetric]` target may also be supplied. The retained algorithm searches globally and around nearby supported designs with multiple perturbation scales. A proposed alternative must satisfy eligibility, pooled target matching and target matching in each of two independent verification seeds. The established tolerance is +/-2% for both mean capacities. Reaching the search budget with fewer than ten verified alternatives is a valid outcome, not a reason to fabricate alternatives.

The design cache signature includes data, model checksums and implementation files. Keep one output directory per chemistry/rate/calibration/cap/protocol combination. Saved tables expose the search evidence and eligibility failures for later GUI display.

## 11. Scrap policies and tolerance windows

```python
from battwin_probabilistic.scrap import assess_scrap
scrap = assess_scrap(chain_particles, target=[150.0, 400.0],
    tolerance_pairs=((.02,.02),(.05,.05),(.10,.10)),
    output_dir="runs/scrap")
```

Replace the illustrative capacity target with the actual engineering target. Each tolerance pair specifies intermediate and final relative tolerances. The retained default centres intermediate windows on each path's predicted mean; final windows use the explicit common target. Use identical window definitions whenever comparing methods. This analysis is different from experimental-reference OOT validation.

Under direct rejection, a particle is counted once at the first stage where any applicable tolerance or physical rule fails. Under flag-only final disposition, intermediate failures are recorded and the particle continues; final rejection is determined by the final electrochemical gate. Reports include total yields, stage scrap, arrival-conditioned rejection, parameter flags and recovery. Later stages of a directly rejected particle are retained for counterfactual diagnostics but do not count as production arrivals.

## 12. Validate out-of-tolerance probabilities against repetitions

```python
from battwin_probabilistic.validation import experimental_oot
test_ec = project.data("electrochemical")["test"]
# First simulate ALL test_ec rows with both chains, preserving row order.
# raw_particles and chain_particles must correspond to those rows.
features = project.schema("electrochemical")["split_group_features"]
manufacturing = [c for c in features if c != "Discharge rate (*C)"]
result = experimental_oot(test_ec,
    {"Full chain": {"Uncalibrated": raw_particles,
                    "Kuleshov": chain_particles}},
    features, manufacturing, targets,
    reference="experimental_mean", bootstrap=5000)
```

The experimental mean fixes the same tolerance centre for all methods. Singletons are excluded; `leave_one_out` uses conditions with at least three repetitions and `leave_one_out_pairs` reports the two-repetition sensitivity separately. Joint failure is the paired union of capacity failures, not the product of marginal acceptance probabilities. Conditions receive equal weight and uncertainty resamples manufacturing groups. The implementation uses paired Monte Carlo particles for both marginal and joint model probabilities; analytic marginal-CDF probabilities can differ by Monte Carlo error.

This is an external repetition/reference diagnostic, not a fitted calibration step and not a factory scrap-rate estimate. Do not tune a calibration map to these test observations.

## 13. Export figures and organise runs

```python
from battwin_probabilistic import reporting
test_prediction = model.predict(test)
p = test_prediction[test_prediction.target.eq(model.targets[0])]
fig = reporting.parity(test[model.targets[0]].to_numpy(),
    p["median"].to_numpy(), "Viscosity (Pa.s)", log=True,
    path="runs/mixing/figures/parity")
```

Exports use 6-inch width, 4.5-inch height, no title, axis labels of at least 14 points, and PDF plus 400-dpi PNG. Logarithmic value axes are appropriate for mixing viscosity; use linear axes for other process outputs. Sort uncertainty-band inputs explicitly if a ranked comparison is intended. Parallel-coordinate plots can use a wider canvas for readability.

Recommended results layout is `runs/<process-or-analysis>/<run>/reports`, `figures/<topic>/combined`, `figures/<topic>/subfigures` and `cache`. Filenames should describe target, split, method and chart type. The package never creates a large shared folder of every plot.

## 14. Command-line and application integration

```powershell
python battwin.py --help
python battwin.py train mixing --tune --trials 50 --output runs/mixing/new_search
python battwin.py cv calendaring --folds 5 --output runs/calendaring/cv
python battwin.py pfi chain --repeats 5 --particles 100 --output runs/pfi/train
python battwin.py optimise --rate .2 --chemistry NMC622 --output runs/design/capped
python battwin.py inverse --rate .2 --target 150 400 --output runs/design/inverse
```

Long-running APIs accept a callback receiving progress dictionaries. The CLI prints those events as JSON. A GUI should call these functions in a worker, forward progress and keep the selected project/run paths explicit. The numerical logic is independent of the display layer.

Before sharing a run, record artifact checksums, data provenance, package/dependency versions, calibration mode, physical-porosity mode, seed, batch size and particle count. Run the supplied tests after an environment upgrade; joblib model files depend on compatible Python/XGBoostLSS/XGBoost versions. Only load trusted joblib/pickle files.


## Reproduction protocol

See [REPRODUCING.md](REPRODUCING.md) for complete-partition evaluation, point definitions, fixed simulation settings and regression references.
