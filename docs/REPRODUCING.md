# Reproducing results

Run commands from the repository root after installing the versions in `requirements-lock.txt`. Use Python 3.11, the supplied CSV partitions, process schemas and model artifacts together. Their checksums are in `data/inventory.json` and `artifacts/manifest.json`.

## Evaluation

```bash
python examples/reproduce.py --output runs/reproduction --particles 1000 --seed 42
```

The default run evaluates every individual process on its complete test partition and simulates the complete electrochemical test partition, using both uncalibrated and Kuleshov-calibrated distributions. No rows, repetitions or particle tails are filtered. Outputs are grouped by individual process and chain, with CSV predictions/metrics, paired chain particles and PDF/PNG figures.

For a short installation check:

```bash
python examples/reproduce.py --output runs/smoke --rows 2 --particles 20 --stages mixing electrochemical
```

This smaller run is a smoke check; it is not a full-partition performance estimate. Use `--partition development` for the combined training/validation cohort. Development rows are restored to stable observation-ID order before chain batching. Use `--partition train` or `--partition validation` for individual partitions.

## Point and interval definitions

For direct individual targets, the reproduction script uses analytic mixture quantiles at 5%, 50% and 95%. The reported numerical mean uses 61-point Gauss-Legendre integration over calibrated or raw mixture quantiles. It is a specified numerical summary, not a claim that a distribution has finite theoretical moments.

For conditional dried porosity, the script uses the paired propagated sample to calculate the mean, median and interval. The full protocol uses 1,000 particles and seed 42. Chain outputs use all saved particles, report mean and median scores separately, and use 16-row batches with seed `42 + first row index`. Changing row order, seed, batch size, calibration mode or particle count changes the numerical experiment. No final-chain calibration is fitted.

The supplied electrochemical StudentT/log1p distribution has no finite theoretical mean. Finite-particle means, numerical quadrature summaries and predictive medians must be labelled separately. Training/validation evaluations are development diagnostics, not external validation.

## Other calculations

```bash
python battwin.py optimise --rate .2 --chemistry NMC622 --output runs/design/0.2C_capped
python battwin.py optimise --rate 1 --chemistry NMC622 --uncapped --raw --output runs/design/1C_raw_uncapped
python battwin.py inverse --rate .2 --target 150 400 --output runs/design/inverse
python battwin.py pfi chain --repeats 5 --particles 100 --output runs/pfi
```

These commands execute new calculations with the supplied artifacts. The tutorial defines search budgets, constraints, target windows and acceptance rules. Record explicit configuration overrides for each result. The illustrative inverse target is user-specified, not an experimental optimum.

The generic training API and CLI train direct process ensembles with explicit grouped partitions and held-out calibration. The supplied coating/drying porosity artifact also includes a conditional parent-input recipe. Use the saved artifact for its numerical evaluation; a direct ensemble trained on a different feature matrix is a different model.

## Regression checks

```bash
python -B -m unittest discover -s tests -v
```

Tests cover distribution numerics, group leakage, calibration, conditional-input dependence, particle counts/pairing, actual training, searches and scrap accounting. `tests/reference/quantiles.json` and `chain_particles.csv` contain numerical regression references for the supplied example inputs. The accompanying provenance file records the particle count and seed. These checks verify implementation consistency; they do not replace a scientific evaluation on independent observations.
