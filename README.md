# BATTWIN Probabilistic

Code, data and fitted models for probabilistic electrode-manufacturing analysis. The framework links mixing, coating/drying, calendaring and electrochemical models through paired Monte Carlo particles. It supports predictive uncertainty, Kuleshov calibration, optimisation, inverse design, feature importance and model-based scrap assessment.

## Install

Use Python 3.11 and the supplied dependency versions for reproducibility. Create and activate a virtual environment, then run from the repository root:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python battwin.py inspect
```

The repository includes all four fitted process artifacts and the train, validation and test partitions. It has no dependency on external project directories or notebooks.

## Run predictions and the process chain

```bash
python battwin.py predict mixing examples/data/mixing_inputs.csv --output runs/mixing/predictions.csv
python battwin.py simulate examples/data/chain_inputs.csv --particles 400 --seed 42 --output runs/chain
```

```python
from pathlib import Path
import pandas as pd
from battwin_probabilistic import Project, SimulationConfig

project = Project(Path.cwd())
inputs = pd.read_csv("examples/data/mixing_inputs.csv")
predictions = project.model("mixing").predict(inputs)

chain_inputs = pd.read_csv("examples/data/chain_inputs.csv")
summary, particles = project.chain(
    SimulationConfig(particles=400, seed=42, batch_size=16)
).simulate(chain_inputs)
```

`predict()` returns the predictive median and 5th/95th percentiles. `sample()` returns paired particles for probabilities and finite-sample summaries. Use `calibration="uncalibrated"` to evaluate the same ensembles without CDF calibration. Particle counts must be multiples of 20.

## Supplied models

| Process | Response | Distribution / target transformation |
| --- | --- | --- |
| Mixing | Viscosity | Gamma / raw |
| Coating/drying | Dry mass loading | Gaussian / log1p |
| Coating/drying | Dry thickness | Gaussian / raw |
| Coating/drying | Dried porosity | Gumbel / log1p, conditional predicted inputs |
| Calendaring | Calendered thickness | Gamma / raw |
| Calendaring | Calendered porosity | Gumbel / log1p |
| Electrochemical | Gravimetric capacity | Gamma / raw |
| Electrochemical | Volumetric capacity | StudentT / log1p |

Each target has 20 bootstrap ensemble members. The process schemas specify feature names, units, categorical variables, grouping rules, hyperparameters and seed. Validation data determine monotone Kuleshov CDF maps; test observations are excluded from fitting and calibration. Preprocessing, fitted members and calibration maps are stored together in each model file.

The chain propagates learned porosities, including dried porosity into calendaring. `SimulationConfig(physical_porosity=True)` enables the mass/thickness/density equation as a separate physical-model comparison. Engineered and conditional input recipes are constructed automatically.

Component inputs use wt% of total slurry. The Python formulation helper derives solid content from active material, conductive additive, binder 1 and binder 2/thickener; dispersant is separate. See the [input conventions](docs/TUTORIAL.md#formulation-inputs) and [slurry-density calculation](docs/SLURRY_DENSITY.md) for exact accounting and units.

## Reproduce numerical results

Follow [Reproducing results](docs/REPRODUCING.md) for partition definitions, point estimates, calibration modes, Monte Carlo settings and reference checks.

```bash
python examples/reproduce.py --output runs/reproduction --particles 1000 --seed 42
python -B -m unittest discover -s tests -v
```

The script evaluates the supplied artifacts and exports predictions, metrics and figures. Training, tuning and design searches are explicit operations described in the [tutorial](docs/TUTORIAL.md), its [PDF](docs/TUTORIAL.pdf) and the [workflow notebook](examples/workflow.ipynb).

## Contents

```text
src/battwin_probabilistic/   framework implementation and CLI
configs/processes/          process contracts and selected configurations
artifacts/models/           fitted ensembles and embedded calibration maps
artifacts/manifest.json     model checksums and artifact format
data/stages/                train, validation and test CSV partitions
data/inventory.json         partition counts and checksums
examples/                   notebook, reproduction script and example inputs
docs/                       methods, usage and reproduction instructions
tests/                      scientific checks and numerical references
```

The [scientific conventions](docs/ARCHITECTURE.md) define calibration, pairing, grouped evaluation and scrap accounting. Generated results belong under `runs/` and are excluded from version control.

Report medians and finite-particle means separately. The exponentiated StudentT model has no finite theoretical mean; its Monte Carlo averages depend on particle count and seed. Numerical reproduction does not establish external experimental validation.

## Licence

The project's original code, documentation and fitted models are licensed under the
[Apache License, Version 2.0](LICENSE). Academic and commercial use, modification
and redistribution are permitted subject to the licence terms.

Third-party dependencies and source datasets retain their respective licences and
attribution requirements. This licence does not replace those terms.
