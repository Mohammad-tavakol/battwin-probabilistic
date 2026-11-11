"""Explicit run settings; importing the package never starts a run."""
from dataclasses import dataclass,asdict
from pathlib import Path
import os,json

@dataclass(frozen=True)
class SimulationConfig:
    particles: int=1000
    seed: int=42
    batch_size: int=16
    calibration: str='kuleshov'
    physical_porosity: bool=False

@dataclass(frozen=True)
class TrainingConfig:
    members: int=20
    seed: int=42
    bootstrap_mode: str='group'
    validation_fraction: float=.2
    selection_fraction: float=.2
    cv_folds: int=5
    outer_seeds: tuple=(1,42,100)
    outer_holdout_fraction: float=.10
    nthread: int=1

@dataclass(frozen=True)
class TuningConfig:
    trials: int=50
    top_candidates: int=2
    objective: str='R2'
    distributions: tuple=('Gaussian','StudentT','Gamma','LogNormal','Weibull','Laplace','Cauchy','Gumbel')
    transformations: tuple=('raw','log1p')
    seed: int=42
    screening_rounds: int=200
    minimum_rounds: int=100
    maximum_rounds: int=1000
    round_step: int=50
    accuracy_tie: float=.01

def project_root(path=None):
    if path is not None:return Path(path).expanduser().resolve()
    if os.environ.get('BATTWIN_PROJECT'):return Path(os.environ['BATTWIN_PROJECT']).expanduser().resolve()
    candidate=Path(__file__).resolve().parents[2]
    if (candidate/'configs').is_dir():return candidate
    raise ValueError('Pass the project directory, or set BATTWIN_PROJECT')

def stage_schema(stage,project=None):
    path=project_root(project)/'configs/processes'/f'{stage}.json'
    return json.loads(path.read_text(encoding='utf-8'))

def emit(callback,event,**details):
    if callback is not None:callback(dict(event=event,**details))
