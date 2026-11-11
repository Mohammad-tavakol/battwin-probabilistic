"""Existing bounded design variables, Pareto selection and search settings."""
from dataclasses import dataclass,asdict,field
import numpy as np
from .formulation import COMPONENTS,SOLIDS
GRAV = "Gravimetric capacity (mA h/g)"
VOL = "Volumetric capacity (mA h/cm3)"
DRY_THICK = "Dry Coating Thickness (\u00b5m)"
OUTPUTS = {
    "Mixing": ["viscosity"],
    "Drying": ["Mass loading dry (mg/cm2)", DRY_THICK, "Dried Porosity (%)"],
    "Calendaring": ["Calendered thickness (um)", "Calendered porosity (%)"],
    "Electrochemical": [GRAV, VOL],
}
CONTINUOUS = ["Coating Speed (m/min)", "Coating Gap (micron)",
              "Coating ratio (%)", "Calendaring Gap (um)"]
DISCRETE = ["Calendering Roll speed(m/min)", "Calendaring Roll temp"]
CONTROL_NAMES = [*COMPONENTS, "Mixing Speed", *CONTINUOUS[:3],
                 "Drying temp-Zone1", "Drying temp-Zone2", *DISCRETE,
                 "Calendaring Gap (um)"]

@dataclass
class SearchConfig:
    chemistry: str = "NMC622"
    discharge_rate: float = 1.0
    apply_capacity_caps: bool = True
    seed: int = 20261008
    initial_random: int = 128
    generations: int = 3
    offspring: int = 64
    inverse_random: int = 256
    screen_particles: int = 80
    refine_particles: int = 1000
    verify_particles_per_seed: int = 2000
    max_refine: int = 24
    max_verify: int = 16
    batch_size: int = 8
    target_tolerance: float = 0.02
    intermediate_tolerances: tuple = (0.02, 0.05, 0.10)
    min_physical_pass: float = 0.80
    max_tail_share_screen: float = 0.10
    max_tail_share_refine: float = 0.02
    max_tail_share_verify: float = 0.02
    output_name: str = "optimisation_results"
    vary_formulation: bool = True
    formulation_bounds: dict = field(default_factory=dict)
    allow_formulation_extrapolation: bool = False

def pareto_mask(values):
    """Non-dominated finite points for simultaneous maximisation of two KPIs."""
    values = np.asarray(values, float)
    result = np.zeros(len(values), dtype=bool)
    for i, point in enumerate(values):
        if np.isfinite(point).all():
            result[i] = not np.any(np.all(values >= point, axis=1) &
                                   np.any(values > point, axis=1))
    return result

def balanced_knee(values):
    """Knee towards the utopia point in normalised objective space.

    Falls back to closest-to-utopia for a straight/two-point front. A one-point
    front is explicitly a jointly best point, not an invented trade-off.
    """
    values = np.asarray(values, float)
    if not len(values):
        raise ValueError("No eligible Pareto point")
    if len(values) == 1:
        return 0, "single non-dominated point; no resolved trade-off"
    scale = np.ptp(values, axis=0)
    norm = (values - values.min(axis=0)) / np.where(scale > 0, scale, 1)
    if np.all(scale > 0) and len(values) >= 3:
        distance = norm.sum(axis=1) - 1
        if distance.max() > 1e-8:
            best = np.flatnonzero(np.isclose(distance, distance.max()))
            return int(best[np.argmin(np.linalg.norm(1 - norm[best], axis=1))]), "normalised front knee towards utopia"
    return int(np.argmin(np.linalg.norm(1 - norm, axis=1))), "balanced closest-to-utopia fallback"

def within_target(means, target, tolerance):
    means = np.asarray(means, float)
    target = np.asarray(target, float)
    return np.isfinite(means).all(axis=-1) & np.all(
        np.abs(means / target - 1) <= tolerance + 1e-12, axis=-1)

def derived_shear(anchor, speed, gap, coating_ratio):
    """Preserve the source row's gap convention while updating machine settings.

    WMG rows use speed/gap; mona coating rows use speed/(gap*ratio).
    Infer the convention from the supported anchor instead of mixing definitions.
    """
    factor = float(anchor['Shear rate']) * 60 * float(anchor['Coating Gap (micron)']) / (1e6 * float(anchor['Coating Speed (m/min)']))
    if np.isclose(factor, 1, rtol=1e-6, atol=1e-8):
        multiplier = 1.0
    elif np.isclose(factor * float(anchor['Coating ratio (%)']) / 100, 1, rtol=1e-6, atol=1e-8):
        multiplier = 100 / float(coating_ratio)
    else:
        raise ValueError('Unrecognised observed shear/gap convention')
    return float(speed) * 1e6 / (60 * float(gap)) * multiplier
