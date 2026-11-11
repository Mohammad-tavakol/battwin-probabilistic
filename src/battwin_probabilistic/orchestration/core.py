"""Original propagation algorithms, extracted without notebook execution.

Only wiring changes: explicit bundle injection and instance-bound physical specs.
Member assignments, sampling order, named-state propagation and summaries are retained.
"""
from __future__ import annotations
from typing import Dict,List,Optional,Sequence,Tuple,Any
from pathlib import Path
import re,math,warnings,hashlib
from copy import deepcopy
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import xgboost as xgb
from ..artifacts import load_base_bundle as load

APPLY_CALIBRATION_TO_PARTICLE_SAMPLES = True

CALIBRATED_CHAIN_CLIP_NONNEGATIVE = True

CHAIN_CALIBRATION_PHYSICAL_BOUNDS = {}

N_MEMBER_REPEATS = 50

N_PARTICLES_OVERRIDE = None

RANDOM_STATE = 42

SAMPLE_QUANTILE_GRID = np.round(np.linspace(0.001, 0.999, 199), 5)

N_QUANTILE_SAMPLES = 1000

PFI_N_REPEATS = 1

PFI_MAX_SCENARIOS = None

CHAIN_PFI_OUTPUTS = 'all'

FINAL_KPI_COLUMNS = ['Gravimetric capacity (mA h/g)', 'Volumetric capacity (mA h/cm3)']

SPEC_LIMITS = {}

USE_PHYSICAL_POROSITY_OVERRIDE = True

POROSITY_MASS_LOADING_SCALE = 10.0

POROSITY_CLIP_BOUNDS = None

POROSITY_MODEL_SUFFIX = '__model_prediction'

PHYSICAL_POROSITY_SPECS = {'Drying': {'Dried Porosity (%)': {'mass_loading_col': 'Dried Coat mass loading (mg.cm2)', 'thickness_col': 'Dried Coat Thickness (µm)', 'bulk_density_col': 'Bulk Density(g/cm3)', 'scale': POROSITY_MASS_LOADING_SCALE, 'clip_bounds': POROSITY_CLIP_BOUNDS}}, 'Calendaring': {'Calendered Porosity (%)': {'mass_loading_col': 'Dried Coat mass loading (mg.cm2)', 'thickness_col': 'Calendered Thickness (um)', 'bulk_density_col': 'Bulk Density(g/cm3)', 'scale': POROSITY_MASS_LOADING_SCALE, 'clip_bounds': POROSITY_CLIP_BOUNDS}}}

SUMMARY_QUANTILES = [0.05, 0.5, 0.95]

def make_safe_filename(text: str, max_len: int = 100) -> str:
    text = str(text)
    text = re.sub(r"[<>:\"/\\|?*\n\r\t]", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.replace("%", "pct").replace("µ", "u").replace("μ", "u")
    text = text.strip(" ._")
    return text[:max_len]

def ensure_columns(df: pd.DataFrame, columns: Sequence[str], context: str = "dataframe") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {context}: {missing}")

def model_prediction_column(output_column: str) -> str:
    """Column name used to keep direct model prediction when a physical override is applied."""
    return f"{output_column}{POROSITY_MODEL_SUFFIX}"

def _physical_porosity_spec(stage_name: str, output_column: str) -> Optional[Dict[str, Any]]:
    if not USE_PHYSICAL_POROSITY_OVERRIDE:
        return None
    return PHYSICAL_POROSITY_SPECS.get(stage_name, {}).get(output_column)

def calculate_physical_porosity(
    particles: pd.DataFrame,
    mass_loading_col: str,
    thickness_col: str,
    bulk_density_col: str,
    scale: float = POROSITY_MASS_LOADING_SCALE,
    clip_bounds: Optional[Tuple[float, float]] = POROSITY_CLIP_BOUNDS,
) -> np.ndarray:
    """
    Calculate electrode porosity from mass loading, thickness, and bulk density.

    Implemented formula:
        porosity = (1 - (mass_loading * scale) / (thickness * bulk_density)) * 100

    Invalid denominators return NaN. Optional clipping can be used to enforce
    physical porosity bounds.
    """
    ensure_columns(
        particles,
        [mass_loading_col, thickness_col, bulk_density_col],
        context="physical porosity calculation",
    )

    mass_loading = pd.to_numeric(particles[mass_loading_col], errors="coerce").to_numpy(dtype=float)
    thickness = pd.to_numeric(particles[thickness_col], errors="coerce").to_numpy(dtype=float)
    bulk_density = pd.to_numeric(particles[bulk_density_col], errors="coerce").to_numpy(dtype=float)

    denom = thickness * bulk_density
    valid = (
        np.isfinite(mass_loading)
        & np.isfinite(thickness)
        & np.isfinite(bulk_density)
        & (denom != 0.0)
    )

    porosity = np.full(len(particles), np.nan, dtype=float)
    porosity[valid] = (1.0 - (mass_loading[valid] * float(scale)) / denom[valid]) * 100.0

    if clip_bounds is not None:
        lo, hi = clip_bounds
        porosity = np.clip(porosity, float(lo), float(hi))

    return porosity

def to_dense_if_needed(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

def finite_mask(*arrays) -> np.ndarray:
    mask = None
    for arr in arrays:
        arr = np.asarray(arr, dtype=float).ravel()
        m = np.isfinite(arr)
        mask = m if mask is None else (mask & m)
    return mask if mask is not None else np.array([], dtype=bool)

def reference_comparison_metrics(reference, permuted, eps=1e-12) -> Dict[str, float]:
    """Compare a permuted prediction vector against the baseline/reference prediction.

    This is used when no ground-truth target exists. The baseline straight-chain
    or straight-stage prediction is treated as the reference response. Importance
    is reported as R2Drop = 1 - R2(reference, permuted), plus absolute shifts.
    """
    reference = np.asarray(reference, dtype=float).ravel()
    permuted = np.asarray(permuted, dtype=float).ravel()
    mask = finite_mask(reference, permuted)
    a = reference[mask]
    b = permuted[mask]
    if len(a) == 0:
        return {
            "N_valid_reference": 0,
            "R2_against_baseline": np.nan,
            "R2Drop": np.nan,
            "MeanAbsShift": np.nan,
            "RMShift": np.nan,
            "MeanSignedShift": np.nan,
            "MeanAbsRelativeShift_percent": np.nan,
        }
    r2 = r2_score(a, b) if len(a) >= 2 else np.nan
    d = b - a
    return {
        "N_valid_reference": int(len(a)),
        "R2_against_baseline": float(r2) if np.isfinite(r2) else np.nan,
        "R2Drop": float(1.0 - r2) if np.isfinite(r2) else np.nan,
        "MeanAbsShift": float(np.mean(np.abs(d))),
        "RMShift": float(np.sqrt(np.mean(d ** 2))),
        "MeanSignedShift": float(np.mean(d)),
        "MeanAbsRelativeShift_percent": float(100.0 * np.mean(np.abs(d) / np.maximum(np.abs(a), eps))),
    }

def extract_quantile_column(q_df: pd.DataFrame, q: float) -> np.ndarray:
    cols = list(q_df.columns)
    candidates = [
        q, str(q), f"{q:.1f}", f"{q:.2f}", f"{q:.3f}", f"{q:.4f}", f"{q:.5f}",
        f"quant_{q}", f"quant_{q:.2f}", f"quant_{q:.3f}", f"quant_{q:.4f}", f"quant_{q:.5f}",
        f"quantile_{q}", f"quantile_{q:.2f}", f"quantile_{q:.3f}", f"quantile_{q:.4f}", f"quantile_{q:.5f}",
        f"q{int(round(q * 100)):02d}",
    ]
    for name in candidates:
        if name in cols:
            return np.asarray(q_df[name], dtype=float)
    for col in cols:
        try:
            if np.isclose(float(col), float(q)):
                return np.asarray(q_df[col], dtype=float)
        except Exception:
            pass
    # Caller may use positional fallback.
    raise ValueError(f"Could not identify quantile column {q}. Columns: {cols}")

def inverse_transform_target(y_model: np.ndarray, target_mode: str) -> np.ndarray:
    y_model = np.asarray(y_model, dtype=float)
    if target_mode == "raw":
        out = y_model
    elif target_mode == "log1p":
        out = np.expm1(y_model)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    out = np.where(np.isfinite(out), out, np.nan)
    return np.clip(out, 0, None)

def predict_model_quantile_grid(
    model,
    preprocessor,
    X: pd.DataFrame,
    target_mode: str,
    quantile_grid: Sequence[float],
    n_quantile_samples: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return array with shape (n_quantiles, n_rows) on original target scale."""
    if xgb is None:
        raise ImportError(f"xgboost could not be imported: {XGBOOST_IMPORT_ERROR}")

    Xp = to_dense_if_needed(preprocessor.transform(X))
    dmat = xgb.DMatrix(Xp)
    quantile_grid = list(map(float, quantile_grid))

    try:
        q_raw = model.predict(
            dmat,
            pred_type="quantiles",
            n_samples=int(n_quantile_samples),
            quantiles=quantile_grid,
            seed=int(seed) if seed is not None else None,
        )
    except TypeError:
        q_raw = model.predict(
            dmat,
            pred_type="quantiles",
            n_samples=int(n_quantile_samples),
            quantiles=quantile_grid,
        )

    n_rows = len(X)
    out = np.full((len(quantile_grid), n_rows), np.nan, dtype=float)

    if isinstance(q_raw, pd.DataFrame):
        for i, q in enumerate(quantile_grid):
            try:
                vals = extract_quantile_column(q_raw, q)
            except ValueError:
                vals = np.asarray(q_raw.iloc[:, i], dtype=float)
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if len(vals) != n_rows:
                raise ValueError(f"Quantile output length {len(vals)} != expected rows {n_rows}.")
            out[i, :] = vals
    else:
        arr = np.asarray(q_raw, dtype=float)
        if arr.ndim == 1:
            if len(quantile_grid) != 1:
                raise ValueError(f"Unexpected 1D quantile output with {len(quantile_grid)} requested quantiles.")
            out[0, :] = arr.reshape(-1)
        elif arr.ndim == 2:
            if arr.shape == (n_rows, len(quantile_grid)):
                out = arr.T
            elif arr.shape == (len(quantile_grid), n_rows):
                out = arr
            else:
                raise ValueError(
                    f"Unexpected quantile output shape {arr.shape}; expected "
                    f"({n_rows}, {len(quantile_grid)}) or ({len(quantile_grid)}, {n_rows})."
                )
        else:
            raise ValueError(f"Unexpected quantile output dimension: {arr.ndim}")

    return inverse_transform_target(out, target_mode)

def interpolate_inverse_cdf(q_grid_values: np.ndarray, q_grid: Sequence[float], u: np.ndarray) -> np.ndarray:
    """Interpolate row-wise inverse CDF values from predicted quantile grid."""
    q_grid = np.asarray(q_grid, dtype=float)
    u = np.asarray(u, dtype=float).ravel()
    q_grid_values = np.asarray(q_grid_values, dtype=float)
    n_rows = q_grid_values.shape[1]
    if len(u) != n_rows:
        raise ValueError(f"u length {len(u)} != n_rows {n_rows}")

    samples = np.full(n_rows, np.nan, dtype=float)
    for i in range(n_rows):
        vals = q_grid_values[:, i]
        mask = np.isfinite(vals)
        if np.sum(mask) < 2:
            continue
        samples[i] = np.interp(u[i], q_grid[mask], vals[mask])
    return samples

def predict_target_ensemble_summary(
    target_ensemble: Dict[str, Any],
    X: pd.DataFrame,
    quantile_grid: Sequence[float] = SAMPLE_QUANTILE_GRID,
    n_quantile_samples: int = N_QUANTILE_SAMPLES,
) -> Dict[str, np.ndarray]:
    """Pool member-wise quantile grids to get mean/q05/q50/q95 for a target."""
    member_grids = []
    target_mode = target_ensemble["target_mode"]
    for member in target_ensemble["members"]:
        q_grid = predict_model_quantile_grid(
            model=member["model"],
            preprocessor=member["preprocessor"],
            X=X,
            target_mode=target_mode,
            quantile_grid=quantile_grid,
            n_quantile_samples=n_quantile_samples,
            seed=member.get("seed", None),
        )
        member_grids.append(q_grid)
    member_grids = np.stack(member_grids, axis=0)  # members x quantiles x rows
    pooled = member_grids.reshape(-1, member_grids.shape[-1])
    return {
        "mean": np.nanmean(pooled, axis=0),
        "q05": np.nanquantile(pooled, 0.05, axis=0),
        "q50": np.nanquantile(pooled, 0.50, axis=0),
        "q95": np.nanquantile(pooled, 0.95, axis=0),
        "std": np.nanstd(pooled, axis=0, ddof=0),
    }

def normalise_calibration_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text

def calibration_factors_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    cal = bundle.get("calibration", {}) if isinstance(bundle, dict) else {}
    factors = cal.get("factors", {}) if isinstance(cal, dict) else {}
    return factors if isinstance(factors, dict) else {}

def get_bundle_calibration_factor(bundle: Dict[str, Any], target: str, default: float = 1.0) -> float:
    factors = calibration_factors_from_bundle(bundle)
    if not factors:
        return float(default)

    if target in factors:
        item = factors[target]
    else:
        target_norm = normalise_calibration_name(target)
        item = None
        for key, value in factors.items():
            if normalise_calibration_name(key) == target_norm:
                item = value
                break

    if item is None:
        return float(default)

    if isinstance(item, dict):
        value = item.get("scale_factor", item.get("c", default))
    else:
        value = item

    try:
        value = float(value)
    except Exception:
        value = float(default)

    if not np.isfinite(value) or value <= 0:
        return float(default)

    return float(value)

def get_bundle_calibration_bounds(bundle: Dict[str, Any], target: str) -> Tuple[Optional[float], Optional[float]]:
    # User-specified chain bounds take priority.
    if CHAIN_CALIBRATION_PHYSICAL_BOUNDS:
        if target in CHAIN_CALIBRATION_PHYSICAL_BOUNDS:
            lo, hi = CHAIN_CALIBRATION_PHYSICAL_BOUNDS[target]
            return lo, hi

        target_norm = normalise_calibration_name(target)
        for key, value in CHAIN_CALIBRATION_PHYSICAL_BOUNDS.items():
            if normalise_calibration_name(key) == target_norm:
                lo, hi = value
                return lo, hi

    # Some calibrated bundles may store bounds inside calibration metadata.
    cal = bundle.get("calibration", {}) if isinstance(bundle, dict) else {}
    bounds = cal.get("physical_bounds", None) if isinstance(cal, dict) else None
    if isinstance(bounds, dict):
        if target in bounds:
            lo, hi = bounds[target]
            return lo, hi
        target_norm = normalise_calibration_name(target)
        for key, value in bounds.items():
            if normalise_calibration_name(key) == target_norm:
                lo, hi = value
                return lo, hi

    return None, None

def apply_calibration_scale_to_samples(
    samples: np.ndarray,
    centre: np.ndarray,
    scale_factor: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    centre = np.asarray(centre, dtype=float)

    out = centre + float(scale_factor) * (samples - centre)

    if CALIBRATED_CHAIN_CLIP_NONNEGATIVE:
        out = np.maximum(out, 0.0)

    if lower_bound is not None:
        out = np.maximum(out, float(lower_bound))

    if upper_bound is not None:
        out = np.minimum(out, float(upper_bound))

    return out

def apply_calibration_scale_to_summary(
    summary: Dict[str, np.ndarray],
    scale_factor: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    if scale_factor == 1.0:
        return summary

    out = deepcopy(summary)

    if "q50" in out:
        centre = np.asarray(out["q50"], dtype=float)
    elif "median" in out:
        centre = np.asarray(out["median"], dtype=float)
    elif "mean" in out:
        centre = np.asarray(out["mean"], dtype=float)
    else:
        return out

    for key in ["q05", "q95"]:
        if key in out:
            out[key] = apply_calibration_scale_to_samples(
                samples=np.asarray(out[key], dtype=float),
                centre=centre,
                scale_factor=scale_factor,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

    for key in ["std", "aleatoric_std", "epistemic_std", "total_std"]:
        if key in out:
            out[key] = float(scale_factor) * np.asarray(out[key], dtype=float)

    return out

class ProbabilisticStage:

    def __init__(self, name: str, bundle_path: str, input_columns: Optional[Sequence[str]], output_columns: Sequence[str], required_input_columns: Optional[Sequence[str]]=None, target_alias: Optional[Dict[str, str]]=None, output_aliases: Optional[Dict[str, Sequence[str]]]=None, input_aliases: Optional[Dict[str, str]]=None, apply_bundle_calibration: bool=False, calibration_label: str='uncalibrated', bundle=None):
        self.name = name
        self.bundle_path = str(bundle_path)
        self.bundle = bundle if bundle is not None else load(self.bundle_path)
        self.apply_bundle_calibration = bool(apply_bundle_calibration)
        self.calibration_label = str(calibration_label)
        if self.apply_bundle_calibration:
            factors = calibration_factors_from_bundle(self.bundle)
            if factors:
                print(f'Stage {self.name}: applying bundle calibration factors:')
                for k, v in factors.items():
                    if isinstance(v, dict):
                        sf = v.get('scale_factor', v.get('c', np.nan))
                    else:
                        sf = v
                    print(f'  - {k}: c={sf}')
            else:
                print(f'Stage {self.name}: calibration requested, but bundle contains no calibration factors. This stage will behave as uncalibrated.')
        if input_columns is None:
            if 'feature_columns' not in self.bundle:
                raise ValueError(f'Stage {self.name}: input_columns=None but bundle has no feature_columns.')
            input_columns = list(self.bundle['feature_columns'])
        self.input_columns = list(input_columns)
        self.output_columns = list(output_columns)
        self.required_input_columns = list(required_input_columns or self.input_columns)
        self.target_alias = dict(target_alias or {})
        self.output_aliases = {k: list(v) for k, v in dict(output_aliases or {}).items()}
        self.input_aliases = dict(input_aliases or {})
        self._validate_bundle()

    def _validate_bundle(self):
        if 'target_ensembles' not in self.bundle:
            raise ValueError(f"Stage {self.name}: bundle does not contain 'target_ensembles'.")
        for out_col in self.output_columns:
            target = self.target_alias.get(out_col, out_col)
            if target not in self.bundle['target_ensembles']:
                available = list(self.bundle['target_ensembles'].keys())
                raise KeyError(f'Stage {self.name}: output column {out_col!r} maps to target {target!r}, which is not in bundle targets. Available targets: {available}')

    @property
    def n_members(self) -> int:
        target0 = self.target_alias.get(self.output_columns[0], self.output_columns[0])
        return len(self.bundle['target_ensembles'][target0]['members'])

    def get_generated_output_columns(self, include_model_comparison: bool=True) -> List[str]:
        """Columns generated by this stage and suitable for summaries/reports."""
        cols = list(self.output_columns)
        if include_model_comparison and self.use_physical_porosity:
            for out_col in self.output_columns:
                if self.physical_spec(out_col) is not None:
                    cols.append(model_prediction_column(out_col))
        return list(dict.fromkeys(cols))

    def _bundle_features_for_target(self, target: str) -> List[str]:
        return list(self.bundle['feature_columns'])

    def _calibration_factor_for_target(self, target: str) -> float:
        if not self.apply_bundle_calibration or not APPLY_CALIBRATION_TO_PARTICLE_SAMPLES:
            return 1.0
        return get_bundle_calibration_factor(self.bundle, target, default=1.0)

    def _calibration_bounds_for_target(self, target: str) -> Tuple[Optional[float], Optional[float]]:
        if not self.apply_bundle_calibration:
            return (None, None)
        return get_bundle_calibration_bounds(self.bundle, target)

    def _apply_calibration_to_sampled_target(self, target: str, sampled_values: np.ndarray, q_grid_values: np.ndarray, quantile_grid: Sequence[float]) -> np.ndarray:
        c = self._calibration_factor_for_target(target)
        if c == 1.0:
            return sampled_values
        centre = interpolate_inverse_cdf(q_grid_values=q_grid_values, q_grid=quantile_grid, u=np.full(len(sampled_values), 0.5, dtype=float))
        lower_bound, upper_bound = self._calibration_bounds_for_target(target)
        return apply_calibration_scale_to_samples(samples=sampled_values, centre=centre, scale_factor=c, lower_bound=lower_bound, upper_bound=upper_bound)

    def _apply_calibration_to_summary_target(self, target: str, pred_summary: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        c = self._calibration_factor_for_target(target)
        if c == 1.0:
            return pred_summary
        lower_bound, upper_bound = self._calibration_bounds_for_target(target)
        return apply_calibration_scale_to_summary(summary=pred_summary, scale_factor=c, lower_bound=lower_bound, upper_bound=upper_bound)

    def sample_target(self, target: str, particles: pd.DataFrame, member_ids: np.ndarray, rng: np.random.Generator, quantile_grid: Sequence[float]=SAMPLE_QUANTILE_GRID, n_quantile_samples: int=N_QUANTILE_SAMPLES) -> np.ndarray:
        target_ensemble = self.bundle['target_ensembles'][target]
        members = target_ensemble['members']
        n_members = len(members)
        n = len(particles)
        member_ids = np.asarray(member_ids, dtype=int).ravel() % n_members
        if len(member_ids) != n:
            raise ValueError('member_ids length must match number of particles.')
        feature_cols = self._bundle_features_for_target(target)
        ensure_columns(particles, feature_cols, context=f'particles for {self.name}/{target}')
        u_all = rng.uniform(low=float(np.min(quantile_grid)), high=float(np.max(quantile_grid)), size=n)
        sampled = np.full(n, np.nan, dtype=float)
        for b in np.unique(member_ids):
            mask = member_ids == b
            if not np.any(mask):
                continue
            member = members[int(b)]
            X_b = particles.loc[mask, feature_cols]
            q_grid_values = predict_model_quantile_grid(model=member['model'], preprocessor=member['preprocessor'], X=X_b, target_mode=target_ensemble['target_mode'], quantile_grid=quantile_grid, n_quantile_samples=n_quantile_samples, seed=member.get('seed', None))
            sampled_b = interpolate_inverse_cdf(q_grid_values, quantile_grid, u_all[mask])
            sampled_b = self._apply_calibration_to_sampled_target(target=target, sampled_values=sampled_b, q_grid_values=q_grid_values, quantile_grid=quantile_grid)
            sampled[mask] = sampled_b
        return sampled

    def sample_outputs(self, particles: pd.DataFrame, member_ids: np.ndarray, rng: np.random.Generator, quantile_grid: Sequence[float]=SAMPLE_QUANTILE_GRID, n_quantile_samples: int=N_QUANTILE_SAMPLES) -> pd.DataFrame:
        particles = particles.copy()
        for required_name, source_name in self.input_aliases.items():
            if required_name not in particles.columns and source_name in particles.columns:
                particles[required_name] = particles[source_name]
        ensure_columns(particles, self.input_columns, context=f'particles entering stage {self.name}')
        for out_col in self.output_columns:
            target = self.target_alias.get(out_col, out_col)
            sampled_values = self.sample_target(target=target, particles=particles, member_ids=member_ids, rng=rng, quantile_grid=quantile_grid, n_quantile_samples=n_quantile_samples)
            phys_spec = self.physical_spec(out_col)
            if phys_spec is not None:
                model_col = model_prediction_column(out_col)
                particles[model_col] = sampled_values
            else:
                particles[out_col] = sampled_values
                for alias in self.output_aliases.get(out_col, []):
                    particles[alias] = sampled_values
        if self.use_physical_porosity:
            for out_col in self.output_columns:
                phys_spec = self.physical_spec(out_col)
                if phys_spec is None:
                    continue
                calculated = calculate_physical_porosity(particles=particles, mass_loading_col=phys_spec['mass_loading_col'], thickness_col=phys_spec['thickness_col'], bulk_density_col=phys_spec['bulk_density_col'], scale=float(phys_spec.get('scale', POROSITY_MASS_LOADING_SCALE)), clip_bounds=phys_spec.get('clip_bounds', POROSITY_CLIP_BOUNDS))
                particles[out_col] = calculated
                for alias in self.output_aliases.get(out_col, []):
                    particles[alias] = calculated
        return particles

    def predict_outputs_summary(self, df_stage: pd.DataFrame, quantile_grid: Sequence[float]=SAMPLE_QUANTILE_GRID, n_quantile_samples: int=N_QUANTILE_SAMPLES) -> Dict[str, Dict[str, np.ndarray]]:
        ensure_columns(df_stage, self.input_columns, context=f'stage-wise PFI data for {self.name}')
        out = {}
        for out_col in self.output_columns:
            target = self.target_alias.get(out_col, out_col)
            feature_cols = self._bundle_features_for_target(target)
            ensure_columns(df_stage, feature_cols, context=f'stage-wise PFI data for {self.name}/{target}')
            pred_summary = predict_target_ensemble_summary(target_ensemble=self.bundle['target_ensembles'][target], X=df_stage[feature_cols], quantile_grid=quantile_grid, n_quantile_samples=n_quantile_samples)
            pred_summary = self._apply_calibration_to_summary_target(target, pred_summary)
            phys_spec = self.physical_spec(out_col)
            if phys_spec is not None:
                out[model_prediction_column(out_col)] = pred_summary
            else:
                out[out_col] = pred_summary
        if self.use_physical_porosity:
            for out_col in self.output_columns:
                phys_spec = self.physical_spec(out_col)
                if phys_spec is None:
                    continue
                mass_col = phys_spec['mass_loading_col']
                thick_col = phys_spec['thickness_col']
                bulk_col = phys_spec['bulk_density_col']
                if bulk_col not in df_stage.columns:
                    continue

                def _summary_or_state(column_name: str, stat: str) -> Optional[np.ndarray]:
                    if column_name in out and stat in out[column_name]:
                        return np.asarray(out[column_name][stat], dtype=float)
                    if column_name in df_stage.columns:
                        return pd.to_numeric(df_stage[column_name], errors='coerce').to_numpy(dtype=float)
                    return None
                bulk = pd.to_numeric(df_stage[bulk_col], errors='coerce').to_numpy(dtype=float)
                calc_summary = {}
                for stat in ['mean', 'q05', 'q50', 'q95']:
                    mass = _summary_or_state(mass_col, stat)
                    thick = _summary_or_state(thick_col, stat)
                    if mass is None or thick is None:
                        continue
                    tmp = pd.DataFrame({mass_col: mass, thick_col: thick, bulk_col: bulk})
                    calc_summary[stat] = calculate_physical_porosity(particles=tmp, mass_loading_col=mass_col, thickness_col=thick_col, bulk_density_col=bulk_col, scale=float(phys_spec.get('scale', POROSITY_MASS_LOADING_SCALE)), clip_bounds=phys_spec.get('clip_bounds', POROSITY_CLIP_BOUNDS))
                if calc_summary:
                    n = len(df_stage)
                    calc_summary.setdefault('std', np.full(n, np.nan, dtype=float))
                    out[out_col] = calc_summary
        return out

class ProbabilisticNamedStateSpaceLine:

    def __init__(self, stages: Sequence[ProbabilisticStage]):
        self.stages = list(stages)
        self.compiled = False

    def compile(self):
        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            raise ValueError('Stage names must be unique.')
        self.compiled = True

    def get_required_input_columns(self) -> List[str]:
        required = set()
        generated = set()
        for stage in self.stages:
            stage_required = list(stage.required_input_columns)
            for out_col in stage.output_columns:
                phys_spec = stage.physical_spec(out_col)
                if phys_spec is not None:
                    stage_required.extend([phys_spec['mass_loading_col'], phys_spec['thickness_col'], phys_spec['bulk_density_col']])
            for c in stage_required:
                if c not in generated:
                    required.add(c)
            generated.update(stage.get_generated_output_columns(include_model_comparison=True))
            for aliases in getattr(stage, 'output_aliases', {}).values():
                generated.update(aliases)
        return sorted(required)

    def validate_input_dataframe(self, df_input: pd.DataFrame):
        ensure_columns(df_input, self.get_required_input_columns(), context='input dataframe')

    def _make_member_ids_for_stage(self, n_particles_total: int, scenario_ids: np.ndarray, n_members: int, rng: np.random.Generator) -> np.ndarray:
        """Stratified member assignment per scenario."""
        member_ids = np.full(n_particles_total, -1, dtype=int)
        for sid in np.unique(scenario_ids):
            idx = np.where(scenario_ids == sid)[0]
            n = len(idx)
            reps = int(math.ceil(n / n_members))
            ids = np.tile(np.arange(n_members), reps)[:n]
            rng.shuffle(ids)
            member_ids[idx] = ids
        return member_ids

    def simulate_items_particles(self, df_input: pd.DataFrame, n_member_repeats: int=N_MEMBER_REPEATS, n_particles_override: Optional[int]=N_PARTICLES_OVERRIDE, random_state: int=RANDOM_STATE, scenario_id_column: Optional[str]=None, final_kpi_columns: Optional[Sequence[str]]=None, spec_limits: Optional[Dict[str, Tuple[float, float]]]=None, keep_particles: bool=True, quantile_grid: Sequence[float]=SAMPLE_QUANTILE_GRID, n_quantile_samples: int=N_QUANTILE_SAMPLES) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if not self.compiled:
            raise RuntimeError('Probabilistic line must be compiled before simulation.')
        self.validate_input_dataframe(df_input)
        first_B = self.stages[0].n_members
        n_particles = int(n_particles_override) if n_particles_override is not None else int(first_B * n_member_repeats)
        if n_particles < 1:
            raise ValueError('n_particles must be positive.')
        rng = np.random.default_rng(random_state)
        df0 = df_input.reset_index(drop=True).copy()
        n_scenarios = len(df0)
        scenario_ids = np.repeat(np.arange(n_scenarios), n_particles)
        particle_ids = np.tile(np.arange(n_particles), n_scenarios)
        particles = df0.loc[df0.index.repeat(n_particles)].reset_index(drop=True).copy()
        particles['scenario_index'] = scenario_ids
        particles['particle_id'] = particle_ids
        if scenario_id_column and scenario_id_column in df0.columns:
            particles['scenario_id'] = np.repeat(df0[scenario_id_column].to_numpy(), n_particles)
        else:
            particles['scenario_id'] = scenario_ids
        stage_member_records = {}
        for stage in self.stages:
            member_ids = self._make_member_ids_for_stage(n_particles_total=len(particles), scenario_ids=particles['scenario_index'].to_numpy(), n_members=stage.n_members, rng=rng)
            particles[f'{stage.name}__member_id'] = member_ids
            stage_member_records[stage.name] = member_ids
            particles = stage.sample_outputs(particles=particles, member_ids=member_ids, rng=rng, quantile_grid=quantile_grid, n_quantile_samples=n_quantile_samples)
        summary = summarise_particles(particles=particles, input_df=df0, final_kpi_columns=final_kpi_columns or FINAL_KPI_COLUMNS, generated_columns=[c for stage in self.stages for c in stage.get_generated_output_columns(include_model_comparison=True)], spec_limits=spec_limits or SPEC_LIMITS, scenario_id_column=scenario_id_column)
        return (summary, particles if keep_particles else None)

def summarise_particles(
    particles: pd.DataFrame,
    input_df: pd.DataFrame,
    generated_columns: Sequence[str],
    final_kpi_columns: Sequence[str],
    spec_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    scenario_id_column: Optional[str] = None,
) -> pd.DataFrame:
    spec_limits = spec_limits or {}
    groups = particles.groupby("scenario_index", sort=True)
    rows = []

    for sid, g in groups:
        row = {}
        # Keep original input columns once.
        for c in input_df.columns:
            row[c] = input_df.loc[int(sid), c]
        row["scenario_index"] = int(sid)
        if scenario_id_column and scenario_id_column in input_df.columns:
            row["scenario_id"] = input_df.loc[int(sid), scenario_id_column]
        else:
            row["scenario_id"] = int(sid)
        row["n_particles"] = int(len(g))

        for col in generated_columns:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            prefix = make_safe_filename(col, max_len=80)
            if len(vals) == 0:
                row[f"{prefix}__mean"] = np.nan
                row[f"{prefix}__std"] = np.nan
                row[f"{prefix}__q05"] = np.nan
                row[f"{prefix}__q50"] = np.nan
                row[f"{prefix}__q95"] = np.nan
            else:
                row[f"{prefix}__mean"] = float(np.mean(vals))
                row[f"{prefix}__std"] = float(np.std(vals, ddof=0))
                for q in SUMMARY_QUANTILES:
                    row[f"{prefix}__q{int(round(q * 100)):02d}"] = float(np.quantile(vals, q))

        # Feasibility probabilities.
        if spec_limits:
            all_inside = np.ones(len(g), dtype=bool)
            for kpi, (lo, hi) in spec_limits.items():
                if kpi not in g.columns:
                    continue
                vals = pd.to_numeric(g[kpi], errors="coerce").to_numpy(dtype=float)
                inside = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
                row[f"P_{make_safe_filename(kpi)}__in_spec"] = float(np.mean(inside))
                all_inside &= inside
            row["P_all_KPIs_in_spec"] = float(np.mean(all_inside))

        rows.append(row)

    return pd.DataFrame(rows)

def select_pfi_features(df_input: pd.DataFrame, line: ProbabilisticNamedStateSpaceLine, pfi_features=None) -> List[str]:
    if pfi_features is not None:
        return [c for c in pfi_features if c in df_input.columns]
    generated = {c for stage in line.stages for c in stage.get_generated_output_columns(include_model_comparison=True)}
    required = line.get_required_input_columns()
    candidates = [c for c in required if c in df_input.columns and c not in generated]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df_input[c])]

def permutation_apply(df: pd.DataFrame, columns: Sequence[str], rng: np.random.Generator) -> pd.DataFrame:
    dfp = df.copy()
    columns = [c for c in columns if c in dfp.columns]
    if not columns:
        return dfp
    order = rng.permutation(len(dfp))
    dfp.loc[:, columns] = dfp.iloc[order][columns].to_numpy()
    return dfp

def default_chain_metric_columns(
    summary: pd.DataFrame,
    output_columns: Sequence[str],
    spec_limits: Dict[str, Tuple[float, float]],
) -> List[str]:
    """
    Return summary columns used by chain-level PFI for any generated output.

    output_columns may contain intermediate IPIs or final KPIs. The function
    does not assume that the requested outputs belong to the final stage.
    """
    cols = []

    for output_name in output_columns:
        safe = make_safe_filename(output_name, max_len=80)
        for suffix in ["mean", "std", "q05", "q50", "q95"]:
            col = f"{safe}__{suffix}"
            if col in summary.columns:
                cols.append(col)

    # Include feasibility metrics when specification limits are configured.
    for col in summary.columns:
        if col.startswith("P_") and col.endswith("__in_spec"):
            cols.append(col)

    if "P_all_KPIs_in_spec" in summary.columns:
        cols.append("P_all_KPIs_in_spec")

    return list(dict.fromkeys(cols))

def compare_summary_shift(base_summary: pd.DataFrame, perm_summary: pd.DataFrame, metric_columns: Sequence[str]) -> Dict[str, float]:
    out = {}
    for col in metric_columns:
        if col not in base_summary.columns or col not in perm_summary.columns:
            continue
        a = pd.to_numeric(base_summary[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(perm_summary[col], errors="coerce").to_numpy(dtype=float)
        comp = reference_comparison_metrics(a, b)
        out[f"N_valid_reference__{col}"] = comp["N_valid_reference"]
        out[f"R2AgainstBaseline__{col}"] = comp["R2_against_baseline"]
        out[f"R2Drop__{col}"] = comp["R2Drop"]
        out[f"MeanAbsShift__{col}"] = comp["MeanAbsShift"]
        out[f"RMShift__{col}"] = comp["RMShift"]
        out[f"MeanSignedShift__{col}"] = comp["MeanSignedShift"]
        out[f"MeanAbsRelativeShiftPercent__{col}"] = comp["MeanAbsRelativeShift_percent"]
    return out

def compute_chain_pfi(
    line: ProbabilisticNamedStateSpaceLine,
    df_input: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    feature_groups: Optional[Dict[str, Sequence[str]]] = None,
    n_repeats: int = PFI_N_REPEATS,
    n_member_repeats: int = N_MEMBER_REPEATS,
    n_particles_override: Optional[int] = N_PARTICLES_OVERRIDE,
    random_state: int = RANDOM_STATE,
    final_kpi_columns: Sequence[str] = FINAL_KPI_COLUMNS,
    pfi_output_columns: Any = CHAIN_PFI_OUTPUTS,
    spec_limits: Dict[str, Tuple[float, float]] = SPEC_LIMITS,
    max_scenarios: Optional[int] = PFI_MAX_SCENARIOS,
    metric_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df_work = df_input.reset_index(drop=True).copy()
    if max_scenarios is not None and len(df_work) > max_scenarios:
        rng_sub = np.random.default_rng(random_state)
        idx = rng_sub.choice(np.arange(len(df_work)), size=int(max_scenarios), replace=False)
        df_work = df_work.iloc[np.sort(idx)].reset_index(drop=True)

    features = select_pfi_features(df_work, line, features)

    resolved_pfi_pairs = _resolve_distribution_outputs(
        stages=line.stages,
        requested=pfi_output_columns,
        final_kpi_columns=final_kpi_columns,
    )
    resolved_pfi_outputs = list(dict.fromkeys(output for _, output in resolved_pfi_pairs))

    if not resolved_pfi_outputs:
        available = [
            output
            for stage in line.stages
            for output in stage.output_columns
        ]
        raise ValueError(
            "No valid chain-PFI outputs were resolved. "
            f"Requested: {pfi_output_columns!r}. "
            f"Available generated outputs: {available}"
        )

    print("Chain-level PFI response outputs:")
    for output in resolved_pfi_outputs:
        print(f"  - {output}")

    tasks = [("feature", f, [f]) for f in features]
    if feature_groups:
        for name, cols in feature_groups.items():
            cols = [c for c in cols if c in df_work.columns]
            if cols:
                tasks.append(("group", name, cols))

    rows = []
    print(f"Chain-level PFI tasks: {len(tasks)}")
    for r in range(n_repeats):
        sim_seed = random_state + 50021 * (r + 1)
        print(f"  PFI repeat {r + 1}/{n_repeats}: baseline simulation")
        base_summary, _ = line.simulate_items_particles(
            df_input=df_work,
            n_member_repeats=n_member_repeats,
            n_particles_override=n_particles_override,
            random_state=sim_seed,
            final_kpi_columns=final_kpi_columns,
            spec_limits=spec_limits,
            keep_particles=False,
        )
        metrics = (
            list(metric_columns)
            if metric_columns is not None
            else default_chain_metric_columns(
                base_summary,
                resolved_pfi_outputs,
                spec_limits,
            )
        )

        if not metrics:
            expected = [
                f"{make_safe_filename(name, max_len=80)}__<mean/std/q05/q50/q95>"
                for name in resolved_pfi_outputs
            ]
            raise ValueError(
                "No summary metric columns were found for the requested PFI outputs. "
                f"Resolved outputs: {resolved_pfi_outputs}. "
                f"Expected summary-column patterns: {expected}. "
                f"Available summary columns: {list(base_summary.columns)}"
            )

        for task_type, name, cols in tasks:
            stable_key = f"{task_type}::{name}".encode("utf-8")
            stable_offset = int(hashlib.md5(stable_key).hexdigest()[:8], 16) % 100000
            perm_seed = random_state + 9109 * (r + 1) + stable_offset
            rng_perm = np.random.default_rng(perm_seed)
            df_perm = permutation_apply(df_work, cols, rng_perm)
            # Common random numbers: same simulation seed as baseline.
            perm_summary, _ = line.simulate_items_particles(
                df_input=df_perm,
                n_member_repeats=n_member_repeats,
                n_particles_override=n_particles_override,
                random_state=sim_seed,
                final_kpi_columns=final_kpi_columns,
                spec_limits=spec_limits,
                keep_particles=False,
            )
            shift = compare_summary_shift(base_summary, perm_summary, metrics)
            row = {
                "Repeat": r + 1,
                "Task_type": task_type,
                "Name": name,
                "Columns": " | ".join(cols),
                "N_scenarios": len(df_work),
                "N_member_repeats": n_member_repeats,
                "N_particles_override": n_particles_override if n_particles_override is not None else np.nan,
            }
            row.update(shift)
            rows.append(row)

    raw = pd.DataFrame(rows)
    return raw

def summarise_chain_pfi(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    metric_cols = [
        c for c in raw.columns
        if c.startswith((
            "R2AgainstBaseline__",
            "R2Drop__",
            "MeanAbsShift__",
            "RMShift__",
            "MeanSignedShift__",
            "MeanAbsRelativeShiftPercent__",
        ))
    ]

    if not metric_cols:
        raise ValueError(
            "The chain-PFI table contains no response metrics. "
            "This normally means that none of the requested PFI output names "
            "matched generated chain outputs. "
            f"Raw PFI columns: {list(raw.columns)}"
        )

    summary = (
        raw.groupby(["Task_type", "Name", "Columns"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if x])
        for col in summary.columns.to_flat_index()
    ]
    return summary

def get_stage_output_metadata(stages: Sequence[ProbabilisticStage], final_kpi_columns: Optional[Sequence[str]]=None) -> pd.DataFrame:
    final_kpi_set = set(final_kpi_columns or [])
    rows = []
    for stage in stages:
        for out_col in stage.output_columns:
            target = stage.target_alias.get(out_col, out_col)
            aliases = list(stage.output_aliases.get(out_col, []))
            phys_spec = stage.physical_spec(out_col)
            role = 'KPI' if out_col in final_kpi_set else 'IPI'
            source_type = 'physical_calculation' if phys_spec is not None else 'model_prediction'
            rows.append({'Stage': stage.name, 'Chain_output_column': out_col, 'Model_target_name': target, 'Output_role': role, 'Prediction_source_type': source_type, 'Safe_name': make_safe_filename(out_col, max_len=80), 'Aliases_written_to_state': '; '.join(map(str, aliases)), 'Bundle_path': stage.bundle_path})
            if phys_spec is not None:
                model_col = model_prediction_column(out_col)
                rows.append({'Stage': stage.name, 'Chain_output_column': model_col, 'Model_target_name': target, 'Output_role': f'{role}_direct_model_comparison', 'Prediction_source_type': 'direct_model_prediction_before_physical_override', 'Safe_name': make_safe_filename(model_col, max_len=80), 'Aliases_written_to_state': '', 'Bundle_path': stage.bundle_path})
    return pd.DataFrame(rows)

def _normalise_output_reference(value: Any) -> str:
    """Normalise a user-supplied output name for robust matching."""
    text = str(value).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text

def _resolve_distribution_outputs(
    stages: Sequence[ProbabilisticStage],
    requested: Any,
    final_kpi_columns: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Resolve requested output references to canonical chain-output columns.

    Matching supports:
      - exact chain-output names;
      - case/spacing/unit-symbol-insensitive names;
      - bundle target names;
      - output aliases written into the particle state.
    """
    stage_meta = get_stage_output_metadata(
        stages,
        final_kpi_columns=final_kpi_columns,
    )

    records = []
    for stage in stages:
        for out_col in stage.get_generated_output_columns(include_model_comparison=True):
            base_out_col = out_col.replace(POROSITY_MODEL_SUFFIX, "")
            target = stage.target_alias.get(base_out_col, base_out_col)
            aliases = list(stage.output_aliases.get(base_out_col, []))
            records.append({
                "Stage": stage.name,
                "Output": out_col,
                "Target": target,
                "Aliases": aliases,
            })

    if requested is None:
        requested_names = [r["Output"] for r in records]
    elif isinstance(requested, str):
        mode = requested.strip().lower()
        if mode == "all":
            requested_names = [r["Output"] for r in records]
        elif mode == "final_kpis":
            requested_names = list(final_kpi_columns or [])
        else:
            requested_names = [requested]
    else:
        requested_names = list(requested)

    resolved = []
    unresolved = []

    for requested_name in requested_names:
        requested_norm = _normalise_output_reference(requested_name)
        matches = []

        for record in records:
            candidate_names = [
                record["Output"],
                record["Target"],
                *record["Aliases"],
            ]
            candidate_norms = {
                _normalise_output_reference(name)
                for name in candidate_names
            }

            if (
                requested_name == record["Output"]
                or requested_norm in candidate_norms
            ):
                matches.append((record["Stage"], record["Output"]))

        if not matches:
            unresolved.append(str(requested_name))
        else:
            for match in matches:
                if match not in resolved:
                    resolved.append(match)

    if unresolved:
        available = [r["Output"] for r in records]
        warnings.warn(
            "The following requested outputs were not recognised and will be "
            f"skipped: {unresolved}. Available canonical output names: {available}"
        )

    return resolved

