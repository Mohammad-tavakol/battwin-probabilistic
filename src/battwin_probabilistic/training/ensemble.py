"""Preprocessing, grouped bootstrap and XGBoostLSS ensemble functions."""
from copy import deepcopy
import copy
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.distributions.StudentT import StudentT
from xgboostlss.distributions.Gamma import Gamma
from xgboostlss.distributions.LogNormal import LogNormal
from xgboostlss.distributions.Weibull import Weibull
from xgboostlss.distributions.Laplace import Laplace
from xgboostlss.distributions.Cauchy import Cauchy
from xgboostlss.distributions.Gumbel import Gumbel
DISTRIBUTION_MAP={'Gaussian':Gaussian,'StudentT':StudentT,'Gamma':Gamma,'LogNormal':LogNormal,'Weibull':Weibull,'Laplace':Laplace,'Cauchy':Cauchy,'Gumbel':Gumbel}
XGBLSS_AVAILABLE=True
XGBLSS_IMPORT_ERROR=None
CLIP_CALIBRATED_TO_NONNEGATIVE=True
CALIBRATION_C_GRID=np.r_[np.arange(1,3001)/1000.,np.arange(601,2001)/200.]
def ensure_list(x):
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)

def safe_2d(y):
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def transform_y(y_raw, target_mode):
    y_raw = safe_2d(y_raw)
    if target_mode == "raw":
        return y_raw
    if target_mode == "log1p":
        return np.log1p(y_raw)
    raise ValueError("target_mode must be 'raw' or 'log1p'.")

def build_preprocessor(X, categorical_columns):
    categorical_columns = [c for c in ensure_list(categorical_columns) or [] if c in X.columns]
    numerical_columns = [c for c in X.columns if c not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    return preprocessor, numerical_columns, categorical_columns

def to_dense_if_needed(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

def bootstrap_indices(n, group_keys=None, mode="row", seed=42):
    rng = np.random.default_rng(seed)

    if mode == "none":
        return np.arange(n)

    if mode == "row" or group_keys is None:
        return rng.choice(np.arange(n), size=n, replace=True)

    if mode == "group":
        group_keys = pd.Series(group_keys).reset_index(drop=True)
        unique_groups = group_keys.unique()
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)

        sampled_indices = []
        for g in sampled_groups:
            idx = np.where(group_keys.to_numpy() == g)[0]
            sampled_indices.extend(idx.tolist())

        return np.asarray(sampled_indices, dtype=int)

    raise ValueError("BOOTSTRAP_MODE must be 'row', 'group', or 'none'.")

def normalise_xgb_params(params, seed):
    params = copy.deepcopy(params)
    if "num_boost_round" in params:
        params.pop("num_boost_round")

    params.setdefault("booster", "gbtree")
    params.setdefault("n_jobs", -1)
    params.setdefault("verbosity", 0)
    params["seed"] = int(seed)
    return params

def get_num_boost_round(config):
    if "num_boost_round" in config and config["num_boost_round"] is not None:
        return int(config["num_boost_round"])

    params = config.get("xgb_params", {})
    if "num_boost_round" in params:
        return int(params["num_boost_round"])

    raise ValueError("Each target config must contain 'num_boost_round' or xgb_params['num_boost_round'].")

def extract_quantile_column(q_df, q):
    cols = list(q_df.columns)
    possible_names = [
        q,
        str(q),
        f"{q:.2f}",
        f"{q:.3f}",
        f"quant_{q}",
        f"quant_{q:.2f}",
        f"quant_{q:.3f}",
        f"quantile_{q}",
        f"quantile_{q:.2f}",
        f"quantile_{q:.3f}",
        f"q{q}",
        f"q{int(round(q * 100)):02d}",
    ]

    for name in possible_names:
        if name in cols:
            return np.asarray(q_df[name], dtype=float)

    for col in cols:
        try:
            if np.isclose(float(col), float(q)):
                return np.asarray(q_df[col], dtype=float)
        except Exception:
            pass

    # Fallback: the caller may use positional extraction.
    raise ValueError(f"Could not find quantile {q}. Columns are: {cols}")

def predict_model_quantile_grid(
    model,
    preprocessor,
    X,
    target_mode,
    quantile_grid,
    n_quantile_samples=1000,
    seed=42,
):
    Xp = to_dense_if_needed(preprocessor.transform(X))
    dmat = xgb.DMatrix(Xp)

    try:
        q_raw = model.predict(
            dmat,
            pred_type="quantiles",
            n_samples=n_quantile_samples,
            quantiles=list(quantile_grid),
            seed=int(seed),
        )
    except TypeError:
        q_raw = model.predict(
            dmat,
            pred_type="quantiles",
            n_samples=n_quantile_samples,
            quantiles=list(quantile_grid),
        )

    n_obs = len(X)
    quantile_grid = list(quantile_grid)
    out = np.zeros((len(quantile_grid), n_obs), dtype=float)

    if isinstance(q_raw, pd.DataFrame):
        for i, q in enumerate(quantile_grid):
            try:
                vals = extract_quantile_column(q_raw, q)
            except ValueError:
                vals = np.asarray(q_raw.iloc[:, i], dtype=float)
            out[i, :] = vals.reshape(-1)
    else:
        arr = np.asarray(q_raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Most common: n_obs x n_quantiles.
        if arr.shape[0] == n_obs and arr.shape[1] == len(quantile_grid):
            out = arr.T
        # Alternative: n_quantiles x n_obs.
        elif arr.shape[0] == len(quantile_grid) and arr.shape[1] == n_obs:
            out = arr
        else:
            raise ValueError(
                f"Unexpected quantile output shape {arr.shape}; "
                f"expected ({n_obs}, {len(quantile_grid)}) or ({len(quantile_grid)}, {n_obs})."
            )

    if target_mode == "log1p":
        out = np.expm1(out)

    out = np.clip(out, 0, None)
    return out

def make_lss_model(distribution_name, dist_kwargs):
    if not XGBLSS_AVAILABLE:
        raise ImportError(f"XGBoostLSS is not available. Import error was: {XGBLSS_IMPORT_ERROR}")

    if distribution_name not in DISTRIBUTION_MAP:
        raise ValueError(f"Unknown distribution {distribution_name!r}. Available: {list(DISTRIBUTION_MAP)}")

    DistClass = DISTRIBUTION_MAP[distribution_name]
    return XGBoostLSS(DistClass(**dist_kwargs))

def train_target_ensemble(
    split,
    target,
    config,
    n_members,
    bootstrap_mode="group",
    random_state=42,
):
    distribution_name = config["distribution"]
    target_mode = config["target_mode"]
    dist_kwargs = config.get(
        "dist_kwargs",
        {"stabilization": "MAD", "response_fn": "softplus", "loss_fn": "nll"},
    )

    xgb_params_base = config.get("xgb_params", None)
    if xgb_params_base is None:
        raise ValueError(f"Missing xgb_params for target {target!r}.")

    num_boost_round = get_num_boost_round(config)

    X_trainval = split["x_trainval"]
    y_trainval_raw = split["y_trainval"][target].to_numpy(dtype=float)
    y_trainval_model = transform_y(y_trainval_raw.reshape(-1, 1), target_mode).ravel()
    group_keys = split["trainval_group_keys"]

    members = []

    for m in range(n_members):
        seed = int(random_state + 1009 * (m + 1))
        idx = bootstrap_indices(
            n=len(X_trainval),
            group_keys=group_keys,
            mode=bootstrap_mode,
            seed=seed,
        )

        X_boot = X_trainval.iloc[idx].reset_index(drop=True)
        y_boot = y_trainval_model[idx]

        preprocessor, numerical_columns, categorical_columns = build_preprocessor(
            X_boot,
            split["categorical_columns"],
        )
        X_boot_p = to_dense_if_needed(preprocessor.fit_transform(X_boot))
        dtrain = xgb.DMatrix(X_boot_p, label=y_boot)

        xgb_params = normalise_xgb_params(xgb_params_base, seed=seed)
        model = make_lss_model(distribution_name, dist_kwargs)

        print(
            f"  Target={target} | member {m + 1}/{n_members} | "
            f"dist={distribution_name} | mode={target_mode} | seed={seed} | n_boot={len(idx)}"
        )

        model.train(
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
        )

        members.append(
            {
                "model": model,
                "preprocessor": preprocessor,
                "seed": seed,
                "xgb_params": xgb_params,
                "num_boost_round": num_boost_round,
                "numerical_columns": numerical_columns,
                "categorical_columns": categorical_columns,
            }
        )

    return {
        "target": target,
        "distribution": distribution_name,
        "target_mode": target_mode,
        "dist_kwargs": dist_kwargs,
        "members": members,
    }

def predict_target_ensemble(
    target_ensemble,
    X,
    y_true,
    quantile_grid,
    n_quantile_samples=1000,
):
    member_grids = []
    member_means = []
    member_vars = []

    target_mode = target_ensemble["target_mode"]

    for member in target_ensemble["members"]:
        q_grid_values = predict_model_quantile_grid(
            model=member["model"],
            preprocessor=member["preprocessor"],
            X=X,
            target_mode=target_mode,
            quantile_grid=quantile_grid,
            n_quantile_samples=n_quantile_samples,
            seed=member["seed"],
        )

        # Shape: n_quantiles x n_observations
        member_grids.append(q_grid_values)
        member_means.append(np.mean(q_grid_values, axis=0))
        member_vars.append(np.var(q_grid_values, axis=0, ddof=0))

    # Shape: n_members x n_quantiles x n_observations
    member_grids = np.stack(member_grids, axis=0)

    # Pooled predictive distribution approximation:
    # all member quantile grids pooled together.
    pooled = member_grids.reshape(-1, member_grids.shape[-1])

    q05 = np.quantile(pooled, 0.05, axis=0)
    q50 = np.quantile(pooled, 0.50, axis=0)
    q95 = np.quantile(pooled, 0.95, axis=0)
    mean = np.mean(pooled, axis=0)
    total_std = np.std(pooled, axis=0, ddof=0)

    member_means = np.stack(member_means, axis=0)
    member_vars = np.stack(member_vars, axis=0)

    aleatoric_var = np.mean(member_vars, axis=0)
    epistemic_var = np.var(member_means, axis=0, ddof=0)

    return {
        "y_true": np.asarray(y_true, dtype=float).ravel(),
        "mean": mean,
        "median": q50,
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "aleatoric_std": np.sqrt(np.maximum(aleatoric_var, 0)),
        "epistemic_std": np.sqrt(np.maximum(epistemic_var, 0)),
        "total_std": total_std,
        "member_means": member_means,
    }

def apply_interval_scale_to_target(pred_target, c, lower_bound=None, upper_bound=None):
    out = copy.deepcopy(pred_target)

    q05 = np.asarray(out["q05"], dtype=float)
    q50 = np.asarray(out["q50"], dtype=float)
    q95 = np.asarray(out["q95"], dtype=float)

    lo_cal = q50 - c * (q50 - q05)
    hi_cal = q50 + c * (q95 - q50)

    if CLIP_CALIBRATED_TO_NONNEGATIVE:
        lo_cal = np.maximum(lo_cal, 0.0)
        hi_cal = np.maximum(hi_cal, 0.0)

    if lower_bound is not None:
        lo_cal = np.maximum(lo_cal, float(lower_bound))
        hi_cal = np.maximum(hi_cal, float(lower_bound))

    if upper_bound is not None:
        lo_cal = np.minimum(lo_cal, float(upper_bound))
        hi_cal = np.minimum(hi_cal, float(upper_bound))

    out["q05_uncalibrated"] = q05
    out["q95_uncalibrated"] = q95
    out["q05"] = lo_cal
    out["q95"] = hi_cal

    # The centre stays unchanged.
    out["q50"] = q50
    out["median"] = q50

    # Scaling the final distribution spread. This is a post-hoc final-output
    # calibration; it should not be interpreted as a new aleatoric/epistemic decomposition.
    for key in ["aleatoric_std", "epistemic_std", "total_std"]:
        if key in out:
            out[f"{key}_uncalibrated"] = np.asarray(out[key], dtype=float)
            out[key] = c * np.asarray(out[key], dtype=float)

    return out

def picp_1d(y_true, lower, upper):
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(mask):
        return np.nan
    return float(np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])))

def calibrate_scale_for_target(y_true, q05, q50, q95, target_coverage=0.90, c_grid=None):
    if c_grid is None:
        c_grid = CALIBRATION_C_GRID

    y_true = np.asarray(y_true, dtype=float).ravel()
    q05 = np.asarray(q05, dtype=float).ravel()
    q50 = np.asarray(q50, dtype=float).ravel()
    q95 = np.asarray(q95, dtype=float).ravel()

    mask = np.isfinite(y_true) & np.isfinite(q05) & np.isfinite(q50) & np.isfinite(q95)
    y_true = y_true[mask]
    q05 = q05[mask]
    q50 = q50[mask]
    q95 = q95[mask]

    best = {
        "scale_factor": np.nan,
        "calibration_picp": np.nan,
        "coverage_error": np.inf,
    }

    for c in c_grid:
        c = float(c)
        lo = q50 - c * (q50 - q05)
        hi = q50 + c * (q95 - q50)

        if CLIP_CALIBRATED_TO_NONNEGATIVE:
            lo = np.maximum(lo, 0.0)
            hi = np.maximum(hi, 0.0)

        cov = picp_1d(y_true, lo, hi)
        err = abs(cov - target_coverage)

        if err < best["coverage_error"]:
            best = {
                "scale_factor": c,
                "calibration_picp": float(cov),
                "coverage_error": float(err),
            }

    return best
