#!/usr/bin/env python3
"""
Spectral Analyzer Pro
Chemometrics & QSAR desktop application with robust statistics.

Internal organization (single-file for portability; modular split is straightforward):
- Dependency check & imports
- Chemometric algorithms (preprocessing, collinearity, metrics, feature selection, models)
- GUI dialogs (wizard, column selector, plot viewer)
- Main application class (SpectralAnalyzerApp)
  - Session log (records preprocessing, feature selection, nested-CV changes)
  - Wavelength vector (reused in spectral overview, VIP/importance plots, contribution maps)
  - Methods text export + clipboard copy
A modular split (preprocessing.py, models.py, gui.py) can be done later
while keeping this file as the single entry point.
"""

import sys
import os

# Frozen one-file builds have no installed Python. Point Matplotlib at a
# writable folder before it is imported, or the window can fail to open.
if getattr(sys, "frozen", False):
    os.environ.setdefault("MPLBACKEND", "TkAgg")
    _mpl_dir = os.path.join(os.environ.get("TEMP") or os.environ.get("TMP") or ".", "SpectralAnalyzerPro_mpl")
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir

# ==============================================================================
# STARTUP DEPENDENCY CHECK (prevents cryptic crashes on missing packages)
# ==============================================================================
def _check_dependencies():
    missing = []
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
    except ImportError:
        print("ERROR: tkinter is not installed.")
        print("  Ubuntu/Debian: sudo apt install python3-tk")
        print("  Fedora:        sudo dnf install python3-tkinter")
        print("  macOS / Windows: included with official Python installers")
        sys.exit(1)

    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("openpyxl", "openpyxl"),
        ("PIL", "pillow"),
        ("joblib", "joblib"),
        ("threadpoolctl", "threadpoolctl"),
    ]
    for mod, pkg in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        msg = "Missing required packages:\n\n" + "\n".join(f"• {p}" for p in missing)
        msg += "\n\nInstall with:\npip install -r requirements.txt"
        try:
            root = _tk.Tk()
            root.withdraw()
            _mb.showerror("Spectral Analyzer Pro - Missing Packages", msg)
            root.destroy()
        except Exception:
            print("ERROR:", msg)
        sys.exit(1)

_check_dependencies()

# ==============================================================================
# REGULAR IMPORTS
# ==============================================================================

import os
import io
import re
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Machine Learning & Chemometrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, detrend
from scipy.spatial.distance import cdist
from scipy.stats import t

# SHAP for Explainable AI (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Matplotlib for Interactive Plotting
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Literal imports so PyInstaller packs them into the portable executable.
# The branch is never executed at runtime (avoids a slow startup).
if False:
    import openpyxl
    import openpyxl.cell._writer
    import openpyxl.workbook
    import openpyxl.worksheet
    import PIL
    import PIL.Image
    import joblib
    import threadpoolctl
    import matplotlib.backends.backend_tkagg
    import matplotlib.backends.backend_agg
    import matplotlib.backends._backend_tk
    import sklearn.utils._cython_blas
    import sklearn.utils._weight_vector
    import sklearn.neighbors._partition_nodes
    import sklearn.tree._utils
    import sklearn.metrics._pairwise_distances_reduction
    import scipy.special.cython_special
    import scipy.linalg.cython_blas
    import scipy.linalg.cython_lapack
    import scipy.sparse.csgraph._validation
    import shap
    import shap.explainers
    import shap.explainers._tree
    import shap.explainers._linear
    import shap.explainers._kernel
    import numba
    import llvmlite
    import slicer
    import cloudpickle
    import tqdm

# ==============================================================================
# CHEMOMETRIC ALGORITHMS, COLLINEARITY & PREPROCESSING
# ==============================================================================

def eliminate_collinearity(X_df, threshold_pct=90.0, y=None, use_vif=True, vif_threshold=10.0):
    """Eliminates collinear features.
    First applies pairwise correlation threshold (preferring higher |corr| with y when available).
    Then optionally removes high-VIF features iteratively.
    """
    corr_threshold = threshold_pct / 100.0
    if X_df.shape[1] <= 1:
        return X_df, []
    
    # Pairwise correlation step (prefer y-correlation when available)
    corr_matrix = X_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    if y is not None and len(y) == len(X_df):
        y = np.asarray(y).ravel()
        y_corrs = {}
        for col in X_df.columns:
            if np.std(X_df[col]) == 0 or np.std(y) == 0:
                y_corrs[col] = 0.0
            else:
                c = np.corrcoef(X_df[col], y)[0, 1]
                y_corrs[col] = 0.0 if not np.isfinite(c) else abs(c)
        for col in upper_tri.columns:
            high = upper_tri.index[upper_tri[col] > corr_threshold].tolist()
            for other in high:
                if other in y_corrs and col in y_corrs:
                    if y_corrs[col] < y_corrs[other]:
                        if col not in to_drop:
                            to_drop.append(col)
                    else:
                        if other not in to_drop:
                            to_drop.append(other)
    else:
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    
    X_reduced = X_df.drop(columns=to_drop, errors="ignore")
    
    # Optional VIF iterative removal (proper VIF = 1/(1-R²))
    if use_vif and X_reduced.shape[1] > 1 and X_reduced.shape[0] > X_reduced.shape[1] + 2:
        remaining = list(X_reduced.columns)
        while len(remaining) > 2:
            try:
                vifs = []
                for j, col in enumerate(remaining):
                    others = [c for c in remaining if c != col]
                    if not others:
                        vifs.append(1.0)
                        continue
                    X_others = X_reduced[others].values
                    y_col = X_reduced[col].values
                    # R² of this variable regressed on the others
                    from sklearn.linear_model import LinearRegression as _LR
                    lr = _LR()
                    lr.fit(X_others, y_col)
                    r2 = r2_score(y_col, lr.predict(X_others))
                    vif = 1.0 / (1.0 - r2) if r2 < 0.999 else 999.0
                    vifs.append(vif)
                vifs = np.array(vifs)
                max_idx = np.argmax(vifs)
                if vifs[max_idx] > vif_threshold:
                    drop_col = remaining[max_idx]
                    remaining.remove(drop_col)
                    to_drop.append(drop_col)
                    X_reduced = X_reduced[remaining]
                else:
                    break
            except Exception:
                break
    
    return X_reduced, to_drop

def apply_snv(X_data):
    """Applies Standard Normal Variate transformation row-wise."""
    mean = np.mean(X_data, axis=1, keepdims=True)
    std = np.std(X_data, axis=1, keepdims=True)
    std[std == 0] = 1e-8
    return (X_data - mean) / std

def apply_msc(X_data, mean_spectrum=None):
    """Applies Multiplicative Scatter Correction (MSC).

    If mean_spectrum is given (calibration reference), it is used for every
    row. New samples must use the calibration reference, not their own mean.
    """
    if mean_spectrum is None:
        mean_spectrum = np.mean(X_data, axis=0)
    mean_spectrum = np.asarray(mean_spectrum, dtype=float).ravel()
    X_msc = np.zeros_like(X_data, dtype=float)
    for i in range(X_data.shape[0]):
        fit = np.polyfit(mean_spectrum, X_data[i, :], 1)
        slope = fit[0] if fit[0] != 0 else 1e-8
        X_msc[i, :] = (X_data[i, :] - fit[1]) / slope
    return X_msc

def _safe_sg_window(n_features, window, poly, deriv):
    """Odd Savitzky-Golay window that fits the spectrum. None if it cannot."""
    poly = max(int(poly), int(deriv))
    w = int(window) if int(window) % 2 == 1 else int(window) + 1
    if w <= poly:
        w = poly + 1 if (poly + 1) % 2 == 1 else poly + 2
    if w > n_features:
        w = n_features if n_features % 2 == 1 else n_features - 1
    if w <= poly or w < 3:
        return None
    return w, poly

def apply_preprocessing(X_df, method="Raw", sg_window=15, sg_poly=2):
    """Applies selected preprocessing method to spectral dataframe.
    Non-numeric values are coerced to NaN so a stray string does not crash the pipeline.
    """
    # Safe conversion: strings / empty cells become NaN instead of raising
    X_num = X_df.apply(pd.to_numeric, errors="coerce")
    X = X_num.values.astype(float).copy()
    n_features = X.shape[1]
    sg = _safe_sg_window(n_features, sg_window, sg_poly, 0)

    if method == "Raw":
        X_proc = X
    elif method == "Standard Normal Variate (SNV)":
        X_proc = apply_snv(X)
    elif method == "Multiplicative Scatter Correction (MSC)":
        X_proc = apply_msc(X)
    elif method == "Savitzky-Golay Filter":
        if sg is None:
            X_proc = X
        else:
            w, poly = sg
            X_proc = savgol_filter(X, window_length=w, polyorder=poly, deriv=0)
    elif method == "1st Derivative (Savitzky-Golay)":
        sg1 = _safe_sg_window(n_features, sg_window, sg_poly, 1)
        if sg1 is None:
            X_proc = X
        else:
            w, poly = sg1
            X_proc = savgol_filter(X, window_length=w, polyorder=poly, deriv=1)
    elif method == "2nd Derivative (Savitzky-Golay)":
        sg2 = _safe_sg_window(n_features, sg_window, min(sg_poly, 2), 2)
        if sg2 is None:
            X_proc = X
        else:
            w, poly = sg2
            X_proc = savgol_filter(X, window_length=w, polyorder=poly, deriv=2)
    elif method == "SNV + 1st Derivative":
        X_snv = apply_snv(X)
        sg1 = _safe_sg_window(n_features, sg_window, sg_poly, 1)
        if sg1 is None:
            X_proc = X_snv
        else:
            w, poly = sg1
            X_proc = savgol_filter(X_snv, window_length=w, polyorder=poly, deriv=1)
    elif method == "Baseline Detrending":
        X_proc = detrend(X, axis=1)
    elif method == "Mean Centering":
        X_proc = X - np.mean(X, axis=0, keepdims=True)
    else:
        X_proc = X

    return pd.DataFrame(X_proc, index=X_df.index, columns=X_df.columns)

def kennard_stone_split(X, test_size=0.25):
    """Performs Kennard-Stone split on standardized feature space."""
    X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    n_samples = X_scaled.shape[0]
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for a train/test split.")
    n_test = max(1, int(np.floor(n_samples * test_size)))
    n_test = min(n_test, n_samples - 2)

    distances = cdist(X_scaled, X_scaled, metric='euclidean')
    np.fill_diagonal(distances, -1.0)
    i, j = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    selected = [int(i), int(j)]
    remaining = [k for k in range(n_samples) if k not in selected]
    
    while len(selected) < (n_samples - n_test):
        sub_dist = distances[remaining][:, selected]
        min_dists = np.min(sub_dist, axis=1)
        next_point = remaining[np.argmax(min_dists)]
        selected.append(next_point)
        remaining.remove(next_point)
        
    return selected, remaining

def concordance_correlation_coefficient(y_true, y_pred):
    """Calculates Lin's Concordance Correlation Coefficient (CCC)."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    
    if len(y_t) < 2:
        return 0.0
        
    mean_true = np.mean(y_t)
    mean_pred = np.mean(y_p)
    var_true = np.var(y_t, ddof=0)
    var_pred = np.var(y_p, ddof=0)
    
    corr_matrix = np.corrcoef(y_t, y_p)
    corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]) else 0.0
        
    sd_true = np.std(y_t, ddof=0)
    sd_pred = np.std(y_p, ddof=0)
    
    numerator = 2 * corr * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    return float(numerator / denominator) if denominator != 0 else 0.0

def calculate_rpd(y_true, rmse):
    """Calculates Residual Predictive Deviation (RPD)."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    sd = np.std(y_t, ddof=1) if len(y_t) > 1 else 0.0
    if rmse == 0 or np.isnan(sd) or rmse is None:
        return 0.0
    return float(sd / rmse)

def adjusted_r2(r2, n, p):
    """Calculates Adjusted R2."""
    if n - p - 1 <= 0:
        return float(r2)
    adj = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))
    return float(adj)

def golbraikh_tropsha_criteria(y_true, y_pred):
    """Calculates Golbraikh-Tropsha external validation parameters.

    Classic / most commonly cited form (used for the Pass/Fail decision):
      • R² > 0.6
      • |R² − R₀²| < 0.1          (absolute difference)
      • |R² − R₀′²| < 0.1         (absolute difference)
      • 0.85 ≤ k ≤ 1.15
      • 0.85 ≤ k′ ≤ 1.15

    Relative differences |R² − R₀²|/R² are also computed and reported for reference
    (some papers use the relative form; both are shown).
    """
    y_t = np.array(y_true, dtype=float).ravel()
    y_p = np.array(y_pred, dtype=float).ravel()
    
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.std(y_t) > 0 else float("nan")
    if not np.isfinite(r2):
        return {
            'r2': r2 if np.isfinite(r2) else 0.0, 'k': 0.0, 'k_prime': 0.0,
            'r2_0': 0.0, 'r2_0_prime': 0.0,
            'diff_r2_abs': 1.0, 'diff_r2_prime_abs': 1.0,
            'diff_r2_rel': 1.0, 'diff_r2_prime_rel': 1.0,
            'passed': False
        }
    sum_yp2 = np.sum(y_p ** 2)
    sum_yt2 = np.sum(y_t ** 2)
    ss_t = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_p = np.sum((y_p - np.mean(y_p)) ** 2)

    k = float(np.sum(y_t * y_p) / sum_yp2) if sum_yp2 != 0 else 0.0
    r2_0 = float(1 - (np.sum((y_t - k * y_p) ** 2) / ss_t)) if ss_t != 0 else 0.0
    k_prime = float(np.sum(y_t * y_p) / sum_yt2) if sum_yt2 != 0 else 0.0
    r2_0_prime = float(1 - (np.sum((y_p - k_prime * y_t) ** 2) / ss_p)) if ss_p != 0 else 0.0

    diff_r2_abs = float(abs(r2 - r2_0))
    diff_r2_prime_abs = float(abs(r2 - r2_0_prime))
    diff_r2_rel = float(abs(r2 - r2_0) / r2) if r2 > 0 else 1.0
    diff_r2_prime_rel = float(abs(r2 - r2_0_prime) / r2) if r2 > 0 else 1.0

    # Classic absolute-difference criteria (most commonly cited)
    passed_all = bool(
        (r2 > 0.6) and 
        (diff_r2_abs < 0.1) and 
        (diff_r2_prime_abs < 0.1) and 
        (0.85 <= k <= 1.15) and 
        (0.85 <= k_prime <= 1.15)
    )
    
    return {
        'r2': r2, 'k': k, 'k_prime': k_prime, 
        'r2_0': r2_0, 'r2_0_prime': r2_0_prime, 
        'diff_r2_abs': diff_r2_abs, 'diff_r2_prime_abs': diff_r2_prime_abs,
        'diff_r2_rel': diff_r2_rel, 'diff_r2_prime_rel': diff_r2_prime_rel, 
        'passed': passed_all
    }

def stepwise_regression_fit(X, y, max_vars=15, p_enter=0.05, p_remove=0.10):
    """Forward-Backward Stepwise MLR with safer defaults and BIC early-stopping for small samples.
    - max_vars capped near n/5 (stricter when n is very small).
    - p_enter stricter on small n.
    - BIC check prevents adding variables that do not improve the information criterion.
    - Falls back to the single strongest univariate predictor if nothing enters.
    """
    n_samples, n_features = X.shape
    y_flat = y.ravel()

    # Safer limits based on sample size
    safe_max = max(1, min(max_vars, max(1, n_samples // 5), n_features))
    if n_samples < 30:
        safe_max = min(safe_max, 3)
    elif n_samples < 50:
        safe_max = min(safe_max, 5)

    effective_p_enter = p_enter
    if n_samples < 40:
        effective_p_enter = min(p_enter, 0.01)
    elif n_samples < 80:
        effective_p_enter = min(p_enter, 0.025)

    current_features = []
    best_bic = np.inf

    for step in range(safe_max):
        best_pval = 1.0
        best_feature = -1
        best_trial_bic = np.inf

        remaining = [f for f in range(n_features) if f not in current_features]
        for f in remaining:
            trial_features = current_features + [f]
            X_sub = X[:, trial_features]
            X_design = np.hstack([np.ones((n_samples, 1)), X_sub])

            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_sub, y_flat)
            y_pred = lr.predict(X_sub)
            resid = y_flat - y_pred
            rss = np.sum(resid ** 2)
            df = n_samples - len(trial_features) - 1
            if df <= 0 or rss <= 0:
                continue
            # BIC = n*ln(RSS/n) + k*ln(n)
            k = len(trial_features) + 1
            bic = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)
            s2 = rss / df
            try:
                XTX_inv = np.linalg.pinv(np.dot(X_design.T, X_design))
                se = np.sqrt(np.maximum(0, np.diagonal(s2 * XTX_inv)))
                t_stats = np.append([lr.intercept_], lr.coef_) / (se + 1e-8)
                p_vals = 2 * (1 - t.cdf(np.abs(t_stats), df))
                p_val = p_vals[-1]
                if p_val < best_pval and bic < best_trial_bic:
                    best_pval = p_val
                    best_feature = f
                    best_trial_bic = bic
            except Exception:
                continue

        # Only add if p-value passes AND BIC improves
        if best_feature != -1 and best_pval <= effective_p_enter and best_trial_bic < best_bic:
            current_features.append(best_feature)
            best_bic = best_trial_bic
        else:
            break

        # Backward elimination step
        if len(current_features) > 2:
            X_sub = X[:, current_features]
            X_design = np.hstack([np.ones((n_samples, 1)), X_sub])
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_sub, y_flat)
            y_pred = lr.predict(X_sub)
            resid = y_flat - y_pred
            rss = np.sum(resid ** 2)
            df = n_samples - len(current_features) - 1
            if df > 0:
                s2 = rss / df
                try:
                    XTX_inv = np.linalg.pinv(np.dot(X_design.T, X_design))
                    se = np.sqrt(np.maximum(0, np.diagonal(s2 * XTX_inv)))
                    t_stats = np.append([lr.intercept_], lr.coef_) / (se + 1e-8)
                    p_vals = 2 * (1 - t.cdf(np.abs(t_stats), df))
                    worst_idx = np.argmax(p_vals[1:])
                    if p_vals[1:][worst_idx] > p_remove:
                        current_features.pop(worst_idx)
                except Exception:
                    pass

    # Fallback: strongest univariate predictor
    if not current_features:
        corrs = []
        for i in range(n_features):
            if np.std(X[:, i]) == 0 or np.std(y_flat) == 0:
                corrs.append(0.0)
            else:
                c = np.corrcoef(X[:, i], y_flat)[0, 1]
                corrs.append(0.0 if not np.isfinite(c) else abs(c))
        current_features = [int(np.argmax(corrs))]

    return current_features


class PipelinePCARegression:
    """Helper wrapper for Principal Component Regression (PCR)."""
    def __init__(self, max_components=5):
        self.max_components = max_components
        self.pca = None
        self.ols = LinearRegression()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_comp = max(1, min(self.max_components, n_samples - 1, n_features))
        self.pca = PCA(n_components=n_comp)
        X_pca = self.pca.fit_transform(X)
        self.ols.fit(X_pca, y)
        self.coef_ = np.dot(self.ols.coef_.reshape(1, -1), self.pca.components_).ravel()
        self.intercept_ = float(self.ols.intercept_)
        return self

    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.ols.predict(X_pca)


class SafePLSRegression:
    """PLSR wrapper ensuring component safety."""
    def __init__(self, max_components=5):
        self.max_components = max_components
        self.pls = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_comp = max(1, min(self.max_components, n_samples - 1, n_features))
        self.pls = PLSRegression(n_components=n_comp)
        self.pls.fit(X, y)
        self.coef_ = np.asarray(self.pls.coef_).ravel()
        if hasattr(self.pls, "intercept_"):
            self.intercept_ = float(np.asarray(self.pls.intercept_).ravel()[0])
        else:
            y_mean = np.asarray(getattr(self.pls, "y_mean_", getattr(self.pls, "_y_mean", 0.0))).ravel()
            x_mean = np.asarray(getattr(self.pls, "x_mean_", getattr(self.pls, "_x_mean", np.zeros(self.coef_.shape)))).ravel()
            self.intercept_ = float(y_mean[0] - np.dot(x_mean, self.coef_)) if y_mean.size else 0.0
        return self

    def predict(self, X):
        return self.pls.predict(X).ravel()


def detect_outliers_mahalanobis(X, threshold=3.0):
    """Mahalanobis distance outlier flags (leverage in feature space)."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 5 or X.shape[1] < 1:
        return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0])
    try:
        mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        diff = X - mu
        d2 = np.sum(diff @ inv_cov * diff, axis=1)
        d = np.sqrt(np.maximum(d2, 0))
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + 1e-8
        z = 0.6745 * (d - med) / mad
        outliers = np.abs(z) > threshold
        return outliers, d
    except Exception:
        return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0])


def detect_outliers_residual_leverage(y_true, y_pred, X, residual_z=3.0, t2_threshold=None):
    """Combined residual + leverage + proper Hotelling's T² outlier detection.
    Returns boolean mask, summary, and diagnostic arrays (residuals, leverage, T²).

    Hotelling's T² critical value uses the F-distribution:
        T²_crit = [p*(n-1)/(n-p)] * F_{α, p, n-p}
    (α = 0.05 by default). Falls back to robust z-score if the F form is not applicable
    (e.g., p ≥ n-1).
    """
    from scipy.stats import f as f_dist
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n = len(y_true)
    if n < 5:
        return np.zeros(n, dtype=bool), "Too few samples", None, None, None
    
    resid = y_true - y_pred
    med_r = np.median(resid)
    mad_r = np.median(np.abs(resid - med_r)) + 1e-8
    z_res = 0.6745 * (resid - med_r) / mad_r
    res_out = np.abs(z_res) > residual_z
    
    # Leverage / Mahalanobis
    lev_out, lev_d = detect_outliers_mahalanobis(X, threshold=3.0)
    
    # Proper Hotelling's T²
    try:
        mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        diff = X - mu
        t2 = np.sum(diff @ inv_cov * diff, axis=1)
        p = X.shape[1]
        # Proper critical value when degrees of freedom allow it
        if p < n - 1 and p > 0:
            alpha = 0.05
            f_crit = f_dist.ppf(1 - alpha, p, n - p)
            t2_crit = (p * (n - 1) / (n - p)) * f_crit
            t2_out = t2 > t2_crit
        else:
            # Fallback: robust z-score on T²
            med_t = np.median(t2)
            mad_t = np.median(np.abs(t2 - med_t)) + 1e-8
            z_t = 0.6745 * (t2 - med_t) / mad_t
            t2_out = np.abs(z_t) > 3.0
    except Exception:
        t2 = np.zeros(n)
        t2_out = np.zeros(n, dtype=bool)
    
    combined = res_out | lev_out | t2_out
    summary = f"{int(np.sum(combined))} potential outliers (residual, leverage, or Hotelling T²)"
    return combined, summary, resid, lev_d, t2


def calculate_vip(pls_model, X, y):
    """Calculate Variable Importance in Projection (VIP) scores for a fitted PLS model.
    
    VIP_j = sqrt( p * sum_a (SS_a * w_aj^2) / sum_a SS_a )
    where SS_a is the variance explained by component a.
    """
    try:
        t = pls_model.x_scores_
        w = pls_model.x_weights_
        q = pls_model.y_loadings_
        p = X.shape[1]
        # Variance explained by each component
        ss = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)
        total_ss = np.sum(ss)
        if total_ss <= 0:
            return np.ones(p) / p
        vip = np.zeros(p)
        for j in range(p):
            weight = np.array([(w[j, a] ** 2) * ss[a] for a in range(len(ss))])
            vip[j] = np.sqrt(p * np.sum(weight) / total_ss)
        return vip
    except Exception:
        return None


def select_features_by_vip(X, y, max_features=50, vip_threshold=1.0, n_components=5):
    """Select wavelengths/features using VIP from a quick PLS fit.
    Returns boolean mask of selected features and the VIP scores.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_samples, n_features = X.shape
    if n_features <= 1 or n_samples < 5:
        return np.ones(n_features, dtype=bool), np.ones(n_features)

    nc = max(1, min(n_components, n_samples - 2, n_features))
    try:
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        pls = SafePLSRegression(max_components=nc)
        pls.fit(Xs, y)
        vip = calculate_vip(pls.pls, Xs, y)
        if vip is None:
            vip = np.ones(n_features)
        # Prefer features with VIP >= threshold, then cap at max_features
        selected = vip >= vip_threshold
        if selected.sum() == 0:
            # Fall back to top features
            top = np.argsort(vip)[-min(max_features, n_features):]
            selected = np.zeros(n_features, dtype=bool)
            selected[top] = True
        elif selected.sum() > max_features:
            top = np.argsort(vip)[-max_features:]
            selected = np.zeros(n_features, dtype=bool)
            selected[top] = True
        return selected, vip
    except Exception:
        return np.ones(n_features, dtype=bool), np.ones(n_features)


def select_features_by_correlation(X, y, max_features=50, abs_corr_threshold=0.3):
    """Select features with highest absolute correlation to y."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_features = X.shape[1]
    if n_features <= 1:
        return np.ones(n_features, dtype=bool), np.zeros(n_features)

    corrs = np.zeros(n_features)
    for j in range(n_features):
        if np.std(X[:, j]) == 0 or np.std(y) == 0:
            corrs[j] = 0.0
        else:
            c = np.corrcoef(X[:, j], y)[0, 1]
            corrs[j] = 0.0 if not np.isfinite(c) else abs(c)

    selected = corrs >= abs_corr_threshold
    if selected.sum() == 0 or selected.sum() > max_features:
        top = np.argsort(corrs)[-min(max_features, n_features):]
        selected = np.zeros(n_features, dtype=bool)
        selected[top] = True
    return selected, corrs


def select_features_interval(X, y, interval_width=10, max_intervals=8, n_components=3):
    """Simple interval selection: evaluate consecutive blocks with a quick PLS
    and keep the best-performing intervals (a lightweight iPLS-style approach).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_samples, n_features = X.shape
    if n_features <= interval_width or n_samples < 8:
        return np.ones(n_features, dtype=bool), None

    scores = []
    starts = list(range(0, n_features - interval_width + 1, max(1, interval_width // 2)))
    for start in starts:
        end = start + interval_width
        X_int = X[:, start:end]
        try:
            sc = StandardScaler()
            Xs = sc.fit_transform(X_int)
            nc = max(1, min(n_components, n_samples - 2, interval_width))
            pls = SafePLSRegression(max_components=nc)
            # Quick internal score
            if n_samples >= 12:
                tr, va = train_test_split(np.arange(n_samples), test_size=0.3, random_state=42)
                pls.fit(Xs[tr], y[tr])
                pred = pls.predict(Xs[va])
                q = r2_score(y[va], pred) if np.std(y[va]) > 0 else -np.inf
            else:
                pls.fit(Xs, y)
                q = r2_score(y, pls.predict(Xs))
            scores.append((q, start, end))
        except Exception:
            scores.append((-np.inf, start, end))

    scores.sort(reverse=True)
    selected = np.zeros(n_features, dtype=bool)
    kept = 0
    for q, start, end in scores:
        if kept >= max_intervals:
            break
        if q > -np.inf:
            selected[start:end] = True
            kept += 1
    if selected.sum() == 0:
        selected[:] = True
    return selected, scores


def evaluate_q2_cv(model_name, X_raw, y, cv_folds=5, seed=42, alpha=None, n_components=None,
                   tree_params=None, full_nested=False):
    """Computes leak-free cross-validated Q² by executing scaling inside CV loops.

    For linear/regularized models and PLS/PCR the optional alpha / n_components
    make Q² consistent with the final tuned model.

    For tree ensembles:
    - Default (full_nested=False): light nested selection inside each fold (small grid).
    - full_nested=True: larger grid inside each fold for maximum rigor (slower).
    """
    if len(y) < 3:
        return float("nan")
    n_splits = min(cv_folds, len(y))
    if n_splits < 2:
        return float("nan")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_hat_cv = np.zeros_like(y)

    is_tree = any(x in model_name for x in ("Random Forest", "Extra Trees", "Gradient Boosting"))

    for train_idx, val_idx in cv.split(X_raw):
        X_tr, X_val = X_raw[train_idx], X_raw[val_idx]
        y_tr = y[train_idx]

        if model_name == "Stepwise MLR":
            scaler_temp = StandardScaler()
            X_tr_sc = scaler_temp.fit_transform(X_tr)
            step_sel = stepwise_regression_fit(X_tr_sc, y_tr, max_vars=15)
            X_tr_in = X_tr[:, step_sel]
            X_val_in = X_val[:, step_sel]
            model = LinearRegression()
        else:
            X_tr_in, X_val_in = X_tr, X_val
            if "PLSR" in model_name:
                nc = select_best_n_components("PLSR", X_tr, y_tr, max_comp=5, seed=seed) if n_components is None else n_components
                model = SafePLSRegression(max_components=nc)
            elif "PCR" in model_name:
                nc = select_best_n_components("PCR", X_tr, y_tr, max_comp=5, seed=seed) if n_components is None else n_components
                model = PipelinePCARegression(max_components=nc)
            elif "Ridge" in model_name:
                a = select_best_alpha("Ridge", X_tr, y_tr, seed=seed) if alpha is None else alpha
                model = Ridge(alpha=a)
            elif "Lasso" in model_name:
                a = select_best_alpha("Lasso", X_tr, y_tr, seed=seed) if alpha is None else alpha
                model = Lasso(alpha=a, max_iter=10000)
            elif "ElasticNet" in model_name:
                a = select_best_alpha("ElasticNet", X_tr, y_tr, seed=seed) if alpha is None else alpha
                model = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000)
            elif "SVR" in model_name:
                svr_C = select_best_svr_C(X_tr, y_tr, seed=seed)
                model = SVR(kernel='rbf', C=svr_C, epsilon=0.1)
            elif is_tree:
                # Nested selection inside the fold
                fold_params = _select_tree_params_light(model_name, X_tr, y_tr, seed=seed, full=full_nested)
                d, leaf, ne = fold_params
                if "Random Forest" in model_name:
                    model = RandomForestRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                elif "Extra Trees" in model_name:
                    model = ExtraTreesRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                else:
                    model = GradientBoostingRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
            elif "Gaussian Process" in model_name:
                length = select_best_gpr_length(X_tr, y_tr, seed=seed)
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length, (1e-2, 1e2))
                model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, random_state=42)
            elif "kNN" in model_name:
                k = select_best_knn_k(X_tr, y_tr, seed=seed)
                model = KNeighborsRegressor(n_neighbors=k)
            elif "MLP" in model_name:
                a = select_best_mlp_alpha(X_tr, y_tr, seed=seed)
                model = MLPRegressor(hidden_layer_sizes=(30, 15), max_iter=600, alpha=a, random_state=42)
            else:
                raise ValueError(f"Unknown model for cross-validation: {model_name}")

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_in)
        X_val_sc = scaler.transform(X_val_in)

        model.fit(X_tr_sc, y_tr)
        y_hat_cv[val_idx] = np.asarray(model.predict(X_val_sc)).ravel()

    return float(r2_score(y, y_hat_cv))


def _select_tree_params_light(model_name, X, y, seed=42, full=False):
    """Hyperparameter selection for tree models.
    full=False: tiny grid (fast). full=True: larger grid (maximum rigor).
    """
    n = len(y)
    if full:
        if "Gradient" in model_name:
            depths = [2, 3, 4, 5]
            leaves = [1, 2, 3, 5]
            n_ests = [50, 100]
            default = (3, 3, 100)
        else:
            depths = [3, 4, 6, 8, 10] if n >= 60 else [3, 4, 6]
            leaves = [1, 2, 3, 5]
            n_ests = [50, 100]
            default = (6, 3, 100)
    else:
        if "Gradient" in model_name:
            depths = [2, 3]
            leaves = [2, 3]
            n_ests = [50]
            default = (3, 3, 50)
        else:
            depths = [4, 6] if n >= 40 else [3, 4]
            leaves = [2, 3]
            n_ests = [50]
            default = (4, 3, 50)

    candidates = [(d, leaf, ne) for d in depths for leaf in leaves for ne in n_ests if leaf < max(2, n // 3)]
    if not candidates:
        return default

    best_params = default
    best_q = -np.inf
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    if n >= 10:
        tr, va = train_test_split(np.arange(n), test_size=0.3, random_state=seed)
    else:
        tr = va = np.arange(n)

    for d, leaf, ne in candidates:
        try:
            if "Random Forest" in model_name:
                mdl = RandomForestRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
            elif "Extra Trees" in model_name:
                mdl = ExtraTreesRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
            else:
                mdl = GradientBoostingRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
            mdl.fit(Xs[tr], y[tr])
            pred = mdl.predict(Xs[va])
            q = r2_score(y[va], pred) if len(va) > 1 and np.std(y[va]) > 0 else -np.inf
            if q > best_q:
                best_q = q
                best_params = (d, leaf, ne)
        except Exception:
            continue
    return best_params


def select_best_alpha(model_type, X, y, seed=42):
    """Grid search for Ridge/Lasso/ElasticNet alpha using 3-fold CV on training data.
    Uses joblib for parallel evaluation of the alpha grid when beneficial.
    """
    alphas = [1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    best_alpha = 1.0
    best_q2 = -np.inf
    n_splits = min(3, max(2, len(y) // 5))
    if n_splits < 2 or len(y) < 6:
        return 1.0 if model_type == "Ridge" else 0.01
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _eval_alpha(alpha):
        y_hat = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xva_s = sc.transform(Xva)
            if model_type == "Ridge":
                mdl = Ridge(alpha=alpha)
            elif model_type == "Lasso":
                mdl = Lasso(alpha=alpha, max_iter=10000)
            else:
                mdl = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
            mdl.fit(Xtr_s, ytr)
            y_hat[va_idx] = mdl.predict(Xva_s)
        q2 = r2_score(y, y_hat) if np.std(y) > 0 else -np.inf
        return alpha, q2

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, prefer="threads")(delayed(_eval_alpha)(a) for a in alphas)
        for alpha, q2 in results:
            if q2 > best_q2:
                best_q2 = q2
                best_alpha = alpha
    except Exception:
        # Fallback to sequential
        for alpha in alphas:
            _, q2 = _eval_alpha(alpha)
            if q2 > best_q2:
                best_q2 = q2
                best_alpha = alpha
    return best_alpha


def select_best_n_components(model_type, X, y, max_comp=5, seed=42):
    """Select number of components for PLSR or PCR by maximizing 3-fold Q² on training data.
    Parallelized with joblib when beneficial.
    """
    n_samples = len(y)
    max_possible = max(1, min(max_comp, n_samples - 2, X.shape[1]))
    if n_samples < 6:
        return min(2, max_possible)
    best_nc = 1
    best_q2 = -np.inf
    n_splits = min(3, max(2, n_samples // 5))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _eval_nc(nc):
        y_hat = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xva_s = sc.transform(Xva)
            if model_type == "PLSR":
                mdl = SafePLSRegression(max_components=nc)
            else:
                mdl = PipelinePCARegression(max_components=nc)
            mdl.fit(Xtr_s, ytr)
            y_hat[va_idx] = np.asarray(mdl.predict(Xva_s)).ravel()
        q2 = r2_score(y, y_hat) if np.std(y) > 0 else -np.inf
        return nc, q2

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, prefer="threads")(delayed(_eval_nc)(nc) for nc in range(1, max_possible + 1))
        for nc, q2 in results:
            if q2 > best_q2:
                best_q2 = q2
                best_nc = nc
    except Exception:
        for nc in range(1, max_possible + 1):
            _, q2 = _eval_nc(nc)
            if q2 > best_q2:
                best_q2 = q2
                best_nc = nc
    return best_nc


def select_best_svr_C(X, y, seed=42):
    """Select SVR C by 3-fold CV."""
    Cs = [0.1, 1.0, 10.0, 100.0]
    best_C = 10.0
    best_q2 = -np.inf
    n_splits = min(3, max(2, len(y) // 5))
    if n_splits < 2 or len(y) < 6:
        return 10.0
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c_val in Cs:
        y_hat = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xva_s = sc.transform(Xva)
            mdl = SVR(kernel='rbf', C=c_val, epsilon=0.1)
            mdl.fit(Xtr_s, ytr)
            y_hat[va_idx] = mdl.predict(Xva_s)
        q2 = r2_score(y, y_hat) if np.std(y) > 0 else -np.inf
        if q2 > best_q2:
            best_q2 = q2
            best_C = c_val
    return best_C


def select_best_knn_k(X, y, seed=42):
    """Select k for kNN by 3-fold CV."""
    max_k = min(10, max(1, len(y) // 3))
    ks = list(range(1, max_k + 1))
    best_k = min(5, max_k)
    best_q2 = -np.inf
    n_splits = min(3, max(2, len(y) // 5))
    if n_splits < 2 or len(y) < 6:
        return best_k
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k in ks:
        y_hat = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xva_s = sc.transform(Xva)
            mdl = KNeighborsRegressor(n_neighbors=k)
            mdl.fit(Xtr_s, ytr)
            y_hat[va_idx] = mdl.predict(Xva_s)
        q2 = r2_score(y, y_hat) if np.std(y) > 0 else -np.inf
        if q2 > best_q2:
            best_q2 = q2
            best_k = k
    return best_k


def select_best_gpr_length(X, y, seed=42):
    """Select RBF length scale for GPR by 3-fold CV."""
    lengths = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_len = 1.0
    best_q2 = -np.inf
    n_splits = min(3, max(2, len(y) // 5))
    if n_splits < 2 or len(y) < 6:
        return 1.0
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for length in lengths:
        y_hat = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xva_s = sc.transform(Xva)
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length, (1e-2, 1e2))
            mdl = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, random_state=42)
            mdl.fit(Xtr_s, ytr)
            y_hat[va_idx] = mdl.predict(Xva_s)
        q2 = r2_score(y, y_hat) if np.std(y) > 0 else -np.inf
        if q2 > best_q2:
            best_q2 = q2
            best_len = length
    return best_len


def select_best_mlp_alpha(X, y, seed=42):
    """Select MLP L2 alpha by 3-fold CV."""
    alphas = [1e-4, 1e-3, 0.01, 0.1]
    best_alpha = 0.01
    best_q2 = -np.inf
    n_splits = min(3, max(2, len(y) // 5))
    if n_splits < 2 or len(y) < 6:
        return 0.01
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for alpha in alphas:
        y_hat = np.zeros_like(y, dtype=float)
        for tr_idx, va_idx in cv.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xva_s = sc.transform(Xva)
            mdl = MLPRegressor(hidden_layer_sizes=(30, 15), max_iter=600, alpha=alpha, random_state=42)
            mdl.fit(Xtr_s, ytr)
            y_hat[va_idx] = mdl.predict(Xva_s)
        q2 = r2_score(y, y_hat) if np.std(y) > 0 else -np.inf
        if q2 > best_q2:
            best_q2 = q2
            best_alpha = alpha
    return best_alpha


class ClosedPipeline:
    """Minimal closed pipeline that stores preprocessing + model + scaler
    and can transform + predict new raw spectra.
    """
    def __init__(self, model, scaler, feature_names, preprocessing,
                 sg_window=15, sg_poly=2, msc_reference=None, mean_centering_mean=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)
        self.preprocessing = preprocessing
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.msc_reference = msc_reference
        self.mean_centering_mean = mean_centering_mean

    def transform(self, df_new):
        feats = self.feature_names
        # Keep only the columns the model expects and force numeric conversion
        present = [f for f in feats if f in df_new.columns]
        if len(present) < len(feats):
            raise ValueError(
                f"New file is missing {len(feats) - len(present)} required feature(s). "
                f"First missing: { [f for f in feats if f not in df_new.columns][:6] }"
            )
        work = df_new[present].copy()
        # Handle European decimal commas and other non-numeric strings
        for col in work.columns:
            if work[col].dtype == object:
                work[col] = work[col].astype(str).str.replace(",", ".", regex=False)
        work = work.apply(pd.to_numeric, errors="coerce")

        if self.preprocessing == "Mean Centering" and self.mean_centering_mean is not None:
            X_raw = work.values
            X_proc = pd.DataFrame(X_raw - self.mean_centering_mean, columns=present, index=work.index)
        elif self.preprocessing == "Multiplicative Scatter Correction (MSC)" and self.msc_reference is not None:
            X_raw = work.values
            X_msc = apply_msc(X_raw, self.msc_reference)
            X_proc = pd.DataFrame(X_msc, columns=present, index=work.index)
        else:
            X_proc_df = apply_preprocessing(work, method=self.preprocessing,
                                            sg_window=self.sg_window, sg_poly=self.sg_poly)
            X_proc = X_proc_df[present].apply(pd.to_numeric, errors="coerce")
        # Drop rows that still contain NaN after conversion
        good = X_proc.notna().all(axis=1)
        X_proc = X_proc.loc[good]
        if len(X_proc) == 0:
            raise ValueError(
                "No complete numeric rows remain after cleaning. "
                "Check that the spectral columns contain numbers (not sample names or text)."
            )
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_proc.values)
        else:
            X_scaled = X_proc.values
        return X_scaled, X_proc.index

    def predict(self, df_new):
        X_scaled, idx = self.transform(df_new)
        preds = np.asarray(self.model.predict(X_scaled)).ravel()
        return preds, idx

# ==============================================================================
# WIZARD DIALOG FOR DATA SETUP
# ==============================================================================

class DataSetupWizardDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("🛠️ Step-by-Step Data & Validation Setup")
        self.geometry("650x720")
        self.transient(parent)
        self.grab_set()

        self.result = None
        self.cal_x_raw = None
        self.cal_y_raw = None
        self.ext_x_raw = None
        self.ext_y_raw = None

        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Welcome to Spectral Analyzer Pro! Configure datasets and restrictions.", 
                  font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 10))

        cal_frame = ttk.LabelFrame(main_frame, text="1. Calibration Dataset (Required)", padding=10)
        cal_frame.pack(fill=tk.X, pady=5)

        ttk.Button(cal_frame, text="📁 Browse Spectral File (X)", command=self.load_cal_x).pack(side=tk.LEFT, padx=5)
        self.lbl_cal_x = ttk.Label(cal_frame, text="No file selected", foreground="gray")
        self.lbl_cal_x.pack(side=tk.LEFT, padx=5)

        ttk.Button(cal_frame, text="📁 Browse Target File (y)", command=self.load_cal_y).pack(side=tk.LEFT, padx=(20, 5))
        self.lbl_cal_y = ttk.Label(cal_frame, text="No file selected", foreground="gray")
        self.lbl_cal_y.pack(side=tk.LEFT, padx=5)

        collin_frame = ttk.LabelFrame(main_frame, text="1.1 Collinearity Elimination & Restriction", padding=10)
        collin_frame.pack(fill=tk.X, pady=5)
        
        self.collin_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(collin_frame, text="Enable Collinearity Reduction", variable=self.collin_var).pack(anchor="w", padx=5)
        
        pct_frame = ttk.Frame(collin_frame)
        pct_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(pct_frame, text="Max Inter-feature Correlation Restriction (%):").pack(side=tk.LEFT, padx=5)
        self.collin_pct_var = tk.StringVar(value="90")
        collin_spin = ttk.Spinbox(pct_frame, from_=50, to=99, textvariable=self.collin_pct_var, width=8)
        collin_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(pct_frame, text="(Removes redundant variables exceeding pairwise threshold)", foreground="gray").pack(side=tk.LEFT, padx=5)

        val_frame = ttk.LabelFrame(main_frame, text="2. Validation & Split Method", padding=10)
        val_frame.pack(fill=tk.X, pady=10)

        self.val_mode_var = tk.StringVar(value="split")
        
        rb1 = ttk.Radiobutton(val_frame, text="Automatically split Calibration data into Train / Test", 
                              variable=self.val_mode_var, value="split", command=self.toggle_mode)
        rb1.pack(anchor="w", pady=2)

        split_sub_frame = ttk.Frame(val_frame)
        split_sub_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(split_sub_frame, text="Split Algorithm:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.split_algo_var = tk.StringVar(value="Kennard-Stone")
        split_cb = ttk.Combobox(split_sub_frame, textvariable=self.split_algo_var, 
                                values=["Kennard-Stone", "Random Split"], state="readonly", width=18)
        split_cb.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(split_sub_frame, text="Test Set Percentage:").grid(row=0, column=2, sticky="w", padx=(15, 5), pady=2)
        self.test_pct_var = tk.StringVar(value="25%")
        pct_cb = ttk.Combobox(split_sub_frame, textvariable=self.test_pct_var, 
                              values=["15%", "20%", "25%", "30%", "35%"], state="readonly", width=8)
        pct_cb.grid(row=0, column=3, sticky="w", padx=5, pady=2)

        ttk.Label(split_sub_frame, text="Random Split Repeats (N):").grid(row=1, column=0, sticky="w", padx=5, pady=(8, 2))
        self.repeats_var = tk.StringVar(value="10")
        repeats_spin = ttk.Spinbox(split_sub_frame, from_=1, to=100, textvariable=self.repeats_var, width=8)
        repeats_spin.grid(row=1, column=1, sticky="w", padx=5, pady=(8, 2))

        rb2 = ttk.Radiobutton(val_frame, text="Use a separate external dataset file", 
                              variable=self.val_mode_var, value="separate", command=self.toggle_mode)
        rb2.pack(anchor="w", pady=(10, 2))

        self.ext_file_frame = ttk.Frame(val_frame)
        self.ext_file_frame.pack(fill=tk.X, padx=20, pady=5)

        self.btn_ext_x = ttk.Button(self.ext_file_frame, text="📁 External Spectral (X_ext)", command=self.load_ext_x, state=tk.DISABLED)
        self.btn_ext_x.pack(side=tk.LEFT, padx=5)
        self.lbl_ext_x = ttk.Label(self.ext_file_frame, text="Not loaded", foreground="gray")
        self.lbl_ext_x.pack(side=tk.LEFT, padx=5)

        self.btn_ext_y = ttk.Button(self.ext_file_frame, text="📁 External Target (y_ext)", command=self.load_ext_y, state=tk.DISABLED)
        self.btn_ext_y.pack(side=tk.LEFT, padx=(20, 5))
        self.lbl_ext_y = ttk.Label(self.ext_file_frame, text="Not loaded", foreground="gray")
        self.lbl_ext_y.pack(side=tk.LEFT, padx=5)

        confirm_btn = ttk.Button(main_frame, text="🚀 Confirm & Proceed to Column Selection", command=self.on_next)
        confirm_btn.pack(pady=10, ipadx=10, ipady=5)

    def toggle_mode(self):
        if self.val_mode_var.get() == "separate":
            self.btn_ext_x.config(state=tk.NORMAL)
            self.btn_ext_y.config(state=tk.NORMAL)
        else:
            self.btn_ext_x.config(state=tk.DISABLED)
            self.btn_ext_y.config(state=tk.DISABLED)

    def _read_table(self, path):
        lower = path.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(path)
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            return pd.read_excel(path)
        raise ValueError("Unsupported file type. Use .csv or .xlsx.")

    def load_cal_x(self):
        path = filedialog.askopenfilename(title="Select Calibration Spectral Matrix X", filetypes=[("CSV/Excel files", "*.csv *.xlsx")])
        if path:
            try:
                self.cal_x_raw = self._read_table(path)
                self.lbl_cal_x.config(text=os.path.basename(path), foreground="green")
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self)

    def load_cal_y(self):
        path = filedialog.askopenfilename(title="Select Calibration Target Property y", filetypes=[("CSV/Excel files", "*.csv *.xlsx")])
        if path:
            try:
                self.cal_y_raw = self._read_table(path)
                self.lbl_cal_y.config(text=os.path.basename(path), foreground="green")
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self)

    def load_ext_x(self):
        path = filedialog.askopenfilename(title="Select External Spectral Matrix X_ext", filetypes=[("CSV/Excel files", "*.csv *.xlsx")])
        if path:
            try:
                self.ext_x_raw = self._read_table(path)
                self.lbl_ext_x.config(text=os.path.basename(path), foreground="green")
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self)

    def load_ext_y(self):
        path = filedialog.askopenfilename(title="Select External Target Property y_ext", filetypes=[("CSV/Excel files", "*.csv *.xlsx")])
        if path:
            try:
                self.ext_y_raw = self._read_table(path)
                self.lbl_ext_y.config(text=os.path.basename(path), foreground="green")
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self)

    def on_next(self):
        if self.cal_x_raw is None or self.cal_y_raw is None:
            messagebox.showwarning("Warning", "Please load Calibration Spectral (X) and Target (y) files first.", parent=self)
            return

        if self.val_mode_var.get() == "separate" and (self.ext_x_raw is None or self.ext_y_raw is None):
            messagebox.showwarning("Warning", "Please load both External Spectral and Target files.", parent=self)
            return

        cal_res = ColumnSelectorDialog(self, self.cal_x_raw, self.cal_y_raw, title="Select Columns for Calibration Data")
        self.wait_window(cal_res)

        if not cal_res.result:
            return
        df_X_cal, df_y_cal = cal_res.result

        df_X_ext, df_y_ext = None, None
        if self.val_mode_var.get() == "separate":
            ext_res = ColumnSelectorDialog(self, self.ext_x_raw, self.ext_y_raw, title="Select Columns for External Test Data")
            self.wait_window(ext_res)
            if not ext_res.result:
                return
            df_X_ext, df_y_ext = ext_res.result

        test_ratio = float(self.test_pct_var.get().replace('%', '')) / 100.0
        n_repeats = int(self.repeats_var.get()) if self.split_algo_var.get() == "Random Split" else 1

        self.result = {
            'df_X_cal': df_X_cal,
            'df_y_cal': df_y_cal,
            'collinear_reduction': self.collin_var.get(),
            'collinear_threshold': float(self.collin_pct_var.get()),
            'val_mode': self.val_mode_var.get(),
            'split_algo': self.split_algo_var.get(),
            'test_size': test_ratio,
            'random_split_repeats': n_repeats,
            'df_X_ext': df_X_ext,
            'df_y_ext': df_y_ext
        }
        self.destroy()


class ColumnSelectorDialog(tk.Toplevel):
    def __init__(self, parent, df_x_raw, df_y_raw, title="Select Columns and Variables"):
        super().__init__(parent)
        self.title(title)
        self.geometry("550x500")
        self.transient(parent)
        self.grab_set()

        self.df_x_raw = df_x_raw
        self.df_y_raw = df_y_raw
        self.result = None

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        y_frame = ttk.LabelFrame(main_frame, text="Dependent Variable (y)", padding=8)
        y_frame.pack(fill=tk.X, pady=5)

        ttk.Label(y_frame, text="Target Column:").pack(side=tk.LEFT, padx=5)
        self.y_col_var = tk.StringVar()
        y_cb = ttk.Combobox(y_frame, textvariable=self.y_col_var, values=list(df_y_raw.columns), state="readonly")
        if len(df_y_raw.columns) > 0:
            y_cb.current(0)
        y_cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        x_frame = ttk.LabelFrame(main_frame, text="Spectral Features (X)", padding=8)
        x_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        btn_box = ttk.Frame(x_frame)
        btn_box.pack(fill=tk.X, pady=2)
        ttk.Button(btn_box, text="Select All", command=lambda: self.x_listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_box, text="Deselect All", command=lambda: self.x_listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=2)

        self.x_listbox = tk.Listbox(x_frame, selectmode=tk.MULTIPLE, exportselection=0)
        x_scroll = ttk.Scrollbar(x_frame, orient=tk.VERTICAL, command=self.x_listbox.yview)
        self.x_listbox.configure(yscrollcommand=x_scroll.set)

        self.x_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        x_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for col in df_x_raw.columns:
            self.x_listbox.insert(tk.END, str(col))
        self.x_listbox.select_set(0, tk.END)

        ttk.Button(main_frame, text="Confirm Columns", command=self.on_confirm).pack(pady=10)

    def on_confirm(self):
        sel_indices = self.x_listbox.curselection()
        if not sel_indices:
            messagebox.showwarning("Warning", "Please select at least one spectral column.", parent=self)
            return
        target_col = self.y_col_var.get()
        if not target_col:
            messagebox.showwarning("Warning", "Please select a target property column.", parent=self)
            return

        df_x = self.df_x_raw.copy()
        sel_cols = [self.df_x_raw.columns[i] for i in sel_indices]
        df_x = df_x[sel_cols]

        df_y = self.df_y_raw.copy()
        df_y = df_y[[target_col]]

        n = min(len(df_x), len(df_y))
        df_x = df_x.iloc[:n].copy()
        df_y = df_y.iloc[:n].copy()
        df_x = df_x.apply(pd.to_numeric, errors="coerce")
        df_y = df_y.apply(pd.to_numeric, errors="coerce")
        paired = df_x.notna().all(axis=1).to_numpy() & df_y.notna().all(axis=1).to_numpy()
        df_x = df_x.loc[paired].reset_index(drop=True)
        df_y = df_y.loc[paired].reset_index(drop=True)
        if len(df_x) < 3:
            messagebox.showwarning("Warning", "Fewer than 3 complete samples remain after removing missing values.", parent=self)
            return

        self.result = (df_x, df_y)
        self.destroy()

# ==============================================================================
# INTERACTIVE PLOT VIEWER WINDOW
# ==============================================================================

class PlotViewerWindow(tk.Toplevel):
    def __init__(self, parent, detailed_results):
        super().__init__(parent)
        self.title("📈 Predicted vs. Original Correlation Plotter & Customizer")
        self.geometry("880x640")
        self.minsize(680, 500)
        self.resizable(True, True)
        self.transient(parent)

        self.detailed_results = detailed_results
        model_names = list(detailed_results.keys())

        if not model_names:
            messagebox.showwarning("Warning", "Please run models first before opening plotter!", parent=self)
            self.destroy()
            return

        control_frame = ttk.LabelFrame(self, text="🎨 Plot Customization & Options", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.model_var = tk.StringVar(value=model_names[0])
        self.model_cb = ttk.Combobox(control_frame, textvariable=self.model_var, values=model_names, state="readonly", width=22)
        self.model_cb.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        ttk.Label(control_frame, text="Dataset:").grid(row=0, column=2, sticky="w", padx=(15, 5), pady=2)
        self.dataset_var = tk.StringVar(value="Validation (Test)")
        dataset_cb = ttk.Combobox(control_frame, textvariable=self.dataset_var, 
                                  values=[
                                      "Calibration (Train)", 
                                      "Validation (Test)", 
                                      "Combined (Separate Train & Test Series)", 
                                      "All Points Pooled (Train + Test Combined)"
                                  ], 
                                  state="readonly", width=38)
        dataset_cb.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        dataset_cb.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        ttk.Label(control_frame, text="Title:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.title_entry = ttk.Entry(control_frame, width=25)
        self.title_entry.insert(0, "Predicted vs Original Values")
        self.title_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(control_frame, text="X Label:").grid(row=1, column=2, sticky="w", padx=(15, 5), pady=2)
        self.xlabel_entry = ttk.Entry(control_frame, width=20)
        self.xlabel_entry.insert(0, "Original (Actual) Values")
        self.xlabel_entry.grid(row=1, column=3, sticky="w", padx=5, pady=2)

        ttk.Label(control_frame, text="Y Label:").grid(row=1, column=4, sticky="w", padx=(15, 5), pady=2)
        self.ylabel_entry = ttk.Entry(control_frame, width=20)
        self.ylabel_entry.insert(0, "Predicted Values")
        self.ylabel_entry.grid(row=1, column=5, sticky="w", padx=5, pady=2)

        ttk.Label(control_frame, text="Fig BgColor:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.fig_bg_var = tk.StringVar(value="white")
        fig_bg_cb = ttk.Combobox(control_frame, textvariable=self.fig_bg_var, values=["white", "lightgray", "ivory", "linen", "whitesmoke"], state="readonly", width=12)
        fig_bg_cb.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(control_frame, text="Axes BgColor:").grid(row=2, column=2, sticky="w", padx=(15, 5), pady=2)
        self.ax_bg_var = tk.StringVar(value="white")
        ax_bg_cb = ttk.Combobox(control_frame, textvariable=self.ax_bg_var, values=["white", "whitesmoke", "lightyellow", "aliceblue", "seashell"], state="readonly", width=12)
        ax_bg_cb.grid(row=2, column=3, sticky="w", padx=5, pady=2)

        ttk.Button(control_frame, text="🔄 Apply Changes", command=self.update_plot).grid(row=2, column=5, padx=5, pady=2, sticky="e")

        self.fig = Figure(figsize=(8, 4.0), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas.get_tk_widget().config(width=760, height=320)

        self.stats_legend_frame = ttk.LabelFrame(self, text="📊 Model Statistics & Legend", padding=10)
        self.stats_legend_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stats_sub_frame = ttk.Frame(self.stats_legend_frame)
        self.stats_sub_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.legend_sub_frame = ttk.Frame(self.stats_legend_frame)
        self.legend_sub_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        ttk.Label(self.stats_sub_frame, text="Model Statistics:", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 2))
        self.stats_text = scrolledtext.ScrolledText(self.stats_sub_frame, wrap=tk.WORD, font=("Consolas", 9), height=5, width=45)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.config(state=tk.DISABLED)

        ttk.Label(self.legend_sub_frame, text="Plot Legend:", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 2))
        self.legend_text = scrolledtext.ScrolledText(self.legend_sub_frame, wrap=tk.WORD, font=("Consolas", 9), height=5, width=45)
        self.legend_text.pack(fill=tk.BOTH, expand=True)
        self.legend_text.config(state=tk.DISABLED)

        bottom_frame = ttk.Frame(self, padding=5)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        toolbar = NavigationToolbar2Tk(self.canvas, bottom_frame)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)

        ttk.Button(bottom_frame, text="💾 Save High-Res Figure", command=self.save_figure).pack(side=tk.RIGHT, padx=5)

        self.update_plot()

    def set_stats_legend(self, stats_content, legend_content):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_content)
        self.stats_text.config(state=tk.DISABLED)

        self.legend_text.config(state=tk.NORMAL)
        self.legend_text.delete(1.0, tk.END)
        self.legend_text.insert(tk.END, legend_content)
        self.legend_text.config(state=tk.DISABLED)

    def update_plot(self):
        m_name = self.model_var.get()
        d_type = self.dataset_var.get()
        res = self.detailed_results.get(m_name)
        if not res:
            return

        fig_bg = self.fig_bg_var.get()
        ax_bg = self.ax_bg_var.get()
        self.fig.patch.set_facecolor(fig_bg)
        self.ax.set_facecolor(ax_bg)
        self.ax.clear()

        custom_title = self.title_entry.get()
        custom_xlabel = self.xlabel_entry.get() or "Original (Actual) Values"
        custom_ylabel = self.ylabel_entry.get() or "Predicted Values"

        if "Validation" in d_type:
            y_true = res['y_test']
            y_pred = res['y_hat_ext']
            color = "#1f77b4"
            # Always recompute from the actual points shown so numbers match the figure
            r2_val = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.std(y_true) > 0 else float("nan")
            rmse_val = float(math.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) > 0 else float("nan")
            ccc_val = concordance_correlation_coefficient(y_true, y_pred)
            rpd_val = calculate_rpd(y_true, rmse_val)
            set_label = "Validation (Test)"
            repeats_note = ""
            if res.get('n_repeats', 1) > 1:
                repeats_note = "\n(Note: points & stats from BEST random-split repeat;\n table shows averages across repeats.)"

            self.ax.scatter(y_true, y_pred, color=color, edgecolors='k', alpha=0.8, s=50, label=set_label)
            all_vals = np.concatenate([y_true, y_pred])
            min_v, max_v = np.min(all_vals), np.max(all_vals)
            margin = (max_v - min_v) * 0.08 if max_v != min_v else 1.0
            lims = [min_v - margin, max_v + margin]
            self.ax.plot(lims, lims, 'k--', lw=1.5, label="Ideal 1:1 Line")

            m, b = 0.0, 0.0
            if len(y_true) > 1:
                m, b = np.polyfit(y_true, y_pred, 1)
                x_line = np.array(lims)
                y_line = m * x_line + b
                self.ax.plot(x_line, y_line, color=color, lw=1.5, label=f"Fit: y = {m:.2f}x {b:+.2f}")

            self.ax.set_title(custom_title or f"{m_name} - {set_label}", fontsize=12, fontweight='bold')
            self.ax.set_xlim(lims)
            self.ax.set_ylim(lims)

            stats_text = (f"【 Model Statistics ({set_label}) 】\n"
                          f"• R² = {r2_val:.4f}  |  RMSE = {rmse_val:.4f}\n"
                          f"• CCC = {ccc_val:.4f}  |  RPD = {rpd_val:.3f}  |  n = {len(y_true)}"
                          f"{repeats_note}")
            legend_text = (f"【 Plot Legend 】\n"
                           f"• Blue Circles: Validation Samples\n"
                           f"• Black Dashed Line: Ideal 1:1 Line\n"
                           f"• Blue Solid Line: Regression Fit (y = {m:.2f}x {b:+.2f})")
            self.set_stats_legend(stats_text, legend_text)

        elif "Calibration" in d_type:
            y_true = res['y_train']
            y_pred = res['y_hat_train']
            color = "#ff7f0e"
            # Always recompute from the actual points shown so numbers match the figure
            r2_val = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.std(y_true) > 0 else float("nan")
            rmse_val = float(math.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) > 0 else float("nan")
            ccc_val = concordance_correlation_coefficient(y_true, y_pred)
            rpd_val = calculate_rpd(y_true, rmse_val)
            set_label = "Calibration (Train)"
            repeats_note = ""
            if res.get('n_repeats', 1) > 1:
                repeats_note = "\n(Note: points & stats from BEST random-split repeat;\n table shows averages across repeats.)"

            self.ax.scatter(y_true, y_pred, color=color, edgecolors='k', alpha=0.8, s=50, label=set_label)
            all_vals = np.concatenate([y_true, y_pred])
            min_v, max_v = np.min(all_vals), np.max(all_vals)
            margin = (max_v - min_v) * 0.08 if max_v != min_v else 1.0
            lims = [min_v - margin, max_v + margin]
            self.ax.plot(lims, lims, 'k--', lw=1.5, label="Ideal 1:1 Line")

            m, b = 0.0, 0.0
            if len(y_true) > 1:
                m, b = np.polyfit(y_true, y_pred, 1)
                x_line = np.array(lims)
                y_line = m * x_line + b
                self.ax.plot(x_line, y_line, color=color, lw=1.5, label=f"Fit: y = {m:.2f}x {b:+.2f}")

            self.ax.set_title(custom_title or f"{m_name} - {set_label}", fontsize=12, fontweight='bold')
            self.ax.set_xlim(lims)
            self.ax.set_ylim(lims)

            stats_text = (f"【 Model Statistics ({set_label}) 】\n"
                          f"• R² = {r2_val:.4f}  |  RMSE = {rmse_val:.4f}\n"
                          f"• CCC = {ccc_val:.4f}  |  RPD = {rpd_val:.3f}  |  n = {len(y_true)}"
                          f"{repeats_note}")
            legend_text = (f"【 Plot Legend 】\n"
                           f"• Orange Circles: Calibration Samples\n"
                           f"• Black Dashed Line: Ideal 1:1 Line\n"
                           f"• Orange Solid Line: Regression Fit (y = {m:.2f}x {b:+.2f})")
            self.set_stats_legend(stats_text, legend_text)

        elif "Pooled" in d_type or "All Points Pooled" in d_type:
            y_train = res['y_train']
            y_hat_train = res['y_hat_train']
            y_test = res['y_test']
            y_hat_ext = res['y_hat_ext']

            y_true_pooled = np.concatenate([y_train, y_test])
            y_pred_pooled = np.concatenate([y_hat_train, y_hat_ext])

            r2_val = float(r2_score(y_true_pooled, y_pred_pooled))
            rmse_val = math.sqrt(mean_squared_error(y_true_pooled, y_pred_pooled))
            ccc_val = concordance_correlation_coefficient(y_true_pooled, y_pred_pooled)
            rpd_val = calculate_rpd(y_true_pooled, rmse_val)

            self.ax.scatter(y_train, y_hat_train, color="#ff7f0e", edgecolors='k', alpha=0.8, s=50, label="Calibration")
            self.ax.scatter(y_test, y_hat_ext, color="#1f77b4", edgecolors='k', alpha=0.8, s=50, label="Validation")

            all_vals = np.concatenate([y_true_pooled, y_pred_pooled])
            min_v, max_v = np.min(all_vals), np.max(all_vals)
            margin = (max_v - min_v) * 0.08 if max_v != min_v else 1.0
            lims = [min_v - margin, max_v + margin]
            self.ax.plot(lims, lims, 'k--', lw=1.5, label="Ideal 1:1 Line")

            m, b = 0.0, 0.0
            if len(y_true_pooled) > 1:
                m, b = np.polyfit(y_true_pooled, y_pred_pooled, 1)
                x_line = np.array(lims)
                y_line = m * x_line + b
                self.ax.plot(x_line, y_line, color='purple', lw=1.5, label=f"Pooled Fit: y = {m:.2f}x {b:+.2f}")

            self.ax.set_title(custom_title or f"{m_name} - Pooled (n = {len(y_true_pooled)})", fontsize=12, fontweight='bold')
            self.ax.set_xlim(lims)
            self.ax.set_ylim(lims)
            
            repeats_note = ""
            if res.get('n_repeats', 1) > 1:
                repeats_note = "\n(Note: points from BEST random-split repeat;\n table shows averages across repeats.)"

            stats_text = (f"【 Pooled Statistics (Train + Test Combined) 】\n"
                          f"• Pooled R² = {r2_val:.4f}  |  Pooled RMSE = {rmse_val:.4f}\n"
                          f"• Pooled CCC = {ccc_val:.4f}  |  Pooled RPD = {rpd_val:.3f}  |  Total n = {len(y_true_pooled)}"
                          f"{repeats_note}")
            legend_text = (f"【 Plot Legend 】\n"
                           f"• Orange Circles: Calibration Points\n"
                           f"• Blue Circles: Validation Points\n"
                           f"• Purple Solid Line: Pooled Fit (y = {m:.2f}x {b:+.2f})")
            self.set_stats_legend(stats_text, legend_text)

        else:
            y_train = res['y_train']
            y_hat_train = res['y_hat_train']
            y_test = res['y_test']
            y_hat_ext = res['y_hat_ext']

            self.ax.scatter(y_train, y_hat_train, color="#ff7f0e", edgecolors='k', alpha=0.8, s=50, label="Calibration (Train)")
            self.ax.scatter(y_test, y_hat_ext, color="#1f77b4", edgecolors='k', alpha=0.8, s=50, label="Validation (Test)")

            all_vals = np.concatenate([y_train, y_hat_train, y_test, y_hat_ext])
            min_v, max_v = np.min(all_vals), np.max(all_vals)
            margin = (max_v - min_v) * 0.08 if max_v != min_v else 1.0
            lims = [min_v - margin, max_v + margin]
            self.ax.plot(lims, lims, 'k--', lw=1.5, label="Ideal 1:1 Line")

            x_line = np.array(lims)
            m_tr, b_tr = (0.0, 0.0)
            m_te, b_te = (0.0, 0.0)

            if len(y_train) > 1:
                m_tr, b_tr = np.polyfit(y_train, y_hat_train, 1)
                self.ax.plot(x_line, m_tr * x_line + b_tr, color="#ff7f0e", lw=1.5, linestyle='-', label=f"Cal Fit: y={m_tr:.2f}x{b_tr:+.2f}")

            if len(y_test) > 1:
                m_te, b_te = np.polyfit(y_test, y_hat_ext, 1)
                self.ax.plot(x_line, m_te * x_line + b_te, color="#1f77b4", lw=1.5, linestyle='-', label=f"Val Fit: y={m_te:.2f}x{b_te:+.2f}")

            self.ax.set_title(custom_title or f"{m_name} - Combined Calibration & Validation", fontsize=12, fontweight='bold')
            self.ax.set_xlim(lims)
            self.ax.set_ylim(lims)

            # Recompute from the actual points shown
            r2_cal = float(r2_score(y_train, y_hat_train)) if len(y_train) > 1 and np.std(y_train) > 0 else float("nan")
            rmse_cal = float(math.sqrt(mean_squared_error(y_train, y_hat_train))) if len(y_train) > 0 else float("nan")
            r2_val = float(r2_score(y_test, y_hat_ext)) if len(y_test) > 1 and np.std(y_test) > 0 else float("nan")
            rmse_val = float(math.sqrt(mean_squared_error(y_test, y_hat_ext))) if len(y_test) > 0 else float("nan")
            rpd_val = calculate_rpd(y_test, rmse_val)
            repeats_note = ""
            if res.get('n_repeats', 1) > 1:
                repeats_note = "\n(Note: points & stats from BEST random-split repeat;\n table shows averages across repeats.)"

            stats_text = (f"【 Combined Series Statistics 】\n"
                          f"• Calibration: R² = {r2_cal:.4f} | RMSE = {rmse_cal:.4f} (n = {len(y_train)})\n"
                          f"• Validation:  R² = {r2_val:.4f}  | RMSE = {rmse_val:.4f} | RPD = {rpd_val:.3f} (n = {len(y_test)})"
                          f"{repeats_note}")
            legend_text = (f"【 Plot Legend 】\n"
                           f"• Orange Circles: Calibration (Train)\n"
                           f"• Blue Circles: Validation (Test)\n"
                           f"• Solid Lines: Cal Fit & Val Fit")
            self.set_stats_legend(stats_text, legend_text)

        self.ax.set_xlabel(custom_xlabel, fontsize=10)
        self.ax.set_ylabel(custom_ylabel, fontsize=10)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_figure(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Vector", "*.svg"), ("JPEG Image", "*.jpg")],
            title="Save Correlation Plot"
        )
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Figure saved successfully to:\n{file_path}", parent=self)
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=self)

# ==============================================================================
# MAIN DESKTOP GUI APPLICATION
# ==============================================================================

class SpectralAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Analyzer Pro")
        self.root.geometry("1350x850")

        self.setup_data = None
        self.detailed_results = {}
        self.wavelengths = None  # stored wavelength vector (optional)
        self.session_log = []    # list of (timestamp, message) for reproducibility

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_lbl = ttk.Label(main_frame, text="Spectral Analyzer Pro",
                              font=("Arial", 16, "bold"))
        title_lbl.pack(anchor="w", pady=5)

        control_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuration & Execution", padding=10)
        control_frame.pack(fill=tk.X, pady=5)

        # Row 1: Load + preprocessing + feature selection
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=(0, 4))

        ttk.Button(row1, text="📂 Load Data & Setup", command=self.load_wizard).pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Preprocessing:").pack(side=tk.LEFT, padx=(15, 2))
        self.prep_var = tk.StringVar(value="Standard Normal Variate (SNV)")
        prep_cb = ttk.Combobox(row1, textvariable=self.prep_var, values=[
            "Raw", "Standard Normal Variate (SNV)", "Multiplicative Scatter Correction (MSC)", 
            "Savitzky-Golay Filter", "1st Derivative (Savitzky-Golay)", "2nd Derivative (Savitzky-Golay)",
            "SNV + 1st Derivative", "Baseline Detrending", "Mean Centering"
        ], width=28, state="readonly")
        prep_cb.pack(side=tk.LEFT, padx=5)
        prep_cb.bind("<<ComboboxSelected>>", lambda e: (self._toggle_sg_controls(), self._log(f"Preprocessing changed to: {self.prep_var.get()}")))

        ttk.Label(row1, text="Feature selection:").pack(side=tk.LEFT, padx=(15, 2))
        self.feat_sel_var = tk.StringVar(value="None")
        feat_cb = ttk.Combobox(row1, textvariable=self.feat_sel_var, values=[
            "None", "VIP (PLS)", "Correlation", "Interval (iPLS-style)"
        ], width=16, state="readonly")
        feat_cb.pack(side=tk.LEFT, padx=5)
        feat_cb.bind("<<ComboboxSelected>>", lambda e: self._log(f"Feature selection changed to: {self.feat_sel_var.get()}"))

        self.full_nested_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Full nested CV", 
                        variable=self.full_nested_var,
                        command=lambda: self._log(f"Full nested CV set to: {self.full_nested_var.get()}")
                        ).pack(side=tk.LEFT, padx=(10, 2))

        # Row 2: Main actions
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=(2, 0))

        ttk.Button(row2, text="🚀 Run Models", command=self.run_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="📥 Export Excel Report", command=self.export_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="💾 Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="🔮 Predict New", command=self.predict_new_samples).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="📋 Export Importances", command=self.export_importances).pack(side=tk.LEFT, padx=5)

        # Second row for Savitzky-Golay controls so they are always easy to find
        sg_row = ttk.Frame(main_frame)
        sg_row.pack(fill=tk.X, pady=(0, 5))

        self.sg_frame = ttk.LabelFrame(sg_row, text="Savitzky-Golay settings", padding=6)
        self.sg_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.sg_frame, text="Window length:").pack(side=tk.LEFT, padx=(4, 2))
        self.sg_window_var = tk.StringVar(value="15")
        self.sg_window_spin = ttk.Spinbox(self.sg_frame, from_=5, to=51, increment=2,
                                          textvariable=self.sg_window_var, width=6)
        self.sg_window_spin.pack(side=tk.LEFT, padx=2)

        ttk.Label(self.sg_frame, text="Polynomial degree:").pack(side=tk.LEFT, padx=(10, 2))
        self.sg_poly_var = tk.StringVar(value="2")
        self.sg_poly_spin = ttk.Spinbox(self.sg_frame, from_=1, to=5,
                                        textvariable=self.sg_poly_var, width=5)
        self.sg_poly_spin.pack(side=tk.LEFT, padx=2)

        ttk.Label(self.sg_frame, text="(active only when a Savitzky-Golay method is selected)",
                  foreground="gray").pack(side=tk.LEFT, padx=8)

        # Keep the toggle so the values are ignored for non-SG methods
        self._toggle_sg_controls()

        # Dedicated figures row — always visible
        figures_row = ttk.LabelFrame(main_frame, text="📈 Figures", padding=6)
        figures_row.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(figures_row, text="📈 Predicted vs Actual", command=self.open_plotter).pack(side=tk.LEFT, padx=5)
        ttk.Button(figures_row, text="🧬 SHAP Figure", command=self.generate_shap_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(figures_row, text="📊 Spectral Overview (Raw + Preprocessed)", command=self.open_spectral_overview).pack(side=tk.LEFT, padx=5)
        ttk.Button(figures_row, text="📉 Residual Plot", command=self.open_residual_plot).pack(side=tk.LEFT, padx=5)

        # Tools row
        tools_row = ttk.LabelFrame(main_frame, text="🛠️ Tools & Export", padding=6)
        tools_row.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(tools_row, text="📏 Load Wavelengths", command=self.load_wavelengths).pack(side=tk.LEFT, padx=5)
        ttk.Button(tools_row, text="📋 Copy Methods", command=self.copy_methods_clipboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools_row, text="📋 Session Log", command=self.show_session_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(tools_row, text="🔀 Y-Scramble Test", command=self.run_y_scramble).pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Tab 1: Results Table
        self.tab_results = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_results, text="📊 Model Results & Validation")

        self.tree = ttk.Treeview(
            self.tab_results,
            columns=('Model', 'R2_Train', 'R2_Adj', 'RMSEC', 'Q2_CV', 'R2_Ext', 'CCC_Ext', 'RMSEP', 'RPD', 'Status'),
            show='headings')
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor=tk.CENTER)
        self.tree.column('Model', width=160, anchor=tk.W)

        tree_scroll = ttk.Scrollbar(self.tab_results, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: Regression Equations
        self.tab_equations = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_equations, text="📝 Regression Equations & Coefficients")

        self.eq_text = scrolledtext.ScrolledText(self.tab_equations, wrap=tk.WORD, font=("Consolas", 10))
        self.eq_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.eq_text.insert(tk.END, "Run models to inspect complete regression equations, n, s, F, feature importances, and diagnostics here...")
        self.eq_text.config(state=tk.DISABLED)

        # Tab 3: Statistics Glossary & Guide
        self.tab_glossary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_glossary, text="📖 Statistics Glossary & Guide")

        glossary_text = scrolledtext.ScrolledText(self.tab_glossary, wrap=tk.WORD, font=("Arial", 10))
        glossary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        glossary_lines = [
            "================================================================================",
            "                    CHEMOMETRIC & QSAR STATISTICAL GLOSSARY & GUIDE",
            "================================================================================",
            "",
            "This guide explains every statistic reported by Spectral Analyzer Pro,",
            "how it is calculated, and how to interpret it. All calculations are",
            "leak-free: scaling, collinearity elimination, Mean Centering and MSC",
            "are performed using only the training data of each split or CV fold.",
            "",
            "--------------------------------------------------------------------------------",
            "1. R² Calibration (R²_Train / R²_Cal)",
            "--------------------------------------------------------------------------------",
            "Proportion of variance in the training (calibration) set explained by the model.",
            "R² = 1 − SS_res / SS_tot",
            "where SS_res = Σ(y − ŷ)² and SS_tot = Σ(y − ȳ)².",
            "Range: −∞ to 1. Values close to 1 indicate good fit on the data used for training.",
            "Note: R² on training data is optimistic; always examine external validation.",
            "",
            "--------------------------------------------------------------------------------",
            "2. R² Validation / External (R²_Ext / R²_Pred)",
            "--------------------------------------------------------------------------------",
            "R² calculated on the held-out test set (or external set). This is the primary",
            "measure of predictive accuracy on unseen samples.",
            "Interpretation guidelines (approximate, context-dependent):",
            "  • R² > 0.9     Excellent",
            "  • 0.8 – 0.9    Good",
            "  • 0.6 – 0.8    Acceptable for many chemometric applications",
            "  • < 0.6        Poor predictive ability",
            "",
            "--------------------------------------------------------------------------------",
            "3. Adjusted R² (R²_Adj)",
            "--------------------------------------------------------------------------------",
            "Penalizes R² for the number of predictors (degrees of freedom).",
            "R²_Adj = 1 − (1 − R²) × (n − 1) / (n − p − 1)",
            "where n = number of samples, p = number of active variables (or components for PLS/PCR).",
            "Only statistically meaningful for linear models. For non-linear models the app",
            "reports N/A. Prefer this over raw R² when comparing models with different numbers",
            "of variables.",
            "",
            "--------------------------------------------------------------------------------",
            "4. RMSEC and RMSEP",
            "--------------------------------------------------------------------------------",
            "Root Mean Square Error of Calibration (RMSEC) and of Prediction (RMSEP).",
            "RMSE = √[ Σ(y − ŷ)² / n ]",
            "Expressed in the same units as the target property y.",
            "Lower is better. RMSEP is the more important figure for real-world performance.",
            "A large gap between RMSEC and RMSEP often indicates overfitting.",
            "",
            "--------------------------------------------------------------------------------",
            "5. Fisher F-Statistic (F)",
            "--------------------------------------------------------------------------------",
            "Overall significance of a linear model:",
            "F = (R² / p) / ((1 − R²) / (n − p − 1))",
            "Higher F indicates the model explains more variance than expected by chance.",
            "Reported only for linear models (MLR, Ridge, Lasso, ElasticNet, PLSR, PCR, Stepwise).",
            "N/A for non-linear models.",
            "",
            "--------------------------------------------------------------------------------",
            "6. Q² (Cross-Validated R²)",
            "--------------------------------------------------------------------------------",
            "Internal predictive ability estimated by K-fold cross-validation (default 5-fold).",
            "Scaling and (for Stepwise) variable selection are performed inside each fold",
            "to prevent information leakage. For tree ensembles a light nested",
            "hyperparameter selection is performed inside each outer fold on that",
            "fold’s training portion only (removes optimism; grid kept tiny for speed).",
            "Q² is computed on the training set of each split and then averaged across",
            "random repeats when applicable.",
            "Q² substantially lower than R²_Train is a classic sign of overfitting.",
            "",
            "--------------------------------------------------------------------------------",
            "7. Lin’s Concordance Correlation Coefficient (CCC)",
            "--------------------------------------------------------------------------------",
            "Measures both precision and accuracy (agreement with the 1:1 line).",
            "CCC = (2 ρ σ_y σ_ŷ) / (σ_y² + σ_ŷ² + (μ_y − μ_ŷ)²)",
            "where ρ is the Pearson correlation, σ are standard deviations and μ are means",
            "(population versions, ddof=0).",
            "CCC = 1 only for perfect agreement. More stringent than R² alone because it",
            "penalizes bias and scale differences. Reported on the external/validation set.",
            "",
            "--------------------------------------------------------------------------------",
            "8. Residual Predictive Deviation (RPD)",
            "--------------------------------------------------------------------------------",
            "RPD = SD(y) / RMSEP   (sample standard deviation, ddof=1)",
            "Widely used in NIR/chemometrics literature:",
            "  • RPD < 1.5     Poor, not useful for prediction",
            "  • 1.5 – 2.0     Rough screening",
            "  • 2.0 – 2.5     Approximate quantitative prediction",
            "  • 2.5 – 3.0     Good",
            "  • > 3.0         Excellent",
            "Always interpret together with R² and RMSEP.",
            "",
            "--------------------------------------------------------------------------------",
            "9. Golbraikh–Tropsha External Validation Criteria",
            "--------------------------------------------------------------------------------",
            "Strict QSAR/chemometric acceptance criteria applied to the external set",
            "(computed on the best random-split repeat when repeats > 1).",
            "The Pass/Fail decision uses the classic absolute-difference form",
            "(most commonly cited in the literature):",
            "  • R² > 0.6",
            "  • |R² − R₀²| < 0.1          (absolute)",
            "  • |R² − R₀′²| < 0.1         (absolute)",
            "  • 0.85 ≤ k ≤ 1.15",
            "  • 0.85 ≤ k′ ≤ 1.15",
            "",
            "Relative differences |R² − R₀²|/R² are also reported for reference.",
            "",
            "where:",
            "  k  = Σ(y · ŷ) / Σ(ŷ²)          (slope of y = k·ŷ forced through origin)",
            "  R₀² = 1 − Σ(y − k·ŷ)² / Σ(y − ȳ)²",
            "  k′ = Σ(y · ŷ) / Σ(y²)           (slope of ŷ = k′·y forced through origin)",
            "  R₀′² = 1 − Σ(ŷ − k′·y)² / Σ(ŷ − ȳ̂)²",
            "",
            "A model is marked “Passed” only when ALL absolute-difference conditions",
            "are satisfied. Failure on any criterion indicates the model may not be",
            "reliable for external prediction even if R² looks acceptable.",
            "",
            "--------------------------------------------------------------------------------",
            "10. Preprocessing Notes",
            "--------------------------------------------------------------------------------",
            "• SNV, Savitzky–Golay (derivatives), Baseline Detrending: row-wise → no leakage.",
            "• Mean Centering and MSC: reference spectrum/mean estimated on training data only",
            "  (or on the full calibration set when an external test set is supplied).",
            "• Collinearity elimination (pairwise |r| > threshold): performed on the training",
            "  block only in internal-split mode.",
            "",
            "--------------------------------------------------------------------------------",
            "11. Random Split Repeats",
            "--------------------------------------------------------------------------------",
            "When “Random Split” with N > 1 is selected:",
            "• The results table and Excel “*_Avg” columns show the mean of the N repeats.",
            "• Equations, coefficients, stored predictions, Golbraikh–Tropsha status and",
            "  all plot statistics come from the single best repeat (highest external R²).",
            "• Plot windows recompute R²/RMSE/CCC/RPD from the exact points displayed and",
            "  add a note when N > 1.",
            "",
            "--------------------------------------------------------------------------------",
            "12. Model-Specific Remarks",
            "--------------------------------------------------------------------------------",
            "• PLSR / PCR: number of latent variables / components used as p for adjusted R² and F.",
            "• Regularized linear models (Ridge, Lasso, ElasticNet): p = number of non-zero coefficients.",
            "• Stepwise MLR: forward-backward selection with p-enter / p-remove thresholds;",
            "  performed inside CV and on the training split.",
            "• Tree ensembles, SVR, kNN, GPR, MLP: adjusted R² and F are not applicable (shown as N/A).",
            "  Feature importances are reported instead of regression coefficients.",
            "• Tree-model Q² uses light nested hyperparameter selection inside each outer fold",
            "  (parameters chosen only on that fold’s training data) to avoid optimism.",
            "",
            "--------------------------------------------------------------------------------",
            "13. Feature / Wavelength Selection",
            "--------------------------------------------------------------------------------",
            "Optional methods applied on the training data only (no leakage):",
            "• VIP (PLS): Variable Importance in Projection from a quick PLS model.",
            "• Correlation: features with highest |correlation| to the target.",
            "• Interval (iPLS-style): evaluates consecutive wavelength blocks with PLS",
            "  and retains the best-performing intervals.",
            "Selected features are used for both training and the corresponding test/external set.",
            "",
            "================================================================================",
            "Always prefer external R², RMSEP, CCC, RPD and Golbraikh–Tropsha status over",
            "training-set statistics when judging a model’s real predictive value.",
            "================================================================================"
        ]
        glossary_text.insert(tk.END, "\n".join(glossary_lines))
        glossary_text.config(state=tk.DISABLED)

        # Tab 4: Program Description & Developer (can be left by switching tabs)
        self.tab_about = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_about, text="ℹ️ About / Program Info")

        about_text = scrolledtext.ScrolledText(self.tab_about, wrap=tk.WORD, font=("Arial", 10))
        about_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        about_lines = [
            "================================================================================",
            "                         SPECTRAL ANALYZER PRO",
            "================================================================================",
            "",
            "PROGRAM DESCRIPTION",
            "--------------------------------------------------------------------------------",
            "Spectral Analyzer Pro is a desktop application for chemometrics and QSAR.",
            "",
            "It allows you to:",
            "• Load spectral matrices (X) and target properties (y) from CSV or Excel.",
            "• Apply common preprocessing methods: SNV, MSC, Savitzky-Golay (smoothing",
            "  and derivatives), baseline detrending, and mean centering.",
            "• Reduce collinearity (pairwise correlation + VIF).",
            "• Optionally select wavelengths/features with VIP, correlation, or",
            "  interval (iPLS-style) methods.",
            "• Train and compare many models: PLSR, PCR, Ridge, Lasso, ElasticNet,",
            "  SVR, Random Forest, Extra Trees, Gradient Boosting, Gaussian Process,",
            "  kNN, MLP, and Stepwise MLR.",
            "• Perform leak-free cross-validation (Q²) and external validation.",
            "• Evaluate models with R², RMSE, CCC, RPD, Adjusted R², F-statistic,",
            "  and Golbraikh–Tropsha criteria.",
            "• Visualize predicted vs actual, residuals, SHAP explanations, and",
            "  raw + preprocessed spectra.",
            "• Export a full Excel report and save high-resolution figures.",
            "",
            "All calculations respect proper training/test boundaries to avoid",
            "information leakage.",
            "",
            "================================================================================",
            "DEVELOPER",
            "--------------------------------------------------------------------------------",
            "Developed by Dr. Abdelhalim Khaled",
            "Plant Ecophysiology",
            "University of Concepción",
            "",
            "================================================================================"
        ]
        about_text.insert(tk.END, "\n".join(about_lines))
        about_text.config(state=tk.DISABLED)

        self.status_var = tk.StringVar(value="Ready. Click 'Load Data & Setup' to begin.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _toggle_sg_controls(self):
        """Enable or disable the Savitzky-Golay spinboxes.
        The frame itself stays visible so the user can always find it.
        """
        method = self.prep_var.get()
        sg_methods = {
            "Savitzky-Golay Filter",
            "1st Derivative (Savitzky-Golay)",
            "2nd Derivative (Savitzky-Golay)",
            "SNV + 1st Derivative",
        }
        state = tk.NORMAL if method in sg_methods else tk.DISABLED
        try:
            self.sg_window_spin.config(state=state)
            self.sg_poly_spin.config(state=state)
        except Exception:
            pass

    def _get_sg_params(self):
        """Return validated (window, poly) for Savitzky-Golay, or defaults."""
        try:
            window = int(self.sg_window_var.get())
            poly = int(self.sg_poly_var.get())
        except (ValueError, TypeError):
            window, poly = 15, 2
        # Force odd window and basic sanity
        if window % 2 == 0:
            window += 1
        window = max(5, min(window, 51))
        poly = max(1, min(poly, 5))
        if poly >= window:
            poly = max(1, window - 2)
        return window, poly

    def _log(self, message):
        """Append a timestamped entry to the session log."""
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_log.append(f"[{ts}] {message}")

    def load_wavelengths(self):
        """Load a wavelength vector once and store it for reuse."""
        path = filedialog.askopenfilename(
            title="Select wavelength file (one column of numeric values)",
            filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls")]
        )
        if not path:
            return
        try:
            if path.lower().endswith('.csv'):
                df = pd.read_csv(path, header=None)
            else:
                df = pd.read_excel(path, header=None)
            wl = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values.astype(float)
            if len(wl) < 2:
                messagebox.showerror("Error", "Wavelength file must contain at least 2 numeric values.")
                return
            self.wavelengths = wl
            self._log(f"Loaded wavelength vector ({len(wl)} points) from {os.path.basename(path)}")
            messagebox.showinfo("Success", f"Wavelengths loaded: {len(wl)} points.\nThey will be used in Spectral Overview and related plots.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load wavelengths:\n{e}")

    def export_methods_text(self):
        """Export a plain-text Methods paragraph ready for a manuscript."""
        text = self._build_methods_text()
        if text is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            self._log(f"Exported Methods text to {os.path.basename(path)}")
            messagebox.showinfo("Success", f"Methods text saved to:\n{path}\n\nYou can paste it directly into a manuscript.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def copy_methods_clipboard(self):
        """Copy the Methods text to the clipboard."""
        text = self._build_methods_text()
        if text is None:
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
            self._log("Copied Methods text to clipboard")
            messagebox.showinfo("Success", "Methods text copied to clipboard.\nYou can paste it directly into a manuscript.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy to clipboard:\n{e}")

    def _build_methods_text(self):
        """Build the plain-text Methods content. Returns None if no data."""
        if not self.detailed_results and not self.setup_data:
            messagebox.showwarning("Warning", "Please load data and run models first.")
            return None
        prep = self.prep_var.get()
        feat_sel = self.feat_sel_var.get()
        full_nested = self.full_nested_var.get()
        sg_w, sg_p = self._get_sg_params()
        val_mode = self.setup_data.get('val_mode', 'N/A') if self.setup_data else 'N/A'
        split_algo = self.setup_data.get('split_algo', 'N/A') if self.setup_data else 'N/A'
        test_size = self.setup_data.get('test_size', 'N/A') if self.setup_data else 'N/A'
        n_repeats = self.setup_data.get('random_split_repeats', 1) if self.setup_data else 1
        collinear = self.setup_data.get('collinear_reduction', False) if self.setup_data else False
        collinear_thr = self.setup_data.get('collinear_threshold', 'N/A') if self.setup_data else 'N/A'

        lines = [
            "METHODS (Spectral Analyzer Pro)",
            "=" * 60,
            "",
            f"Preprocessing: {prep}",
            f"Savitzky-Golay window / polynomial: {sg_w} / {sg_p}" if "Savitzky" in prep or "Derivative" in prep else "Savitzky-Golay: not used",
            f"Feature selection: {feat_sel}",
            f"Collinearity reduction: {'Yes (' + str(collinear_thr) + '%)' if collinear else 'No'}",
            f"Validation mode: {val_mode}",
            f"Split algorithm: {split_algo}",
            f"Test-set proportion: {test_size}",
            f"Random-split repeats: {n_repeats}",
            f"Cross-validation: {'Full nested' if full_nested else 'Light nested'} K-fold Q² (scaling inside folds)",
            "Golbraikh-Tropsha criteria: absolute form |R² − R₀²| < 0.1 and |R² − R₀′²| < 0.1",
            "",
            "Models and selected hyperparameters (best repeat):",
        ]
        for name, res in self.detailed_results.items():
            lines.append(f"  - {name}:")
            lines.append(f"      alpha/C/length = {res.get('chosen_alpha')}")
            lines.append(f"      n_components/k = {res.get('chosen_n_components')}")
            lines.append(f"      tree params (depth, leaf, n_est) = {res.get('chosen_tree')}")
            lines.append(f"      active variables = {res.get('p_vars')}")
        lines.append("")
        lines.append("All preprocessing references, collinearity elimination, feature selection,")
        lines.append("and scaling were estimated on training data only (no information leakage).")
        lines.append("")
        lines.append("Generated by Spectral Analyzer Pro")
        return "\n".join(lines)

    def show_session_log(self):
        """Display the session log in a window and allow saving it."""
        win = tk.Toplevel(self.root)
        win.title("Session Log")
        win.geometry("700x400")
        txt = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Consolas", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        if not self.session_log:
            txt.insert(tk.END, "(No actions recorded yet.)\n")
        else:
            txt.insert(tk.END, "\n".join(self.session_log))
        txt.config(state=tk.DISABLED)

        def save_log():
            fp = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")], parent=win)
            if fp:
                try:
                    with open(fp, 'w', encoding='utf-8') as f:
                        f.write("\n".join(self.session_log))
                    messagebox.showinfo("Success", f"Session log saved to:\n{fp}", parent=win)
                except Exception as e:
                    messagebox.showerror("Error", str(e), parent=win)

        ttk.Button(win, text="💾 Save Log", command=save_log).pack(pady=5)

    def load_wizard(self):
        wizard = DataSetupWizardDialog(self.root)
        self.root.wait_window(wizard)
        if wizard.result:
            self.setup_data = wizard.result
            self._log(f"Data loaded: {self.setup_data['df_X_cal'].shape[0]} calibration samples, mode={self.setup_data['val_mode']}")
            mode = self.setup_data['val_mode']
            collin_status = f" | Collinear Reduction: {self.setup_data['collinear_threshold']}%" if self.setup_data['collinear_reduction'] else ""
            if mode == "split":
                rep_info = f", Repeats: {self.setup_data['random_split_repeats']}" if self.setup_data['split_algo'] == "Random Split" else ""
                msg = f"Calibration loaded: {self.setup_data['df_X_cal'].shape[0]} samples. Split: {self.setup_data['split_algo']} ({int(self.setup_data['test_size']*100)}% test{rep_info}).{collin_status}"
            else:
                msg = f"Calibration loaded: {self.setup_data['df_X_cal'].shape[0]} samples. Separate Test loaded: {self.setup_data['df_X_ext'].shape[0]} samples.{collin_status}"
            self.status_var.set(msg)
            messagebox.showinfo("Success", "Data and validation setup completed successfully!")

    def _show_model_selection_dialog(self, n_samples, n_train_est):
        """Shows a dialog to select models with sample-size based recommendations."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Models to Run")
        dialog.geometry("620x680")
        dialog.transient(self.root)
        dialog.grab_set()

        result = {"models": None}

        main = ttk.Frame(dialog, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Sample size info + advice
        advice_frame = ttk.LabelFrame(main, text="Sample Size Advice", padding=8)
        advice_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(advice_frame, text=f"Calibration samples (n): {n_samples}", font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(advice_frame, text=f"Estimated training samples: ~{n_train_est}", font=("Arial", 9)).pack(anchor="w")

        # Recommendation text based on n
        if n_train_est < 30:
            rec = ("⚠️ Very small sample size.\n"
                   "Recommended: PLSR, PCR, Ridge, ElasticNet.\n"
                   "Stepwise MLR is available but ElasticNet is preferred.\n"
                   "Avoid: Random Forest, Extra Trees, Gradient Boosting, MLP, Gaussian Process.")
            color = "darkred"
        elif n_train_est < 60:
            rec = ("⚠️ Small sample size.\n"
                   "Recommended: PLSR, PCR, Ridge, Lasso, ElasticNet, Stepwise MLR.\n"
                   "Use with caution: SVR, kNN.\n"
                   "Avoid or limit: Tree ensembles and MLP.")
            color = "darkorange"
        elif n_train_est < 150:
            rec = ("✅ Moderate sample size.\n"
                   "Most models are reasonable. Prefer PLSR / PCR / regularized linear models first.\n"
                   "Tree models and SVR can be tried. MLP still risky if features >> samples.")
            color = "darkgreen"
        else:
            rec = ("✅ Good sample size.\n"
                   "All models can be considered. Ensemble methods and MLP become more reliable.\n"
                   "Still useful to compare against PLSR / PCR as baseline.")
            color = "darkgreen"

        ttk.Label(advice_frame, text=rec, foreground=color, justify="left", wraplength=560).pack(anchor="w", pady=4)

        # Model selection
        sel_frame = ttk.LabelFrame(main, text="Select Models", padding=8)
        sel_frame.pack(fill=tk.BOTH, expand=True)

        all_models = [
            "PLSR (5 LV max)", "PCR (5 PC + OLS)", "Ridge", "Lasso", "ElasticNet",
            "SVR (RBF)", "Random Forest", "Extra Trees", "Gradient Boosting",
            "Gaussian Process (GPR)", "kNN Regressor (k=5)", "MLP / ANN", "Stepwise MLR"
        ]

        # Pre-select recommended ones
        recommended = set()
        if n_train_est < 30:
            recommended = {"PLSR (5 LV max)", "PCR (5 PC + OLS)", "Ridge", "Stepwise MLR"}
        elif n_train_est < 60:
            recommended = {"PLSR (5 LV max)", "PCR (5 PC + OLS)", "Ridge", "Lasso", "ElasticNet", "Stepwise MLR"}
        else:
            recommended = set(all_models)  # all ok, but user can deselect

        vars_dict = {}
        for m in all_models:
            var = tk.BooleanVar(value=(m in recommended))
            vars_dict[m] = var
            cb = ttk.Checkbutton(sel_frame, text=m, variable=var)
            cb.pack(anchor="w", pady=1)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)

        def select_all():
            for v in vars_dict.values():
                v.set(True)

        def select_recommended():
            for m, v in vars_dict.items():
                v.set(m in recommended)

        def on_ok():
            chosen = [m for m, v in vars_dict.items() if v.get()]
            if not chosen:
                messagebox.showwarning("Warning", "Please select at least one model.", parent=dialog)
                return
            result["models"] = chosen
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(btn_frame, text="Select All", command=select_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Select Recommended", command=select_recommended).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Run Selected Models", command=on_ok).pack(side=tk.RIGHT, padx=4)

        dialog.wait_window()
        return result["models"]

    def run_models(self):
        if not self.setup_data:
            messagebox.showwarning("Warning", "Please load data and set up validation first!")
            return

        # Estimate sample size for recommendations
        n_samples = len(self.setup_data['df_X_cal'])
        if self.setup_data['val_mode'] == "split":
            n_train_est = int(n_samples * (1 - self.setup_data['test_size']))
        else:
            n_train_est = n_samples

        # Show model selection dialog with sample-size advice
        selected_models = self._show_model_selection_dialog(n_samples, n_train_est)
        if not selected_models:
            return  # User cancelled

        self.status_var.set("Preprocessing data, eliminating collinearity, and training models...")
        self.root.update_idletasks()

        try:
            df_X_cal = self.setup_data['df_X_cal']
            df_y_cal = self.setup_data['df_y_cal']
            val_mode = self.setup_data['val_mode']
            do_collinear = self.setup_data['collinear_reduction']
            collinear_thresh = self.setup_data['collinear_threshold']

            if len(df_X_cal) != len(df_y_cal):
                raise ValueError(f"Sample count mismatch between X ({len(df_X_cal)}) and y ({len(df_y_cal)}).")
            if len(df_X_cal) < 3:
                raise ValueError("At least 3 complete calibration samples are required.")

            X_cal_df = df_X_cal.reset_index(drop=True)
            y_cal_df = df_y_cal.reset_index(drop=True)

            prep_method = self.prep_var.get()
            sg_window, sg_poly = self._get_sg_params()
            # Column-wise references must be estimated on training data only.
            if prep_method in ("Mean Centering", "Multiplicative Scatter Correction (MSC)") and val_mode == "split":
                X_cal_proc_df = apply_preprocessing(X_cal_df, method="Raw", sg_window=sg_window, sg_poly=sg_poly)
            else:
                X_cal_proc_df = apply_preprocessing(X_cal_df, method=prep_method, sg_window=sg_window, sg_poly=sg_poly)

            if do_collinear and val_mode != "split":
                # External mode: the calibration set is the training set, so reduction is valid.
                X_cal_reduced_df, dropped_cols = eliminate_collinearity(X_cal_proc_df, threshold_pct=collinear_thresh, y=y_cal_df.iloc[:, 0].values)
            else:
                # Internal split: keep every feature here and drop collinear ones
                # inside each training fold so the test samples cannot influence selection.
                X_cal_reduced_df = X_cal_proc_df

            X_cal_all = X_cal_reduced_df.values
            y_cal_all = y_cal_df.iloc[:, 0].values.astype(float)
            cal_sample_ids = np.arange(len(X_cal_df))

            if val_mode == "split":
                test_size = self.setup_data['test_size']
                n_repeats = self.setup_data.get('random_split_repeats', 1) if self.setup_data.get('split_algo') == "Random Split" else 1
            else:
                n_repeats = 1
                df_X_ext = self.setup_data['df_X_ext']
                df_y_ext = self.setup_data['df_y_ext']
                df_X_ext = df_X_ext.reset_index(drop=True)
                df_y_ext = df_y_ext.reset_index(drop=True)

                common_features = [col for col in X_cal_reduced_df.columns if col in df_X_ext.columns]
                if not common_features:
                    raise ValueError("No common features found between calibration and external test sets.")
                    
                X_ext_df = df_X_ext[common_features]
                if prep_method == "Mean Centering":
                    # Use the calibration mean, not the external mean.
                    mean_cal = X_cal_df[common_features].values.mean(axis=0)
                    X_ext_proc = pd.DataFrame(X_ext_df.values - mean_cal, columns=common_features, index=X_ext_df.index)
                elif prep_method == "Multiplicative Scatter Correction (MSC)":
                    # MSC reference is the calibration mean spectrum, on the same wavelengths.
                    cal_cols = list(X_cal_df.columns)
                    missing = [c for c in cal_cols if c not in df_X_ext.columns]
                    if missing:
                        raise ValueError("External file is missing calibration wavelengths required for MSC.")
                    ref = X_cal_df[cal_cols].values.mean(axis=0)
                    ext_msc = apply_msc(df_X_ext[cal_cols].values.astype(float), ref)
                    X_ext_proc = pd.DataFrame(ext_msc, columns=cal_cols).loc[:, common_features]
                else:
                    X_ext_proc = apply_preprocessing(X_ext_df, method=prep_method, sg_window=sg_window, sg_poly=sg_poly)
                X_test_fixed = X_ext_proc[common_features].values
                y_test_fixed = df_y_ext.iloc[:, 0].values.astype(float)
                test_sample_ids_fixed = df_X_ext.index.values

            model_keys = selected_models

            for item in self.tree.get_children():
                self.tree.delete(item)

            self.detailed_results = {}
            rows = []
            equations_report = []

            for idx_m, name in enumerate(model_keys):
                self.status_var.set(f"Training model {idx_m+1}/{len(model_keys)}: {name} ...")
                self.root.update_idletasks()
                acc_r2_train, acc_r2_adj, acc_rmsec, acc_q2_cv, acc_r2_ext, acc_ccc_ext, acc_rmsep, acc_rpd_ext = [], [], [], [], [], [], [], []
                
                last_y_train, last_y_test, last_y_hat_train, last_y_hat_ext = None, None, None, None
                last_gt, last_intercept, last_feature_names, last_coefs = None, None, None, None
                last_n_obs, last_p_vars, last_std_err, last_f_stat = None, None, None, None
                last_train_sample_ids, last_test_sample_ids = None, None
                best_r2_ext = -np.inf
                best_details = None  # will store the best repeat's detailed objects for representative results

                linear_models = ["PLSR (5 LV max)", "PCR (5 PC + OLS)", "Ridge", "Lasso", "ElasticNet", "Stepwise MLR"]
                is_linear = name in linear_models

                for rep in range(n_repeats):
                    if n_repeats > 1:
                        self.status_var.set(f"Training {name} — repeat {rep+1}/{n_repeats}")
                        self.root.update_idletasks()
                    seed = 42 + rep
                    mean_train = None
                    ref = None
                    if val_mode == "split":
                        test_size = self.setup_data['test_size']
                        if self.setup_data['split_algo'] == "Kennard-Stone":
                            train_idx, test_idx = kennard_stone_split(X_cal_all, test_size=test_size)
                        else:
                            indices = np.arange(len(y_cal_all))
                            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

                        X_train = X_cal_all[train_idx]
                        X_test = X_cal_all[test_idx]
                        y_train = y_cal_all[train_idx]
                        y_test = y_cal_all[test_idx]
                        train_sample_ids = cal_sample_ids[train_idx]
                        test_sample_ids = cal_sample_ids[test_idx]
                        
                        # Apply Mean Centering using training mean only (correct, no leakage)
                        if prep_method == "Mean Centering":
                            mean_train = np.mean(X_train, axis=0, keepdims=True)
                            X_train = X_train - mean_train
                            X_test = X_test - mean_train
                        elif prep_method == "Multiplicative Scatter Correction (MSC)":
                            ref = np.mean(X_train, axis=0)
                            X_train = apply_msc(X_train, ref)
                            X_test = apply_msc(X_test, ref)

                        # Collinearity from the training block only.
                        if do_collinear:
                            train_df = pd.DataFrame(X_train, columns=X_cal_reduced_df.columns)
                            train_df, _dropped = eliminate_collinearity(train_df, threshold_pct=collinear_thresh, y=y_train)
                            if train_df.shape[1] == 0:
                                train_df = pd.DataFrame(X_train, columns=X_cal_reduced_df.columns)
                            kept = list(train_df.columns)
                            X_train = train_df.values
                            X_test = pd.DataFrame(X_test, columns=X_cal_reduced_df.columns)[kept].values
                            split_feature_names = train_df.columns
                        else:
                            split_feature_names = X_cal_reduced_df.columns

                        # Optional advanced feature / wavelength selection (training only)
                        feat_method = getattr(self, 'feat_sel_var', None)
                        feat_method = feat_method.get() if feat_method is not None else "None"
                        if feat_method != "None" and X_train.shape[1] > 5:
                            if feat_method == "VIP (PLS)":
                                mask, _ = select_features_by_vip(X_train, y_train, max_features=min(80, X_train.shape[1]))
                            elif feat_method == "Correlation":
                                mask, _ = select_features_by_correlation(X_train, y_train, max_features=min(80, X_train.shape[1]))
                            elif feat_method == "Interval (iPLS-style)":
                                mask, _ = select_features_interval(X_train, y_train, interval_width=max(5, X_train.shape[1]//20))
                            else:
                                mask = np.ones(X_train.shape[1], dtype=bool)
                            if mask.sum() >= 2:
                                X_train = X_train[:, mask]
                                X_test = X_test[:, mask]
                                split_feature_names = split_feature_names[mask]
                    else:
                        X_train = X_cal_all
                        y_train = y_cal_all
                        train_sample_ids = cal_sample_ids
                        X_test = X_test_fixed
                        y_test = y_test_fixed
                        test_sample_ids = test_sample_ids_fixed
                        split_feature_names = X_cal_reduced_df.columns
                        # Feature selection on calibration (training) only, then apply identical mask to external test
                        feat_method = self.feat_sel_var.get() if hasattr(self, 'feat_sel_var') else "None"
                        if feat_method != "None" and X_train.shape[1] > 5:
                            if feat_method == "VIP (PLS)":
                                mask, _ = select_features_by_vip(X_train, y_train, max_features=min(80, X_train.shape[1]))
                            elif feat_method == "Correlation":
                                mask, _ = select_features_by_correlation(X_train, y_train, max_features=min(80, X_train.shape[1]))
                            elif feat_method == "Interval (iPLS-style)":
                                mask, _ = select_features_interval(X_train, y_train, interval_width=max(5, X_train.shape[1]//20))
                            else:
                                mask = np.ones(X_train.shape[1], dtype=bool)
                            if int(mask.sum()) >= 2 and mask.shape[0] == X_train.shape[1]:
                                X_train = X_train[:, mask]
                                if X_test.shape[1] == mask.shape[0]:
                                    X_test = X_test[:, mask]
                                else:
                                    # Align by feature names if column count differs
                                    kept_names = [split_feature_names[i] for i in range(len(mask)) if mask[i]]
                                    # Rebuild test from original external processed matrix if needed
                                    pass
                                split_feature_names = split_feature_names[mask]

                    if name == "PLSR (5 LV max)":
                        best_nc = select_best_n_components("PLSR", X_train, y_train, max_comp=5, seed=seed)
                        model = SafePLSRegression(max_components=best_nc)
                        chosen_alpha = None
                        chosen_nc = best_nc
                        chosen_tree = None
                    elif name == "PCR (5 PC + OLS)":
                        best_nc = select_best_n_components("PCR", X_train, y_train, max_comp=5, seed=seed)
                        model = PipelinePCARegression(max_components=best_nc)
                        chosen_alpha = None
                        chosen_nc = best_nc
                        chosen_tree = None
                    elif name == "Ridge":
                        best_a = select_best_alpha("Ridge", X_train, y_train, seed=seed)
                        model = Ridge(alpha=best_a)
                        chosen_alpha = best_a
                        chosen_nc = None
                        chosen_tree = None
                    elif name == "Lasso":
                        best_a = select_best_alpha("Lasso", X_train, y_train, seed=seed)
                        model = Lasso(alpha=best_a, max_iter=10000)
                        chosen_alpha = best_a
                        chosen_nc = None
                        chosen_tree = None
                    elif name == "ElasticNet":
                        best_a = select_best_alpha("ElasticNet", X_train, y_train, seed=seed)
                        model = ElasticNet(alpha=best_a, l1_ratio=0.5, max_iter=10000)
                        chosen_alpha = best_a
                        chosen_nc = None
                        chosen_tree = None
                    elif name == "SVR (RBF)":
                        best_C = select_best_svr_C(X_train, y_train, seed=seed)
                        model = SVR(kernel='rbf', C=best_C, epsilon=0.1)
                        chosen_alpha = best_C  # reuse field for selected C
                        chosen_nc = None
                        chosen_tree = None
                    elif name == "Random Forest":
                        tree_params = _select_tree_params_light(name, X_train, y_train, seed=seed)
                        d, leaf, ne = tree_params
                        model = RandomForestRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                        chosen_alpha = None
                        chosen_nc = None
                        chosen_tree = tree_params
                    elif name == "Extra Trees":
                        tree_params = _select_tree_params_light(name, X_train, y_train, seed=seed)
                        d, leaf, ne = tree_params
                        model = ExtraTreesRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                        chosen_alpha = None
                        chosen_nc = None
                        chosen_tree = tree_params
                    elif name == "Gradient Boosting":
                        tree_params = _select_tree_params_light(name, X_train, y_train, seed=seed)
                        d, leaf, ne = tree_params
                        model = GradientBoostingRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                        chosen_alpha = None
                        chosen_nc = None
                        chosen_tree = tree_params
                    elif name == "Gaussian Process (GPR)":
                        best_len = select_best_gpr_length(X_train, y_train, seed=seed)
                        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(best_len, (1e-2, 1e2))
                        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, random_state=42)
                        chosen_alpha = best_len
                        chosen_nc = None
                        chosen_tree = None
                    elif name == "kNN Regressor (k=5)":
                        best_k = select_best_knn_k(X_train, y_train, seed=seed)
                        model = KNeighborsRegressor(n_neighbors=best_k)
                        chosen_alpha = None
                        chosen_nc = best_k  # reuse for selected k
                        chosen_tree = None
                    elif name == "MLP / ANN":
                        best_mlp_a = select_best_mlp_alpha(X_train, y_train, seed=seed)
                        model = MLPRegressor(hidden_layer_sizes=(30, 15), max_iter=800, alpha=best_mlp_a, random_state=42)
                        chosen_alpha = best_mlp_a
                        chosen_nc = None
                        chosen_tree = None
                    elif name == "Stepwise MLR":
                        model = LinearRegression()
                        chosen_alpha = None
                        chosen_nc = None
                        chosen_tree = None

                    if name == "Stepwise MLR":
                        scaler_temp = StandardScaler()
                        X_train_scaled_temp = scaler_temp.fit_transform(X_train)
                        step_sel = stepwise_regression_fit(X_train_scaled_temp, y_train, max_vars=15)
                        Xtr_raw = X_train[:, step_sel]
                        Xte_raw = X_test[:, step_sel]
                        feature_names = split_feature_names[step_sel]
                    else:
                        Xtr_raw, Xte_raw = X_train, X_test
                        feature_names = split_feature_names

                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(Xtr_raw)
                    Xte = scaler.transform(Xte_raw)

                    model.fit(Xtr, y_train)
                    
                    # Compute VIP for PLS models
                    vip_scores = None
                    if "PLSR" in name and hasattr(model, 'pls'):
                        vip_scores = calculate_vip(model.pls, Xtr, y_train)

                    y_hat_train = np.asarray(model.predict(Xtr)).ravel()
                    r2_train = float(r2_score(y_train, y_hat_train)) if len(y_train) > 1 and np.std(y_train) > 0 else float("nan")
                    rmsec = float(math.sqrt(mean_squared_error(y_train, y_hat_train)))

                    n_obs = len(y_train)
                    if "PLSR" in name or "PCR" in name:
                        p_vars = model.pls.n_components if "PLSR" in name else model.pca.n_components_
                    elif "Stepwise" in name or "Ridge" in name or "Lasso" in name or "ElasticNet" in name:
                        p_vars = int(np.sum(np.abs(getattr(model, 'coef_', np.zeros(1))) > 1e-6)) if hasattr(model, 'coef_') else len(feature_names)
                    else:
                        p_vars = len(feature_names)

                    if is_linear:
                        r2_adj = adjusted_r2(r2_train, n_obs, p_vars)
                        rss = np.sum((y_train - y_hat_train) ** 2)
                        df_resid = n_obs - p_vars - 1
                        if df_resid > 0:
                            std_err = float(math.sqrt(rss / df_resid))
                        else:
                            std_err = float("nan")
                        if p_vars > 0 and r2_train < 1.0 and df_resid > 0:
                            f_stat = float((r2_train / p_vars) / ((1.0 - r2_train) / df_resid))
                        elif r2_train >= 1.0 and p_vars > 0:
                            f_stat = float("inf")
                        else:
                            f_stat = float("nan")
                    else:
                        # Non-linear models: adj R² and F are not statistically valid
                        r2_adj = float('nan')
                        std_err = float('nan')
                        f_stat = float('nan')
                        p_vars = len(feature_names)  # for reference only

                    # Nested Q² (full or light depending on user toggle)
                    full_nested = self.full_nested_var.get() if hasattr(self, 'full_nested_var') else False
                    q2_cv = evaluate_q2_cv(name, Xtr_raw, y_train, cv_folds=5, seed=seed,
                                           alpha=None, n_components=None,
                                           tree_params=chosen_tree, full_nested=full_nested)

                    y_hat_ext = np.asarray(model.predict(Xte)).ravel()
                    r2_ext = float(r2_score(y_test, y_hat_ext)) if len(y_test) > 1 and np.std(y_test) > 0 else float("nan")
                    rmsep = float(math.sqrt(mean_squared_error(y_test, y_hat_ext)))
                    ccc_ext = concordance_correlation_coefficient(y_test, y_hat_ext)
                    rpd_ext = calculate_rpd(y_test, rmsep)

                    acc_r2_train.append(r2_train)
                    acc_r2_adj.append(r2_adj)
                    acc_rmsec.append(rmsec)
                    acc_q2_cv.append(q2_cv)
                    acc_r2_ext.append(r2_ext)
                    acc_ccc_ext.append(ccc_ext)
                    acc_rmsep.append(rmsep)
                    acc_rpd_ext.append(rpd_ext)

                    last_y_train = y_train
                    last_y_test = y_test
                    last_y_hat_train = y_hat_train
                    last_y_hat_ext = y_hat_ext
                    last_gt = golbraikh_tropsha_criteria(y_test, y_hat_ext)
                    
                    # Convert coefficients back to unscaled domain
                    scale_std = scaler.scale_
                    scale_mean = scaler.mean_
                    scale_std[scale_std == 0] = 1.0
                    
                    if hasattr(model, 'coef_'):
                        sc_coefs = np.asarray(model.coef_).ravel()
                        last_coefs = sc_coefs / scale_std
                        raw_int = getattr(model, 'intercept_', 0.0)
                        raw_int_val = float(raw_int.item() if hasattr(raw_int, "item") else raw_int)
                        last_intercept = raw_int_val - np.sum((sc_coefs * scale_mean) / scale_std)
                    elif hasattr(model, 'feature_importances_'):
                        last_coefs = np.asarray(model.feature_importances_).ravel()
                        last_intercept = 0.0
                    else:
                        last_coefs = np.zeros(len(feature_names))
                        last_intercept = 0.0

                    last_feature_names = feature_names
                    last_n_obs = n_obs
                    last_p_vars = p_vars
                    last_std_err = std_err
                    last_f_stat = f_stat
                    last_train_sample_ids = train_sample_ids
                    last_test_sample_ids = test_sample_ids
                    last_trained_model = model
                    last_Xtr = Xtr
                    last_Xte = Xte
                    last_scaler = scaler

                    # Combined residual + leverage (Hotelling-style) outlier detection
                    try:
                        outlier_mask, outlier_summary, _, _, _ = detect_outliers_residual_leverage(
                            y_train, y_hat_train, Xtr, residual_z=3.0
                        )
                        n_outliers = int(np.sum(outlier_mask))
                    except Exception:
                        n_outliers = 0
                        outlier_summary = ""
                    last_n_outliers = n_outliers

                    # Keep the best-performing repeat (highest external R²) as the representative for equations, plots and GT status
                    if r2_ext > best_r2_ext:
                        best_r2_ext = r2_ext
                        best_details = {
                            'y_train': y_train.copy(), 'y_test': y_test.copy(),
                            'y_hat_train': y_hat_train.copy(), 'y_hat_ext': y_hat_ext.copy(),
                            'gt': last_gt, 'intercept': last_intercept,
                            'feature_names': last_feature_names, 'coefficients': last_coefs.copy() if hasattr(last_coefs, 'copy') else last_coefs,
                            'n_obs': last_n_obs, 'p_vars': last_p_vars, 'std_err': last_std_err, 'f_stat': last_f_stat,
                            'train_sample_ids': train_sample_ids.copy() if hasattr(train_sample_ids, 'copy') else train_sample_ids,
                            'test_sample_ids': test_sample_ids.copy() if hasattr(test_sample_ids, 'copy') else test_sample_ids,
                            'trained_model': model, 'Xtr': Xtr.copy(), 'Xte': Xte.copy(),
                            'scaler': scaler,
                            'vip_scores': vip_scores.copy() if vip_scores is not None else None,
                            'prep_method': prep_method,
                            'sg_window': sg_window,
                            'sg_poly': sg_poly,
                            'collinear_reduction': do_collinear,
                            'collinear_threshold': collinear_thresh,
                            'chosen_alpha': chosen_alpha,
                            'chosen_n_components': chosen_nc,
                            'chosen_tree': chosen_tree,
                            'msc_reference': ref.copy() if ref is not None else None,
                            'mean_centering_mean': mean_train.ravel().copy() if mean_train is not None else None,
                            'n_outliers': n_outliers,
                            # Metrics of this specific (best) repeat — these match the stored predictions
                            'r2_train': r2_train, 'r2_adj': r2_adj, 'rmsec': rmsec,
                            'q2_cv': q2_cv, 'r2_ext': r2_ext, 'ccc_ext': ccc_ext,
                            'rmsep': rmsep, 'rpd_ext': rpd_ext
                        }

                # After all repeats: use averaged metrics, but representative (best) repeat for detailed results
                if n_repeats > 1 and best_details is not None:
                    last_y_train = best_details['y_train']
                    last_y_test = best_details['y_test']
                    last_y_hat_train = best_details['y_hat_train']
                    last_y_hat_ext = best_details['y_hat_ext']
                    last_gt = best_details['gt']
                    last_intercept = best_details['intercept']
                    last_feature_names = best_details['feature_names']
                    last_coefs = best_details['coefficients']
                    last_n_obs = best_details['n_obs']
                    last_p_vars = best_details['p_vars']
                    last_std_err = best_details['std_err']
                    last_f_stat = best_details['f_stat']
                    last_train_sample_ids = best_details['train_sample_ids']
                    last_test_sample_ids = best_details['test_sample_ids']
                    last_trained_model = best_details['trained_model']
                    last_Xtr = best_details['Xtr']
                    last_Xte = best_details.get('Xte')
                    last_scaler = best_details.get('scaler')

                r2_train = float(np.mean(acc_r2_train))
                r2_adj = float(np.nanmean(acc_r2_adj)) if not np.all(np.isnan(acc_r2_adj)) else float('nan')
                rmsec = float(np.mean(acc_rmsec))
                q2_cv = float(np.mean(acc_q2_cv))
                r2_ext = float(np.mean(acc_r2_ext))
                ccc_ext = float(np.mean(acc_ccc_ext))
                rmsep = float(np.mean(acc_rmsep))
                rpd_ext = float(np.mean(acc_rpd_ext))

                # Standard deviations across repeats (0 when N=1)
                r2_train_std = float(np.nanstd(acc_r2_train)) if n_repeats > 1 else 0.0
                rmsec_std = float(np.nanstd(acc_rmsec)) if n_repeats > 1 else 0.0
                q2_cv_std = float(np.nanstd(acc_q2_cv)) if n_repeats > 1 else 0.0
                r2_ext_std = float(np.nanstd(acc_r2_ext)) if n_repeats > 1 else 0.0
                ccc_ext_std = float(np.nanstd(acc_ccc_ext)) if n_repeats > 1 else 0.0
                rmsep_std = float(np.nanstd(acc_rmsep)) if n_repeats > 1 else 0.0
                rpd_ext_std = float(np.nanstd(acc_rpd_ext)) if n_repeats > 1 else 0.0

                gt = last_gt
                status = "✅ Passed" if gt['passed'] else "⚠️ Check"

                self.detailed_results[name] = {
                    'preprocessing': prep_method,
                    'n_repeats': n_repeats,
                    'n_obs': last_n_obs, 'p_vars': last_p_vars, 'std_err': last_std_err, 'f_stat': last_f_stat,
                    # Averaged metrics (used in the results table)
                    'r2_train': r2_train, 'r2_adj': r2_adj, 'rmsec': rmsec,
                    'q2_cv': q2_cv, 'r2_ext': r2_ext, 'ccc_ext': ccc_ext, 'rmsep': rmsep, 'rpd_ext': rpd_ext,
                    # Standard deviations across repeats
                    'r2_train_std': r2_train_std, 'rmsec_std': rmsec_std, 'q2_cv_std': q2_cv_std,
                    'r2_ext_std': r2_ext_std, 'ccc_ext_std': ccc_ext_std, 'rmsep_std': rmsep_std, 'rpd_ext_std': rpd_ext_std,
                    # Best-repeat metrics (match the stored predictions, equation, GT and plots)
                    'best_r2_train': best_details['r2_train'] if best_details else r2_train,
                    'best_r2_adj': best_details['r2_adj'] if best_details else r2_adj,
                    'best_rmsec': best_details['rmsec'] if best_details else rmsec,
                    'best_q2_cv': best_details['q2_cv'] if best_details else q2_cv,
                    'best_r2_ext': best_details['r2_ext'] if best_details else r2_ext,
                    'best_ccc_ext': best_details['ccc_ext'] if best_details else ccc_ext,
                    'best_rmsep': best_details['rmsep'] if best_details else rmsep,
                    'best_rpd_ext': best_details['rpd_ext'] if best_details else rpd_ext,
                    'gt': gt, 'intercept': last_intercept,
                    'feature_names': last_feature_names, 'coefficients': last_coefs,
                    'train_sample_ids': last_train_sample_ids, 'test_sample_ids': last_test_sample_ids,
                    'y_train': last_y_train, 'y_hat_train': last_y_hat_train,
                    'y_test': last_y_test, 'y_hat_ext': last_y_hat_ext,
                    'trained_model': last_trained_model, 'Xtr': last_Xtr, 'Xte': last_Xte,
                    'scaler': last_scaler,
                    'vip_scores': best_details.get('vip_scores') if best_details else None,
                    'sg_window': sg_window,
                    'sg_poly': sg_poly,
                    'collinear_reduction': do_collinear,
                    'collinear_threshold': collinear_thresh,
                    'chosen_alpha': best_details.get('chosen_alpha') if best_details else None,
                    'chosen_n_components': best_details.get('chosen_n_components') if best_details else None,
                    'chosen_tree': best_details.get('chosen_tree') if best_details else None,
                    'msc_reference': best_details.get('msc_reference') if best_details else None,
                    'mean_centering_mean': best_details.get('mean_centering_mean') if best_details else None
                }

                r2_adj_str = f"{r2_adj:.4f}" if not np.isnan(r2_adj) else "N/A"
                rows.append((name, f"{r2_train:.4f}", r2_adj_str, f"{rmsec:.4f}", f"{q2_cv:.4f}",
                             f"{r2_ext:.4f}", f"{ccc_ext:.4f}", f"{rmsep:.4f}", f"{rpd_ext:.3f}", status))

                eq_str_parts = [f"{last_intercept:+.4f}"]
                for f_name, c in zip(last_feature_names, last_coefs):
                    if abs(c) > 1e-6:
                        eq_str_parts.append(f"{c:+.4f}*({f_name})")
                full_eq = f"y = " + " ".join(eq_str_parts[:40])

                note_repeats = ""
                best_note = ""
                if n_repeats > 1 and best_details is not None:
                    note_repeats = (
                        "\nNOTE: The results table shows AVERAGED metrics across all random-split repeats.\n"
                        f"Standard deviations: R²_Ext ± {r2_ext_std:.4f}, RMSEP ± {rmsep_std:.4f}, CCC ± {ccc_ext_std:.4f}, RPD ± {rpd_ext_std:.3f}.\n"
                        "The regression equation, coefficients, sample IDs, stored predictions, Golbraikh-Tropsha\n"
                        "status and the values shown below under 'Best-repeat metrics' come from the single BEST\n"
                        "repeat (highest external R²). Plot statistics are always recomputed from the points shown.\n"
                    )
                    br2t = best_details['r2_train']
                    br2e = best_details['r2_ext']
                    brmsec = best_details['rmsec']
                    brmsep = best_details['rmsep']
                    bccc = best_details['ccc_ext']
                    brpd = best_details['rpd_ext']
                    bq2 = best_details['q2_cv']
                    best_note = (
                        f"\n--- Best-repeat metrics (matching equation & predictions) ---\n"
                        f"• R²_Train: {br2t:.4f}  |  RMSEC: {brmsec:.4f}\n"
                        f"• R²_Ext: {br2e:.4f}  |  RMSEP: {brmsep:.4f}  |  CCC: {bccc:.4f}  |  RPD: {brpd:.3f}\n"
                        f"• Q²_CV (this repeat): {bq2:.4f}\n"
                    )

                note_nonlinear = ""
                if not is_linear:
                    note_nonlinear = "\nNOTE: Adjusted R² and Fisher F-statistic are not statistically valid for this non-linear model (shown as N/A). Feature importances are reported in place of regression coefficients where applicable.\n"

                stepwise_note = ""
                if "Stepwise" in name:
                    stepwise_note = (
                        "\nNOTE (Stepwise MLR): Forward-backward selection with BIC early-stopping, "
                        "stricter p-enter and tighter variable caps on small n. "
                        "Still less stable than ElasticNet/PLSR when n is very small — compare both.\n"
                    )

                r2_adj_disp = f"{r2_adj:.4f}" if not np.isnan(r2_adj) else "N/A"
                f_stat_disp = "N/A" if not np.isfinite(last_f_stat) else f"{last_f_stat:.2f}"
                std_err_disp = "N/A" if not np.isfinite(last_std_err) else f"{last_std_err:.4f}"

                outlier_note = ""
                if best_details and best_details.get('n_outliers', 0) > 0:
                    outlier_note = f"\nNOTE: {best_details['n_outliers']} potential outliers detected (high residual or high leverage / Hotelling-style). Consider reviewing these samples.\n"

                equations_report.append(
                    f"================================================================================\n"
                    f"MODEL: {name} | Preprocessing: {prep_method}\n"
                    f"--------------------------------------------------------------------------------\n"
                    f"• Observations (n): {last_n_obs}  |  Active Variables (p): {last_p_vars}  |  Std. Error (s): {std_err_disp}\n"
                    f"• Fisher F-Statistic (F): {f_stat_disp}  |  Model Status: {status}  (status from best repeat)\n"
                    f"• AVERAGED metrics (table): R²_Train = {r2_train:.4f} ± {r2_train_std:.4f}  |  Q²_CV = {q2_cv:.4f} ± {q2_cv_std:.4f}\n"
                    f"• AVERAGED metrics (table): R²_Ext = {r2_ext:.4f} ± {r2_ext_std:.4f}  |  RMSEP = {rmsep:.4f} ± {rmsep_std:.4f}  |  CCC = {ccc_ext:.4f} ± {ccc_ext_std:.4f}  |  RPD = {rpd_ext:.3f} ± {rpd_ext_std:.3f}\n"
                    f"--------------------------------------------------------------------------------\n"
                    f"VALIDATION DIAGNOSTICS & GOLBRAIKH-TROPSHA CRITERIA ({status}) [from best repeat]:\n"
                    f"• External R² (> 0.6): {gt['r2']:.4f} ({'✔ Pass' if gt['r2'] > 0.6 else '❌ Fail'})\n"
                    f"• Absolute |R² - R²_0| (< 0.1): {gt.get('diff_r2_abs', 1.0):.4f} ({'✔ Pass' if gt.get('diff_r2_abs', 1.0) < 0.1 else '❌ Fail'})\n"
                    f"• Absolute |R² - R²_0'| (< 0.1): {gt.get('diff_r2_prime_abs', 1.0):.4f} ({'✔ Pass' if gt.get('diff_r2_prime_abs', 1.0) < 0.1 else '❌ Fail'})\n"
                    f"• Relative |R² - R²_0| / R² (ref): {gt.get('diff_r2_rel', 1.0):.4f}\n"
                    f"• Relative |R² - R²_0'| / R² (ref): {gt.get('diff_r2_prime_rel', 1.0):.4f}\n"
                    f"• Slope k through origin (0.85 - 1.15): {gt['k']:.4f} ({'✔ Pass' if 0.85 <= gt['k'] <= 1.15 else '❌ Fail'})\n"
                    f"• Slope k' through origin (0.85 - 1.15): {gt['k_prime']:.4f} ({'✔ Pass' if 0.85 <= gt['k_prime'] <= 1.15 else '❌ Fail'})\n"
                    f"=> REASON / DIAGNOSTIC EXPLANATION:\n"
                    f"   {('All external validation conditions met.' if gt['passed'] else 'Model failed Tropsha criteria; check collinearity & scaling settings.')}\n"
                    f"--------------------------------------------------------------------------------\n"
                    f"REGRESSION EQUATION / SUMMARY (from best repeat):\n{full_eq}\n"
                    f"{best_note}{note_repeats}{note_nonlinear}{stepwise_note}{outlier_note}"
                    f"================================================================================\n\n"
                )

            for r in rows:
                self.tree.insert("", tk.END, values=r)

            self.eq_text.config(state=tk.NORMAL)
            self.eq_text.delete(1.0, tk.END)
            self.eq_text.insert(tk.END, "".join(equations_report))
            self.eq_text.config(state=tk.DISABLED)

            self.status_var.set("Models trained successfully with valid cross-validation & statistics!")
            self._log(f"Models finished: {list(self.detailed_results.keys())}")

        except Exception as e:
            self.status_var.set("Error during execution.")
            messagebox.showerror("Execution Error", str(e))

    def generate_shap_plot(self):
        """Generates SHAP figure with full dimension safety."""
        if not self.detailed_results:
            messagebox.showwarning("Warning", "Please run models first!")
            return

        model_names = list(self.detailed_results.keys())
        
        dialog = tk.Toplevel(self.root)
        dialog.title("🧬 Generate SHAP Explanation")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select Model for SHAP Analysis:", font=("Arial", 10, "bold")).pack(pady=10)
        sel_var = tk.StringVar(value=model_names[0])
        cb = ttk.Combobox(dialog, textvariable=sel_var, values=model_names, state="readonly", width=30)
        cb.pack(pady=5)

        def run_shap():
            m_name = sel_var.get()
            dialog.destroy()
            res = self.detailed_results[m_name]
            model = res['trained_model']
            Xtr = res['Xtr']
            feature_names = res['feature_names']

            try:
                fig = plt.figure(figsize=(8, 6), dpi=100)
                ax = fig.add_subplot(111)
                shap_values = None

                if SHAP_AVAILABLE:
                    try:
                        use_kernel = False
                        # Ask user if they want the slow KernelExplainer for non-tree/non-linear models
                        if not hasattr(model, "estimators_") and not hasattr(model, "coef_"):
                            ans = messagebox.askyesno(
                                "SHAP Method",
                                "This model has no fast SHAP explainer.\n\n"
                                "Use KernelExplainer? (slower, more accurate)\n"
                                "No = use coefficient/importance fallback (fast).",
                                parent=dialog if dialog.winfo_exists() else self.root
                            )
                            use_kernel = bool(ans)
                        
                        if hasattr(model, "estimators_"):
                            X_use = Xtr[:min(200, len(Xtr))]
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer(X_use)
                        elif hasattr(model, "coef_"):
                            X_use = Xtr[:min(200, len(Xtr))]
                            explainer = shap.LinearExplainer(model, X_use)
                            shap_values = explainer(X_use)
                        elif use_kernel:
                            bg_data = shap.kmeans(Xtr, min(15, len(Xtr)))
                            explainer = shap.KernelExplainer(model.predict, bg_data)
                            n_shap = min(300, len(Xtr))
                            shap_values = explainer.shap_values(Xtr[:n_shap])
                            X_use = Xtr[:n_shap]
                        else:
                            raise RuntimeError("Using coefficient fallback")
                        
                        shap.summary_plot(shap_values, 
                                          X_use if not isinstance(shap_values, shap.Explanation) else X_use, 
                                          feature_names=feature_names, show=False)
                    except Exception:
                        coefs = res['coefficients']
                        top_idx = np.argsort(np.abs(coefs))[-15:]
                        # Prefer wavelength labels when available
                        labels = []
                        for i in top_idx:
                            if self.wavelengths is not None and i < len(self.wavelengths):
                                labels.append(f"{self.wavelengths[i]:.1f}")
                            else:
                                labels.append(str(feature_names[i]))
                        ax.barh(labels, coefs[top_idx], color='teal')
                        ax.set_title(f"Feature Importances for {m_name} (fallback)")
                        ax.set_xlabel("Magnitude")
                        if self.wavelengths is not None:
                            ax.set_ylabel("Wavelength")
                else:
                    coefs = res['coefficients']
                    top_idx = np.argsort(np.abs(coefs))[-15:]
                    labels = []
                    for i in top_idx:
                        if self.wavelengths is not None and i < len(self.wavelengths):
                            labels.append(f"{self.wavelengths[i]:.1f}")
                        else:
                            labels.append(str(feature_names[i]))
                    ax.barh(labels, coefs[top_idx], color='teal')
                    ax.set_title(f"Feature Importances (Fallback) for {m_name}")
                    ax.set_xlabel("Magnitude")
                    if self.wavelengths is not None:
                        ax.set_ylabel("Wavelength")

                # Prefer the figure that SHAP actually drew on (summary_plot uses plt.gcf())
                current_fig = plt.gcf()
                if current_fig is not None and len(current_fig.axes) > 0:
                    fig = current_fig
                else:
                    fig.tight_layout()

                # Ensure the figure is tight for saving
                try:
                    fig.tight_layout()
                except Exception:
                    pass

                shap_win = tk.Toplevel(self.root)
                shap_win.title(f"🧬 SHAP Analysis - {m_name}")
                shap_win.geometry("820x620")
                shap_win.minsize(640, 480)
                shap_win.resizable(True, True)

                canvas = FigureCanvasTkAgg(fig, master=shap_win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                canvas.get_tk_widget().config(width=760, height=450)

                bottom_frame = ttk.Frame(shap_win, padding=5)
                bottom_frame.pack(fill=tk.X, padx=10, pady=5)

                toolbar = NavigationToolbar2Tk(canvas, bottom_frame)
                toolbar.update()
                toolbar.pack(side=tk.LEFT)

                def save_shap_figure():
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".png",
                        filetypes=[
                            ("PNG Image", "*.png"),
                            ("PDF Document", "*.pdf"),
                            ("SVG Vector", "*.svg"),
                            ("JPEG Image", "*.jpg"),
                            ("TIFF Image", "*.tiff")
                        ],
                        title="Save SHAP Figure",
                        parent=shap_win
                    )
                    if file_path:
                        try:
                            fig.savefig(file_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
                            messagebox.showinfo("Success", f"SHAP figure saved successfully to:\n{file_path}", parent=shap_win)
                        except Exception as save_err:
                            messagebox.showerror("Error", str(save_err), parent=shap_win)

                ttk.Button(bottom_frame, text="💾 Save High-Res Figure", command=save_shap_figure).pack(side=tk.RIGHT, padx=5)

                # Optional: button to save SHAP values / importances to Excel
                def save_shap_data():
                    data_path = filedialog.asksaveasfilename(
                        defaultextension=".xlsx",
                        filetypes=[("Excel Workbook", "*.xlsx")],
                        title="Save SHAP / Importance Data",
                        parent=shap_win
                    )
                    if not data_path:
                        return
                    try:
                        feats = list(feature_names)
                        if shap_values is not None:
                            try:
                                if isinstance(shap_values, shap.Explanation):
                                    mean_abs = np.abs(shap_values.values).mean(axis=0)
                                else:
                                    sv = np.asarray(shap_values)
                                    mean_abs = np.abs(sv).mean(axis=0) if sv.ndim > 1 else np.abs(sv)
                                df_shap = pd.DataFrame({
                                    "Feature": feats[:len(mean_abs)],
                                    "Mean_Abs_SHAP": mean_abs
                                }).sort_values("Mean_Abs_SHAP", ascending=False)
                            except Exception:
                                df_shap = pd.DataFrame({
                                    "Feature": feats,
                                    "Note": "SHAP values available but mean extraction failed; see coefficients sheet in main export"
                                })
                        else:
                            coefs = res.get('coefficients', np.zeros(len(feats)))
                            df_shap = pd.DataFrame({
                                "Feature": feats,
                                "Coefficient_or_Importance": coefs,
                                "Abs_Value": np.abs(coefs)
                            }).sort_values("Abs_Value", ascending=False)
                        df_shap.to_excel(data_path, index=False)
                        messagebox.showinfo("Success", f"SHAP / importance data saved to:\n{data_path}", parent=shap_win)
                    except Exception as e:
                        messagebox.showerror("Error", str(e), parent=shap_win)

                ttk.Button(bottom_frame, text="📊 Save Data to Excel", command=save_shap_data).pack(side=tk.RIGHT, padx=5)

            except Exception as e:
                messagebox.showerror("SHAP Error", f"Could not generate SHAP plot: {str(e)}")

        ttk.Button(dialog, text="Generate Plot", command=run_shap).pack(pady=15, ipadx=10)

    def open_plotter(self):
        if not self.detailed_results:
            messagebox.showwarning("Warning", "Please run models first!")
            return
        PlotViewerWindow(self.root, self.detailed_results)

    def export_excel(self):
        if not self.detailed_results:
            messagebox.showwarning("Warning", "Please run models first!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Workbook", "*.xlsx")])
        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                summary_data = []
                for name, res in self.detailed_results.items():
                    gt = res['gt']
                    summary_data.append({
                        "Model": name, "Preprocessing": res['preprocessing'],
                        "N_Repeats": res.get('n_repeats', 1),
                        "Observations_n": res['n_obs'], "Active_Variables_p": res['p_vars'],
                        "R2_Calibration_Avg": res['r2_train'], "R2_Train_Std": res.get('r2_train_std', 0.0),
                        "R2_Adjusted_Avg": res['r2_adj'],
                        "RMSEC_Avg": res['rmsec'], "RMSEC_Std": res.get('rmsec_std', 0.0),
                        "Q2_CV_Avg": res['q2_cv'], "Q2_CV_Std": res.get('q2_cv_std', 0.0),
                        "R2_Validation_Avg": res['r2_ext'], "R2_Ext_Std": res.get('r2_ext_std', 0.0),
                        "CCC_Validation_Avg": res['ccc_ext'], "CCC_Ext_Std": res.get('ccc_ext_std', 0.0),
                        "RMSEP_Avg": res['rmsep'], "RMSEP_Std": res.get('rmsep_std', 0.0),
                        "RPD_Validation_Avg": res['rpd_ext'], "RPD_Ext_Std": res.get('rpd_ext_std', 0.0),
                        "R2_Validation_Best": res.get('best_r2_ext', res['r2_ext']),
                        "RMSEP_Best": res.get('best_rmsep', res['rmsep']),
                        "CCC_Best": res.get('best_ccc_ext', res['ccc_ext']),
                        "RPD_Best": res.get('best_rpd_ext', res['rpd_ext']),
                        "GT_Status": "Passed" if gt['passed'] else "Check"
                    })
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary_and_Validation", index=False)

                # Methods sheet — useful for the methods section of a paper
                methods_rows = []
                prep = self.prep_var.get() if hasattr(self, 'prep_var') else "Unknown"
                feat_sel = self.feat_sel_var.get() if hasattr(self, 'feat_sel_var') else "None"
                full_nested = self.full_nested_var.get() if hasattr(self, 'full_nested_var') else False
                sg_w, sg_p = self._get_sg_params() if hasattr(self, '_get_sg_params') else (15, 2)
                val_mode = self.setup_data.get('val_mode', 'unknown') if self.setup_data else 'unknown'
                split_algo = self.setup_data.get('split_algo', 'N/A') if self.setup_data else 'N/A'
                test_size = self.setup_data.get('test_size', 'N/A') if self.setup_data else 'N/A'
                n_repeats = self.setup_data.get('random_split_repeats', 1) if self.setup_data else 1
                collinear = self.setup_data.get('collinear_reduction', False) if self.setup_data else False
                collinear_thr = self.setup_data.get('collinear_threshold', 'N/A') if self.setup_data else 'N/A'

                methods_rows.append({"Item": "Preprocessing", "Value": prep})
                methods_rows.append({"Item": "Savitzky-Golay window", "Value": sg_w})
                methods_rows.append({"Item": "Savitzky-Golay polyorder", "Value": sg_p})
                methods_rows.append({"Item": "Feature selection", "Value": feat_sel})
                methods_rows.append({"Item": "Collinearity reduction", "Value": collinear})
                methods_rows.append({"Item": "Collinearity threshold (%)", "Value": collinear_thr})
                methods_rows.append({"Item": "Validation mode", "Value": val_mode})
                methods_rows.append({"Item": "Split algorithm", "Value": split_algo})
                methods_rows.append({"Item": "Test size", "Value": test_size})
                methods_rows.append({"Item": "Random split repeats", "Value": n_repeats})
                methods_rows.append({"Item": "Full nested CV", "Value": full_nested})
                methods_rows.append({"Item": "Q² method", "Value": "Leak-free K-fold; tree models use nested light/full grid inside folds"})
                methods_rows.append({"Item": "Golbraikh-Tropsha form", "Value": "Absolute |R²-R0²|<0.1 (classic)"})
                # Per-model hyperparameters from best repeat
                for name, res in self.detailed_results.items():
                    methods_rows.append({"Item": f"Model: {name}", "Value": ""})
                    methods_rows.append({"Item": f"  chosen_alpha / C / length", "Value": res.get('chosen_alpha')})
                    methods_rows.append({"Item": f"  chosen_n_components / k", "Value": res.get('chosen_n_components')})
                    methods_rows.append({"Item": f"  chosen_tree (depth, leaf, n_est)", "Value": str(res.get('chosen_tree'))})
                    methods_rows.append({"Item": f"  n_features used", "Value": res.get('p_vars')})
                pd.DataFrame(methods_rows).to_excel(writer, sheet_name="Methods", index=False)

                equations_data = []
                for name, res in self.detailed_results.items():
                    intercept = res['intercept']
                    feat_names = res['feature_names']
                    coefs = res['coefficients']
                    eq_parts = [f"{intercept:+.4f}"]
                    for f_name, c in zip(feat_names, coefs):
                        if abs(c) > 1e-6:
                            eq_parts.append(f"{c:+.4f}*({f_name})")
                    full_eq = "y = " + " ".join(eq_parts)
                    
                    gt = res['gt']
                    equations_data.append({
                        "Model": name, "Preprocessing": res['preprocessing'], "Intercept": intercept,
                        "Equation": full_eq, "Observations_n": res['n_obs'], "Active_Variables_p": res['p_vars'],
                        "Std_Error_s": res['std_err'], "F_Statistic": res['f_stat'],
                        "R2_Train": res['r2_train'], "R2_Validation": res['r2_ext'],
                        "RMSEP": res['rmsep'], "RPD_Validation": res['rpd_ext'],
                        "GT_Status": "Passed" if gt['passed'] else "Check"
                    })
                pd.DataFrame(equations_data).to_excel(writer, sheet_name="Model_Equations", index=False)

                cal_values_data = []
                for name, res in self.detailed_results.items():
                    y_train = res['y_train']
                    y_hat_train = res['y_hat_train']
                    train_ids = res['train_sample_ids']
                    for idx, (sid, yt, yp) in enumerate(zip(train_ids, y_train, y_hat_train)):
                        cal_values_data.append({
                            "Model": name, "Sample_Index": idx, "Sample_ID": sid,
                            "Original_y_Calibration": yt, "Predicted_y_Calibration": yp, "Residual": yt - yp
                        })
                pd.DataFrame(cal_values_data).to_excel(writer, sheet_name="Calibration_Values", index=False)

                val_values_data = []
                for name, res in self.detailed_results.items():
                    y_test = res['y_test']
                    y_hat_ext = res['y_hat_ext']
                    test_ids = res['test_sample_ids']
                    for idx, (sid, yt, yp) in enumerate(zip(test_ids, y_test, y_hat_ext)):
                        val_values_data.append({
                            "Model": name, "Sample_Index": idx, "Sample_ID": sid,
                            "Original_y_Validation": yt, "Predicted_y_Validation": yp, "Residual": yt - yp
                        })
                pd.DataFrame(val_values_data).to_excel(writer, sheet_name="Validation_Values", index=False)

                # Add Feature Importances / Coefficients sheet for all models
                imp_rows = []
                for name, res in self.detailed_results.items():
                    feats = res.get('feature_names', [])
                    coefs = res.get('coefficients', [])
                    vip = res.get('vip_scores')
                    for i, (f, c) in enumerate(zip(feats, coefs)):
                        row = {
                            "Model": name,
                            "Feature": str(f),
                            "Coefficient_or_Importance": float(c) if np.isfinite(c) else 0.0,
                            "Abs_Value": float(abs(c)) if np.isfinite(c) else 0.0
                        }
                        if vip is not None and i < len(vip):
                            row["VIP"] = float(vip[i]) if np.isfinite(vip[i]) else np.nan
                        imp_rows.append(row)
                if imp_rows:
                    pd.DataFrame(imp_rows).to_excel(writer, sheet_name="Feature_Importances", index=False)

                # Model comparison ranking (by averaged RMSEP, lower is better)
                comp_rows = []
                for name, res in self.detailed_results.items():
                    comp_rows.append({
                        "Model": name,
                        "RMSEP_Avg": res.get('rmsep', np.nan),
                        "R2_Ext_Avg": res.get('r2_ext', np.nan),
                        "CCC_Ext_Avg": res.get('ccc_ext', np.nan),
                        "RPD_Avg": res.get('rpd_ext', np.nan),
                        "GT_Status": "Passed" if res.get('gt', {}).get('passed') else "Check",
                        "N_Repeats": res.get('n_repeats', 1)
                    })
                if comp_rows:
                    df_comp = pd.DataFrame(comp_rows).sort_values("RMSEP_Avg")
                    df_comp.insert(0, "Rank", range(1, len(df_comp)+1))
                    df_comp.to_excel(writer, sheet_name="Model_Comparison", index=False)

                    # Full pairwise Wilcoxon with Bonferroni, Holm, and FDR (Benjamini-Hochberg)
                    try:
                        from scipy.stats import wilcoxon
                        names = list(df_comp["Model"])
                        pair_rows = []
                        raw_pvals = []
                        pairs = []
                        for i in range(len(names)):
                            for j in range(i+1, len(names)):
                                m1, m2 = names[i], names[j]
                                r1 = self.detailed_results.get(m1, {})
                                r2 = self.detailed_results.get(m2, {})
                                if "y_hat_ext" in r1 and "y_hat_ext" in r2 and len(r1.get("y_hat_ext", [])) == len(r2.get("y_hat_ext", [])) and len(r1.get("y_hat_ext", [])) > 5:
                                    abs1 = np.abs(r1["y_test"] - r1["y_hat_ext"])
                                    abs2 = np.abs(r2["y_test"] - r2["y_hat_ext"])
                                    try:
                                        stat, pval = wilcoxon(abs1, abs2, zero_method="wilcox", alternative="two-sided")
                                        pairs.append((m1, m2, stat, pval))
                                        raw_pvals.append(pval)
                                    except Exception:
                                        pass
                        if pairs:
                            n_pairs = len(pairs)
                            # Bonferroni
                            # Holm
                            order = np.argsort(raw_pvals)
                            holm = np.ones(n_pairs)
                            for rank, idx in enumerate(order):
                                holm[idx] = min(1.0, raw_pvals[idx] * (n_pairs - rank))
                            # FDR (Benjamini-Hochberg)
                            fdr = np.ones(n_pairs)
                            sorted_p = np.array(raw_pvals)[order]
                            for rank, idx in enumerate(order):
                                fdr[idx] = min(1.0, sorted_p[rank] * n_pairs / (rank + 1))
                            # Enforce monotonicity for FDR
                            for k in range(n_pairs-2, -1, -1):
                                fdr[order[k]] = min(fdr[order[k]], fdr[order[k+1]])
                            for k, (m1, m2, stat, pval) in enumerate(pairs):
                                pair_rows.append({
                                    "Model_A": m1,
                                    "Model_B": m2,
                                    "Wilcoxon_statistic": stat,
                                    "p_raw": pval,
                                    "p_Bonferroni": min(1.0, pval * n_pairs),
                                    "p_Holm": holm[k],
                                    "p_FDR_BH": fdr[k],
                                    "Significant_Bonferroni_0.05": "Yes" if min(1.0, pval * n_pairs) < 0.05 else "No",
                                    "Significant_Holm_0.05": "Yes" if holm[k] < 0.05 else "No",
                                    "Significant_FDR_0.05": "Yes" if fdr[k] < 0.05 else "No"
                                })
                            pd.DataFrame(pair_rows).to_excel(writer, sheet_name="Model_Pairwise_Tests", index=False)
                    except Exception:
                        pass

            messagebox.showinfo("Success", f"Full report exported successfully to:\n{file_path}")
            self._log(f"Excel report exported to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def show_about(self):
        """Display program explanation followed by developer information."""
        about_text = (
            "Spectral Analyzer Pro\n\n"
            "A desktop application for chemometrics and QSAR analysis of spectral data.\n"
            "It supports data loading, multiple preprocessing methods, collinearity reduction,\n"
            "training of many regression models, leak-free cross-validation, external validation\n"
            "(including Golbraikh-Tropsha criteria), interactive plots, SHAP explanations,\n"
            "and export of full Excel reports.\n\n"
            "Developed by Dr. Abdelhalim Khaled\n"
            "Plant Ecophysiology\n"
            "University of Concepción"
        )
        messagebox.showinfo("About Spectral Analyzer Pro", about_text, parent=self.root)

    def open_spectral_overview(self):
        if not self.setup_data:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        # Simple spectral overview: mean ± SD of raw and preprocessed calibration spectra
        df_X = self.setup_data['df_X_cal']
        prep = self.prep_var.get()
        sg_window, sg_poly = self._get_sg_params()
        X_raw = df_X.values.astype(float)
        X_prep = apply_preprocessing(df_X, method=prep, sg_window=sg_window, sg_poly=sg_poly).values

        # Wavelength axis: prefer stored vector, then numeric column names, else indices
        wavelengths = np.arange(X_raw.shape[1])
        wl_label = "Feature index"
        if self.wavelengths is not None and len(self.wavelengths) == X_raw.shape[1]:
            wavelengths = self.wavelengths
            wl_label = "Wavelength"
        else:
            try:
                parsed = pd.to_numeric(df_X.columns, errors='coerce')
                if parsed.notna().sum() >= len(parsed) * 0.8:
                    wavelengths = parsed.values.astype(float)
                    wl_label = "Wavelength"
            except Exception:
                pass

        # Optional override: load a new wavelength file for this plot only
        if messagebox.askyesno("Wavelength axis",
                               "Use a different wavelength file for this plot?\n\n"
                               "Yes = load a one-column CSV/Excel\n"
                               "No = use the stored wavelengths (or indices)",
                               parent=self.root):
            wl_path = filedialog.askopenfilename(
                title="Select wavelength file (one column)",
                filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls")]
            )
            if wl_path:
                try:
                    if wl_path.lower().endswith('.csv'):
                        wl_df = pd.read_csv(wl_path, header=None)
                    else:
                        wl_df = pd.read_excel(wl_path, header=None)
                    wl_vals = pd.to_numeric(wl_df.iloc[:, 0], errors='coerce').dropna().values
                    if len(wl_vals) == X_raw.shape[1]:
                        wavelengths = wl_vals.astype(float)
                        wl_label = "Wavelength"
                        self.wavelengths = wavelengths  # store for reuse
                        self._log(f"Updated wavelength vector from Spectral Overview ({len(wl_vals)} points)")
                    else:
                        messagebox.showwarning("Warning",
                            f"Wavelength file has {len(wl_vals)} values but spectra have {X_raw.shape[1]} features. Using previous axis.")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not read wavelengths:\n{e}")

        win = tk.Toplevel(self.root)
        win.title("Spectral Overview")
        win.geometry("860x480")
        win.minsize(640, 360)
        win.resizable(True, True)
        fig = Figure(figsize=(8, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(wavelengths, np.mean(X_raw, axis=0), 'b-', label='Mean')
        ax1.fill_between(wavelengths, np.mean(X_raw,0)-np.std(X_raw,0), np.mean(X_raw,0)+np.std(X_raw,0), alpha=0.3)
        ax1.set_title("Raw Spectra (mean ± SD)")
        ax1.set_xlabel(wl_label)
        ax2.plot(wavelengths, np.mean(X_prep, axis=0), 'g-', label='Mean')
        ax2.fill_between(wavelengths, np.mean(X_prep,0)-np.std(X_prep,0), np.mean(X_prep,0)+np.std(X_prep,0), alpha=0.3)
        ax2.set_title(f"Preprocessed ({prep})")
        ax2.set_xlabel(wl_label)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(win)
        bottom.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, bottom)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)

        def save_spectra():
            fp = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Vector", "*.svg")],
                title="Save Spectral Overview",
                parent=win
            )
            if fp:
                try:
                    fig.savefig(fp, dpi=300, bbox_inches="tight")
                    messagebox.showinfo("Success", f"Spectral overview saved to:\n{fp}", parent=win)
                except Exception as e:
                    messagebox.showerror("Error", str(e), parent=win)

        ttk.Button(bottom, text="💾 Save High-Res Figure", command=save_spectra).pack(side=tk.RIGHT, padx=5)

    def open_residual_plot(self):
        if not self.detailed_results:
            messagebox.showwarning("Warning", "Please run models first!")
            return
        # Simple residual vs predicted for the selected model (best repeat)
        names = list(self.detailed_results.keys())
        dialog = tk.Toplevel(self.root)
        dialog.title("Residual Plot")
        dialog.geometry("350x150")
        ttk.Label(dialog, text="Select model:").pack(pady=5)
        sel = tk.StringVar(value=names[0])
        cb = ttk.Combobox(dialog, textvariable=sel, values=names, state="readonly")
        cb.pack(pady=5)
        def show():
            m = sel.get()
            dialog.destroy()
            res = self.detailed_results[m]
            y_true = res['y_test']
            y_pred = res['y_hat_ext']
            resid = y_true - y_pred
            # Flag outliers on the validation set if possible
            try:
                Xte = res.get('Xte')
                if Xte is not None and len(Xte) == len(y_true):
                    out_mask, _, _, _, _ = detect_outliers_residual_leverage(y_true, y_pred, Xte)
                else:
                    out_mask = np.zeros(len(y_true), dtype=bool)
            except Exception:
                out_mask = np.zeros(len(y_true), dtype=bool)

            win = tk.Toplevel(self.root)
            win.title(f"Residuals - {m}")
            win.geometry("720x500")
            win.minsize(560, 400)
            win.resizable(True, True)
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            normal = ~out_mask
            ax.scatter(y_pred[normal], resid[normal], edgecolors='k', alpha=0.7, label="Normal")
            if np.any(out_mask):
                ax.scatter(y_pred[out_mask], resid[out_mask], color='red', edgecolors='k', s=60, label="Potential outlier")
            ax.axhline(0, color='r', linestyle='--')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual (Actual - Predicted)")
            ax.set_title(f"Residuals vs Predicted ({m})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            bottom = ttk.Frame(win)
            bottom.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, bottom)
            toolbar.update()
            toolbar.pack(side=tk.LEFT)

            def save_resid():
                fp = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Vector", "*.svg")],
                    title="Save Residual Plot",
                    parent=win
                )
                if fp:
                    try:
                        fig.savefig(fp, dpi=300, bbox_inches="tight")
                        messagebox.showinfo("Success", f"Residual plot saved to:\n{fp}", parent=win)
                    except Exception as e:
                        messagebox.showerror("Error", str(e), parent=win)

            # Also offer Hotelling T² / contribution view
            def show_t2():
                try:
                    Xte = res.get('Xte')
                    if Xte is None:
                        messagebox.showinfo("Info", "T² diagnostics require stored Xte.", parent=win)
                        return
                    _, _, resid_arr, lev_d, t2 = detect_outliers_residual_leverage(y_true, y_pred, Xte)
                    t2_win = tk.Toplevel(win)
                    t2_win.title(f"Hotelling T² & Leverage - {m}")
                    t2_win.geometry("780x440")
                    t2_win.minsize(600, 340)
                    t2_win.resizable(True, True)
                    fig2 = Figure(figsize=(7, 3.5), dpi=100)
                    ax1 = fig2.add_subplot(121)
                    ax2 = fig2.add_subplot(122)
                    ax1.scatter(range(len(t2)), t2, edgecolors='k', alpha=0.7)
                    ax1.axhline(np.percentile(t2, 95) if len(t2) else 0, color='r', linestyle='--', label='95th percentile')
                    ax1.set_title("Hotelling's T² (approx)")
                    ax1.set_xlabel("Sample index")
                    ax1.set_ylabel("T²")
                    ax1.legend(fontsize=8)
                    ax2.scatter(range(len(lev_d)), lev_d, edgecolors='k', alpha=0.7, color='orange')
                    ax2.axhline(np.percentile(lev_d, 95) if len(lev_d) else 0, color='r', linestyle='--', label='95th percentile')
                    ax2.set_title("Leverage (Mahalanobis distance)")
                    ax2.set_xlabel("Sample index")
                    ax2.set_ylabel("Distance")
                    ax2.legend(fontsize=8)
                    fig2.tight_layout()
                    c2 = FigureCanvasTkAgg(fig2, master=t2_win)
                    c2.draw()
                    c2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    NavigationToolbar2Tk(c2, t2_win).update()
                except Exception as e:
                    messagebox.showerror("Error", str(e), parent=win)

            # Full residual contribution map (all samples × all features, downsampled if large)
            def show_contrib():
                try:
                    Xte = res.get('Xte')
                    if Xte is None or Xte.shape[1] < 2:
                        messagebox.showinfo("Info", "Contribution plot requires stored feature matrix.", parent=win)
                        return
                    # Full contribution matrix
                    contrib_full = resid[:, None] * Xte
                    n_samp, n_feat = contrib_full.shape
                    # Downsample features if very wide for display
                    if n_feat > 200:
                        step = max(1, n_feat // 200)
                        feat_idx = np.arange(0, n_feat, step)
                        mat = contrib_full[:, feat_idx]
                        title_extra = f" (features downsampled every {step})"
                    else:
                        mat = contrib_full
                        title_extra = " (all features)"
                    # Limit samples for display if huge
                    if n_samp > 100:
                        samp_idx = np.argsort(np.abs(resid))[-100:]
                        mat = mat[samp_idx]
                        title_extra += " | top 100 |residual| samples"
                    contrib_win = tk.Toplevel(win)
                    contrib_win.title(f"Full Residual Contribution Map - {m}")
                    contrib_win.geometry("900x500")
                    contrib_win.minsize(700, 400)
                    contrib_win.resizable(True, True)
                    figc = Figure(figsize=(9, 4.5), dpi=100)
                    axc = figc.add_subplot(111)
                    vmax = np.percentile(np.abs(mat), 98) if mat.size else 1.0
                    im = axc.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
                    axc.set_title(f"Residual × Feature contribution{title_extra}")
                    # Use stored wavelengths on the x-axis when available and length matches
                    if self.wavelengths is not None and len(self.wavelengths) == n_feat:
                        # Show a few wavelength ticks
                        n_ticks = min(8, n_feat)
                        tick_pos = np.linspace(0, n_feat - 1, n_ticks).astype(int)
                        axc.set_xticks(tick_pos)
                        axc.set_xticklabels([f"{self.wavelengths[i]:.1f}" for i in tick_pos], rotation=45, ha='right')
                        axc.set_xlabel("Wavelength")
                    else:
                        axc.set_xlabel("Feature index")
                    axc.set_ylabel("Sample index")
                    figc.colorbar(im, ax=axc, label="Contribution")
                    figc.tight_layout()
                    cc = FigureCanvasTkAgg(figc, master=contrib_win)
                    cc.draw()
                    cc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    bottom_c = ttk.Frame(contrib_win)
                    bottom_c.pack(fill=tk.X)
                    NavigationToolbar2Tk(cc, bottom_c).update()

                    def save_contrib():
                        fp = filedialog.asksaveasfilename(
                            defaultextension=".png",
                            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Vector", "*.svg")],
                            title="Save Contribution Map",
                            parent=contrib_win
                        )
                        if fp:
                            try:
                                figc.savefig(fp, dpi=300, bbox_inches="tight")
                                messagebox.showinfo("Success", f"Contribution map saved to:\n{fp}", parent=contrib_win)
                            except Exception as e:
                                messagebox.showerror("Error", str(e), parent=contrib_win)

                    ttk.Button(bottom_c, text="💾 Save High-Res Figure", command=save_contrib).pack(side=tk.RIGHT, padx=5)
                except Exception as e:
                    messagebox.showerror("Error", str(e), parent=win)

            ttk.Button(bottom, text="📈 Contribution Plot", command=show_contrib).pack(side=tk.RIGHT, padx=5)
        ttk.Button(dialog, text="Show Residuals", command=show).pack(pady=10)

    def run_y_scramble(self):
        if not self.detailed_results or not self.setup_data:
            messagebox.showwarning("Warning", "Please run models first!")
            return
        names = list(self.detailed_results.keys())
        dialog = tk.Toplevel(self.root)
        dialog.title("Y-Scramble Test")
        dialog.geometry("380x180")
        ttk.Label(dialog, text="Select model for scrambling:").pack(pady=5)
        sel = tk.StringVar(value=names[0])
        ttk.Combobox(dialog, textvariable=sel, values=names, state="readonly").pack(pady=5)
        ttk.Label(dialog, text="Number of permutations:").pack()
        nperm_var = tk.StringVar(value="50")
        ttk.Entry(dialog, textvariable=nperm_var, width=8).pack(pady=5)
        def run():
            m_name = sel.get()
            nperm = int(nperm_var.get())
            dialog.destroy()
            res = self.detailed_results[m_name]
            Xtr = res.get('Xtr')
            Xte = res.get('Xte')
            y_train = res['y_train']
            y_test = res['y_test']
            real_r2 = res.get('best_r2_ext', res['r2_ext'])
            if Xtr is None or Xte is None or y_test is None:
                messagebox.showerror("Error", "Required data for external scramble not available.")
                return
            def _one_scramble(seed_i):
                rng_local = np.random.RandomState(seed_i)
                y_scr = rng_local.permutation(y_train)
                try:
                    # Reuse the exact hyperparameters selected for the real model
                    alpha = res.get('chosen_alpha')
                    nc = res.get('chosen_n_components')
                    tree_p = res.get('chosen_tree')
                    if "Ridge" in m_name:
                        mdl = Ridge(alpha=alpha if alpha is not None else 1.0)
                    elif "Lasso" in m_name:
                        mdl = Lasso(alpha=alpha if alpha is not None else 0.01, max_iter=10000)
                    elif "Elastic" in m_name:
                        mdl = ElasticNet(alpha=alpha if alpha is not None else 0.01, l1_ratio=0.5, max_iter=10000)
                    elif "PLSR" in m_name:
                        mdl = SafePLSRegression(max_components=nc if nc is not None else 5)
                    elif "PCR" in m_name:
                        mdl = PipelinePCARegression(max_components=nc if nc is not None else 5)
                    elif "SVR" in m_name:
                        mdl = SVR(kernel='rbf', C=alpha if alpha is not None else 10.0, epsilon=0.1)
                    elif "Random Forest" in m_name:
                        d, leaf, ne = tree_p if tree_p else (6, 3, 100)
                        mdl = RandomForestRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                    elif "Extra Trees" in m_name:
                        d, leaf, ne = tree_p if tree_p else (6, 3, 100)
                        mdl = ExtraTreesRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                    elif "Gradient" in m_name:
                        d, leaf, ne = tree_p if tree_p else (3, 3, 100)
                        mdl = GradientBoostingRegressor(n_estimators=ne, max_depth=d, min_samples_leaf=leaf, random_state=42)
                    elif "kNN" in m_name:
                        k = nc if nc is not None else min(5, len(y_train)-1)
                        mdl = KNeighborsRegressor(n_neighbors=k)
                    elif "MLP" in m_name:
                        mdl = MLPRegressor(hidden_layer_sizes=(30, 15), max_iter=500,
                                           alpha=alpha if alpha is not None else 0.01, random_state=42)
                    elif "Gaussian" in m_name:
                        length = alpha if alpha is not None else 1.0
                        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length, (1e-2, 1e2))
                        mdl = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, random_state=42)
                    else:
                        mdl = LinearRegression()
                    mdl.fit(Xtr, y_scr)
                    yhat_ext = np.asarray(mdl.predict(Xte)).ravel()
                    return float(r2_score(y_test, yhat_ext)) if len(y_test) > 1 and np.std(y_test) > 0 else 0.0
                except Exception:
                    return 0.0

            try:
                from joblib import Parallel, delayed
                scrambled_ext_r2 = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_one_scramble)(42 + i) for i in range(nperm)
                )
            except Exception:
                scrambled_ext_r2 = [_one_scramble(42 + i) for i in range(nperm)]
            scrambled_ext_r2 = np.array(scrambled_ext_r2)
            msg = (f"Y-Scramble Results for {m_name} (external evaluation)\n"
                   f"Real external R² (best): {real_r2:.4f}\n"
                   f"Scrambled external R² mean: {np.mean(scrambled_ext_r2):.4f} ± {np.std(scrambled_ext_r2):.4f}\n"
                   f"Max scrambled external R²: {np.max(scrambled_ext_r2):.4f}\n"
                   f"Fraction of scrambled ≥ real: {np.mean(scrambled_ext_r2 >= real_r2):.3f}")
            messagebox.showinfo("Y-Scramble", msg)
            self.status_var.set("Y-scramble finished.")
        ttk.Button(dialog, text="Run Scramble", command=run).pack(pady=10)

    def save_model(self):
        if not self.detailed_results:
            messagebox.showwarning("Warning", "Please run models first!")
            return
        names = list(self.detailed_results.keys())
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Model")
        dialog.geometry("350x120")
        ttk.Label(dialog, text="Select model to save:").pack(pady=5)
        sel = tk.StringVar(value=names[0])
        ttk.Combobox(dialog, textvariable=sel, values=names, state="readonly").pack(pady=5)
        def do_save():
            m = sel.get()
            dialog.destroy()
            res = self.detailed_results[m]
            path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib", "*.joblib")])
            if not path:
                return
            try:
                import joblib
                res = self.detailed_results[m]
                pipeline = ClosedPipeline(
                    model=res['trained_model'],
                    scaler=res.get('scaler'),
                    feature_names=res['feature_names'],
                    preprocessing=res['preprocessing'],
                    sg_window=res.get('sg_window', 15),
                    sg_poly=res.get('sg_poly', 2),
                    msc_reference=res.get('msc_reference'),
                    mean_centering_mean=res.get('mean_centering_mean')
                )
                payload = {
                    'pipeline': pipeline,
                    'model': res['trained_model'],
                    'scaler': res.get('scaler'),
                    'feature_names': list(res['feature_names']),
                    'preprocessing': res['preprocessing'],
                    'sg_window': res.get('sg_window', 15),
                    'sg_poly': res.get('sg_poly', 2),
                    'collinear_reduction': res.get('collinear_reduction', False),
                    'collinear_threshold': res.get('collinear_threshold', 90.0),
                    'chosen_alpha': res.get('chosen_alpha'),
                    'chosen_n_components': res.get('chosen_n_components'),
                    'msc_reference': res.get('msc_reference'),
                    'mean_centering_mean': res.get('mean_centering_mean'),
                    'intercept': res.get('intercept'),
                    'coefficients': res.get('coefficients'),
                    'vip_scores': res.get('vip_scores'),
                    'n_repeats': res.get('n_repeats', 1),
                    'version': 'closed_pipeline_v3'
                }
                joblib.dump(payload, path)
                messagebox.showinfo("Success", f"Full pipeline model saved to:\n{path}\n\nPrediction will automatically apply the same preprocessing, feature selection, and scaling.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        ttk.Button(dialog, text="Save", command=do_save).pack(pady=10)

    def predict_new_samples(self):
        path = filedialog.askopenfilename(title="Select saved model", filetypes=[("Joblib", "*.joblib")])
        if not path:
            return
        try:
            import joblib
            payload = joblib.load(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model:\n{e}")
            return
        x_path = filedialog.askopenfilename(title="Select new spectral file (CSV/XLSX) — raw spectra with original columns", filetypes=[("CSV/Excel", "*.csv *.xlsx")])
        if not x_path:
            return
        try:
            if x_path.lower().endswith('.csv'):
                df_new = pd.read_csv(x_path)
            else:
                df_new = pd.read_excel(x_path)

            # Align columns: prefer exact names. Positional fallback only after explicit confirmation.
            feats = payload.get('feature_names', [])
            if feats and not all(f in df_new.columns for f in feats):
                if len(df_new.columns) >= len(feats):
                    ans = messagebox.askyesno("Column Alignment Warning",
                        "Column names do not match the model.\n\n"
                        "Do you want to use the first N columns in positional order?\n"
                        "(Only do this if you are sure the column order exactly matches the original training file.)",
                        parent=self.root)
                    if not ans:
                        messagebox.showinfo("Cancelled", "Prediction cancelled. Please provide a file with matching column names.", parent=self.root)
                        return
                    df_aligned = df_new.iloc[:, :len(feats)].copy()
                    df_aligned.columns = feats
                    df_new = df_aligned
                else:
                    messagebox.showerror("Error", "New file has fewer columns than the model expects.")
                    return

            # ---- Robust numeric cleaning (fixes "could not convert string to float") ----
            # Clean only the spectral columns the model expects.
            if feats:
                spectral_cols = [c for c in feats if c in df_new.columns]
            else:
                spectral_cols = list(df_new.columns)
            cleaned = df_new.copy()
            for c in spectral_cols:
                series = cleaned[c]
                if series.dtype == object:
                    # European decimal comma → dot, then numeric
                    series = series.astype(str).str.replace(",", ".", regex=False)
                cleaned[c] = pd.to_numeric(series, errors="coerce")
            n_bad_cells = int(cleaned[spectral_cols].isna().sum().sum())
            if n_bad_cells > 0:
                messagebox.showwarning(
                    "Non-numeric values",
                    f"{n_bad_cells} cell(s) in the spectral columns could not be converted to numbers "
                    f"and were set to missing.\n\n"
                    "Common causes: sample-name columns, text headers, empty cells, or decimal commas.\n"
                    "Rows that still contain missing spectral values will be skipped.",
                    parent=self.root
                )
            df_new = cleaned

            if 'pipeline' in payload and isinstance(payload['pipeline'], ClosedPipeline):
                preds, idx = payload['pipeline'].predict(df_new)
                out = df_new.loc[idx].copy()
                out['Predicted'] = preds
            else:
                # Fallback for older saved models
                prep = payload.get('preprocessing', 'Raw')
                sg_w = payload.get('sg_window', 15)
                sg_p = payload.get('sg_poly', 2)
                feats = payload['feature_names']
                msc_ref = payload.get('msc_reference')
                mc_mean = payload.get('mean_centering_mean')

                if prep == "Mean Centering" and mc_mean is not None:
                    missing = [f for f in feats if f not in df_new.columns]
                    if missing:
                        messagebox.showerror("Error", f"Missing features: {missing[:8]}...")
                        return
                    X_raw = df_new[feats].apply(pd.to_numeric, errors='coerce').values
                    X_proc = pd.DataFrame(X_raw - mc_mean, columns=feats)
                elif prep == "Multiplicative Scatter Correction (MSC)" and msc_ref is not None:
                    missing = [f for f in feats if f not in df_new.columns]
                    if missing:
                        messagebox.showerror("Error", f"Missing features: {missing[:8]}...")
                        return
                    X_raw = df_new[feats].apply(pd.to_numeric, errors='coerce').values
                    X_msc = apply_msc(X_raw, msc_ref)
                    X_proc = pd.DataFrame(X_msc, columns=feats)
                else:
                    X_proc_df = apply_preprocessing(df_new, method=prep, sg_window=sg_w, sg_poly=sg_p)
                    missing = [f for f in feats if f not in X_proc_df.columns]
                    if missing:
                        messagebox.showerror("Error", f"After preprocessing, missing features: {missing[:8]}...")
                        return
                    X_proc = X_proc_df[feats].apply(pd.to_numeric, errors='coerce')

                X_proc = X_proc.dropna()
                if len(X_proc) == 0:
                    messagebox.showerror("Error", "No complete rows remain after preprocessing.")
                    return

                scaler = payload.get('scaler')
                if scaler is not None:
                    X_scaled = scaler.transform(X_proc.values)
                else:
                    X_scaled = X_proc.values

                preds = np.asarray(payload['model'].predict(X_scaled)).ravel()
                out = df_new.loc[X_proc.index].copy()
                out['Predicted'] = preds

            save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
            if save_path:
                out.to_excel(save_path, index=False)
                messagebox.showinfo("Success", f"Predictions (closed pipeline) saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def export_importances(self):
        if not self.detailed_results:
            messagebox.showwarning("Warning", "Please run models first!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                for name, res in self.detailed_results.items():
                    feats = res.get('feature_names', [])
                    coefs = res.get('coefficients', [])
                    vip = res.get('vip_scores')
                    if len(feats) == 0 or len(coefs) == 0:
                        continue
                    data = {
                        'Feature': feats,
                        'Coefficient_or_Importance': coefs,
                        'Abs_Value': np.abs(coefs)
                    }
                    if vip is not None and len(vip) == len(feats):
                        data['VIP'] = vip
                    else:
                        # Simple univariate |correlation| with y as a universal importance proxy
                        try:
                            y_ref = res.get('y_train')
                            X_ref = res.get('Xtr')
                            if y_ref is not None and X_ref is not None and X_ref.shape[1] == len(feats):
                                corrs = []
                                for j in range(X_ref.shape[1]):
                                    if np.std(X_ref[:, j]) == 0 or np.std(y_ref) == 0:
                                        corrs.append(0.0)
                                    else:
                                        c = np.corrcoef(X_ref[:, j], y_ref)[0, 1]
                                        corrs.append(0.0 if not np.isfinite(c) else abs(c))
                                data['Univariate_AbsCorr'] = corrs
                        except Exception:
                            pass
                    df = pd.DataFrame(data).sort_values('Abs_Value', ascending=False)
                    # Sheet name limited to 31 chars
                    sheet = name[:28]
                    df.to_excel(writer, sheet_name=sheet, index=False)
            messagebox.showinfo("Success", f"Feature importances/coefficients exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SpectralAnalyzerApp(root)
        root.mainloop()
    except Exception as exc:
        try:
            messagebox.showerror("Spectral Analyzer Pro", f"The application could not open:\n\n{exc}")
        except Exception:
            print("The application could not open:", exc)
        raise