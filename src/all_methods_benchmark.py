"""
Smart Packaging Line — 12 Classifier Methods Benchmark
========================================================

Two-level weight-based kit inspection system implementing 12 methods:

STATISTICAL METHODS (8):
    1. Runs 1.0 (k=3)        — Shewhart absolute deficit baseline
    2. Runs 1.0 + 3.5σ       — Relaxed control limits
    3. Runs 1.0 + 6σ         — Six Sigma strict
    4. Runs Z-score          — σ-normalized deficit
    5. Runs Z-score + 6σ     — Z-score with relaxed L1
    6. Mahalanobis           — F-Hotelling threshold + bias correction
    7. GMM                   — Gaussian Mixture Model
    8. Bayes Optimal         — Maximum likelihood under normality

MACHINE LEARNING METHODS (5, with 5-fold Stratified CV):
    9.  Logistic Regression  — Multinomial with 16 derived features
    10. LDA                  — Linear Discriminant Analysis (Fisher 1936)
    11. QDA                  — Quadratic Discriminant Analysis
    12. SVM (RBF)            — Support Vector Machine
    13. Random Forest        — 200 decision trees ensemble

All numerical parameters are extracted directly from the DOE Excel
workbook (DOE_Smart_Packaging_Line_v7.xlsx) — calibrated with n=30
samples per component.

Authors: Ian Vargas, José Luis Espinosa, Rubén Figueroa
Capstone: Smart Packaging Line — Tec de Monterrey × TE Connectivity
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, f as f_dist
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# System Parameters (extracted from Excel: Components sheet, n=30 per component)
# Source: DOE_Smart_Packaging_Line_v7.xlsx → Components → measured_mean_g / std_g
# ═════════════════════════════════════════════════════════════════════════════

COMPONENTS: Dict[str, Dict[str, float]] = {
    "S1": {"mean": 2.753333333333333,  "std": 0.006608945522512526},
    "S2": {"mean": 4.036000000000000,  "std": 0.013796551293211020},
    "W1": {"mean": 0.993333333333333,  "std": 0.013978637231524930},
    "W2": {"mean": 4.562666666666667,  "std": 0.025179813114988600},
    "N1": {"mean": 1.155666666666666,  "std": 0.027628487713469430},
    "N2": {"mean": 1.374333333333333,  "std": 0.025145553296253180},
}

RECIPES: Dict[str, Dict[str, int]] = {
    "R3": {"S2": 1, "W2": 1, "N2": 1},
    "R4": {"S1": 1, "W1": 1, "N1": 1, "S2": 1},
    "R6": {"S1": 1, "W1": 2, "N1": 1, "N2": 2},
}

COMP_LIST = ["S1", "S2", "W1", "W2", "N1", "N2"]


def mu_kit(recipe_id: str) -> float:
    """Expected total weight of a complete kit (variance propagation)."""
    return sum(qty * COMPONENTS[c]["mean"]
               for c, qty in RECIPES[recipe_id].items())


def sigma_kit(recipe_id: str) -> float:
    """Propagated standard deviation of kit weight."""
    return np.sqrt(sum(qty * COMPONENTS[c]["std"] ** 2
                       for c, qty in RECIPES[recipe_id].items()))


# ═════════════════════════════════════════════════════════════════════════════
# Level 1 — OK / NG Classification (Shewhart)
# ═════════════════════════════════════════════════════════════════════════════

def l1_shewhart(weight_g: float, recipe_id: str, k: float = 3.0) -> str:
    """
    L1 classifier: Shewhart control chart with k-sigma limits.
    A kit is OK if |z| <= k, where z = (w - mu) / sigma.

    Reference: Shewhart (1931). Economic Control of Quality of
               Manufactured Product. Van Nostrand.
    """
    mu = mu_kit(recipe_id)
    sig = sigma_kit(recipe_id)
    z = (weight_g - mu) / sig
    return "OK" if abs(z) <= k else "NG"


# ═════════════════════════════════════════════════════════════════════════════
# Level 2 — Statistical Methods (no training required)
# ═════════════════════════════════════════════════════════════════════════════

def l2_runs10(deficit_g: float, recipe_id: str) -> str:
    """
    Method 1: Runs 1.0 — Absolute deficit.
    missing = argmin_i |deficit - mu_i|

    Best statistical baseline (96.06% L2 in DOE).
    """
    distances = {c: abs(deficit_g - COMPONENTS[c]["mean"])
                 for c in RECIPES[recipe_id]}
    return min(distances, key=distances.get)


def l2_zscore(deficit_g: float, recipe_id: str) -> str:
    """
    Method 2: Z-score — Normalized deficit by component sigma.
    missing = argmin_i |deficit - mu_i| / sigma_i

    Penalizes components with small sigma more heavily — sometimes
    detrimental for W1 vs N1 discrimination (W1 has small sigma).
    """
    distances = {c: abs(deficit_g - COMPONENTS[c]["mean"]) / COMPONENTS[c]["std"]
                 for c in RECIPES[recipe_id]}
    return min(distances, key=distances.get)


def l1_mahalanobis(weight_g: float, recipe_id: str,
                    n_cal: int = 30, alpha: float = 0.0027) -> str:
    """
    Method 3: Mahalanobis L1 with F-Hotelling threshold and bias correction.

    D² = D₁² - 2P/n           (eq. 4.12, Mahalanobis 1936)
    Threshold = P(n+1)/(n-P-1) · F_crit  (eq. 4.31)

    With P=1 variable, theoretically equivalent to z-score but with
    statistically calibrated threshold and bias correction.

    Reference: Mahalanobis (1936). On the Generalised Distance in Statistics.
               PNISI, 2(1), 49-55. Eqs. 4.11, 4.12, 4.31.
    """
    P = 1  # number of variables
    F_crit = f_dist.ppf(1 - alpha, P, n_cal - P - 1)
    D2_threshold = P * (n_cal + 1) / (n_cal - P - 1) * F_crit
    z_sq = ((weight_g - mu_kit(recipe_id)) / sigma_kit(recipe_id)) ** 2
    D2_corrected = max(0.0, z_sq - 2 * P / n_cal)
    return "OK" if D2_corrected <= D2_threshold else "NG"


def l2_gmm(deficit_g: float, recipe_id: str) -> str:
    """
    Method 4: GMM — Gaussian Mixture Model via log-likelihood.
    missing = argmax_i log N(deficit; mu_i, sigma_i^2)

    Reference: McLachlan & Peel (2000). Finite Mixture Models. Wiley.
    """
    log_probs = {
        c: norm.logpdf(deficit_g,
                       loc=COMPONENTS[c]["mean"],
                       scale=COMPONENTS[c]["std"])
        for c in RECIPES[recipe_id]
    }
    return max(log_probs, key=log_probs.get)


def l2_bayes_optimal(deficit_g: float, recipe_id: str) -> str:
    """
    Method 5: Bayes Optimal — Like GMM but evaluates against ONE unit
    of each component (matches DOE assumption: only 1 unit is missing).

    Reference: Hastie, Tibshirani, Friedman (2009). Elements of Statistical
               Learning §4.4. Springer.
    """
    log_probs = {
        c: norm.logpdf(deficit_g,
                       loc=COMPONENTS[c]["mean"],
                       scale=COMPONENTS[c]["std"])
        for c in RECIPES[recipe_id]
    }
    return max(log_probs, key=log_probs.get)


# ═════════════════════════════════════════════════════════════════════════════
# Level 2 — Machine Learning Methods (require training)
# ═════════════════════════════════════════════════════════════════════════════

def build_features(weight_g: float, recipe_id: str) -> np.ndarray:
    """
    Build the 16-dimensional feature vector for an NG kit.

    Features (16 total):
        4 global       : weight, deficit, z, weight_norm
        12 per-comp    : dist_X and absdist_X for each of 6 components

    For components not in the recipe (qty=0), features are set to 99.0
    as sentinel value.
    """
    mu, sig = mu_kit(recipe_id), sigma_kit(recipe_id)
    deficit = mu - weight_g
    feats = [
        weight_g,
        deficit,
        deficit / sig,
        (weight_g - mu) / sig,
    ]
    for comp in COMP_LIST:
        qty = RECIPES[recipe_id].get(comp, 0)
        if qty > 0:
            mu_c = COMPONENTS[comp]["mean"] * qty
            sig_c = COMPONENTS[comp]["std"] * np.sqrt(qty)
            feats.append((deficit - mu_c) / sig_c)
            feats.append(abs(deficit - mu_c))
        else:
            feats.append(99.0)
            feats.append(99.0)
    return np.array(feats)


def get_ml_models(random_state: int = 42) -> Dict[str, object]:
    """
    Return the 5 ML model templates with publication-quality hyperparameters.

    References:
        - LogisticRegression: Cox & Snell (1989), Agresti (2002)
        - LDA: Fisher (1936)
        - QDA: Hastie et al. (2009) §4.3
        - SVM: Vapnik (1995)
        - Random Forest: Breiman (2001)
    """
    return {
        "logistic": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "lda": LinearDiscriminantAnalysis(solver="svd"),
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "svm": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def cross_validate_ml(
    df: pd.DataFrame,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train an ML model with k-fold Stratified CV per recipe.

    Each kit is predicted exactly once by a model that did NOT see it
    during training. No data leakage.

    Returns
    -------
    df : DataFrame with new columns:
        {model_name}_pred       — predicted missing component
        {model_name}_prob_top1  — top-1 probability
        {model_name}_prob_top2  — top-2 probability
    """
    df = df.copy()
    df[f"{model_name}_pred"]      = "NONE"
    df[f"{model_name}_prob_top1"] = 1.0
    df[f"{model_name}_prob_top2"] = 0.0

    template = get_ml_models(random_state)[model_name]
    ng_indices = df[df["condition"] == "NG"].index.tolist()
    ng_df = df.loc[ng_indices]

    X_all = np.array([build_features(r["weight"], r["recipe"])
                      for _, r in ng_df.iterrows()])
    y_all = ng_df["missing"].values
    recipes_all = ng_df["recipe"].values

    for recipe in ["R3", "R4", "R6"]:
        mask = recipes_all == recipe
        positions = np.where(mask)[0]
        X = X_all[mask]
        y = y_all[mask]
        if len(np.unique(y)) < 2:
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, test_idx in cv.split(X, y):
            m = type(template)(**template.get_params())
            m.fit(X[train_idx], y[train_idx])
            preds  = m.predict(X[test_idx])
            probas = m.predict_proba(X[test_idx])
            for j, ti in enumerate(test_idx):
                df_idx = ng_indices[positions[ti]]
                sp = sorted(probas[j], reverse=True)
                df.at[df_idx, f"{model_name}_pred"]      = preds[j]
                df.at[df_idx, f"{model_name}_prob_top1"] = round(float(sp[0]), 4)
                df.at[df_idx, f"{model_name}_prob_top2"] = round(float(sp[1]) if len(sp) > 1 else 0, 4)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Full benchmark — apply all 12 methods to the DOE
# ═════════════════════════════════════════════════════════════════════════════

def run_full_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 12 classifier methods to the DOE dataset.

    Parameters
    ----------
    df : DataFrame with columns [run_id, recipe, condition, missing, weight]

    Returns
    -------
    df : Same DataFrame with prediction columns added for each method.
    """
    df = df.copy()

    # Pre-compute kit parameters
    df["mu_kit"]  = df["recipe"].apply(mu_kit)
    df["sig_kit"] = df["recipe"].apply(sigma_kit)
    df["deficit"] = df["mu_kit"] - df["weight"]
    df["z"]       = df["deficit"] / df["sig_kit"]

    # ── Statistical L1 ────────────────────────────────────────────────────────
    df["l1_k3"]   = df["z"].abs().le(3.0).map({True: "OK", False: "NG"})
    df["l1_k35"]  = df["z"].abs().le(3.5).map({True: "OK", False: "NG"})
    df["l1_k6"]   = df["z"].abs().le(6.0).map({True: "OK", False: "NG"})
    df["maha_l1"] = df.apply(lambda r: l1_mahalanobis(r["weight"], r["recipe"]), axis=1)

    # ── Statistical L2 ────────────────────────────────────────────────────────
    def apply_l2(row, l1_col, fn):
        return "NONE" if row[l1_col] == "OK" else fn(row["deficit"], row["recipe"])

    df["runs10_pred"]      = df.apply(lambda r: apply_l2(r, "l1_k3", l2_runs10), axis=1)
    df["runs10_35_pred"]   = df.apply(lambda r: apply_l2(r, "l1_k35", l2_runs10), axis=1)
    df["runs10_6_pred"]    = df.apply(lambda r: apply_l2(r, "l1_k6", l2_runs10), axis=1)
    df["zscore_pred"]      = df.apply(lambda r: apply_l2(r, "l1_k3", l2_zscore), axis=1)
    df["zscore_6_pred"]    = df.apply(lambda r: apply_l2(r, "l1_k6", l2_zscore), axis=1)
    df["maha_pred"]        = df.apply(lambda r: apply_l2(r, "maha_l1", l2_runs10), axis=1)
    df["gmm_pred"]         = df.apply(lambda r: apply_l2(r, "l1_k3", l2_gmm), axis=1)
    df["bayes_pred"]       = df.apply(lambda r: apply_l2(r, "l1_k3", l2_bayes_optimal), axis=1)

    # ── ML L2 (5-fold CV) ─────────────────────────────────────────────────────
    for name in ["logistic", "lda", "qda", "svm", "random_forest"]:
        df = cross_validate_ml(df, name)

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary table of all 12 methods."""
    ng = df[df["condition"] == "NG"]

    methods = [
        ("Runs 1.0 (k=3)",       "runs10_pred",       "Statistical"),
        ("Runs 1.0 + 3.5σ",      "runs10_35_pred",    "Statistical"),
        ("Runs 1.0 + 6σ",        "runs10_6_pred",     "Statistical"),
        ("Runs Z-score",         "zscore_pred",       "Statistical"),
        ("Runs Z-score + 6σ",    "zscore_6_pred",     "Statistical"),
        ("Runs Mahalanobis",     "maha_pred",         "Statistical"),
        ("Runs GMM",             "gmm_pred",          "Probabilistic"),
        ("Runs 1.0 + Bayes",     "bayes_pred",        "Probabilistic"),
        ("★ Logistic Regression", "logistic_pred",     "ML"),
        ("★ LDA",                 "lda_pred",          "ML"),
        ("★ QDA",                 "qda_pred",          "ML"),
        ("★ SVM (RBF)",           "svm_pred",          "ML"),
        ("★ Random Forest",       "random_forest_pred","ML"),
    ]

    rows = []
    for name, col, typ in methods:
        if col not in df.columns:
            continue
        l2_global = (ng[col] == ng["missing"]).mean()
        l2_by_recipe = {
            r: ((ng[ng["recipe"] == r][col]) == ng[ng["recipe"] == r]["missing"]).mean()
            for r in ["R3", "R4", "R6"]
        }
        l2_w1 = ((ng[ng["missing"] == "W1"][col]) == "W1").mean()
        rows.append({
            "Method":    name,
            "Type":      typ,
            "L2_Global": f"{l2_global*100:.2f}%",
            "L2_R3":     f"{l2_by_recipe['R3']*100:.2f}%",
            "L2_R4":     f"{l2_by_recipe['R4']*100:.2f}%",
            "L2_R6":     f"{l2_by_recipe['R6']*100:.2f}%",
            "L2_W1":     f"{l2_w1*100:.1f}%",
        })

    return pd.DataFrame(rows).sort_values("L2_Global", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    csv_path = Path("data/doe_data.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run 'python src/extract_doe_data.py --input results/DOE_Smart_Packaging_Line_v7.xlsx'")
        sys.exit(1)

    print("Smart Packaging Line — 12-Method Benchmark")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} kits  ({(df['condition']=='OK').sum()} OK, "
          f"{(df['condition']=='NG').sum()} NG)")
    print("Running 12 methods with 5-fold CV for ML...\n")

    df_results = run_full_benchmark(df)
    summary = summarize(df_results)

    print(summary.to_string(index=False))

    # Save predictions
    output = Path("data/predictions_all_methods.csv")
    df_results.to_csv(output, index=False)
    print(f"\n✓ Predictions saved to {output}")
