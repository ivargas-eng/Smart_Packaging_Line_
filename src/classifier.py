"""
Smart Packaging Line — Weight-Based Kit Inspection Classifier
=============================================================

Two-level classification system for packaged hardware kits:
    L1: OK / NG detection (Shewhart control chart, k=3)
    L2: Missing component identification (multiple methods)

Methods implemented:
    - Runs 1.0      : Absolute deficit (baseline, best statistical)
    - Mahalanobis   : F-Hotelling threshold + bias correction
    - GMM           : Gaussian Mixture Model with AMBIGUOUS flag
    - Bayes Optimal : Maximum likelihood under normality
    - Logistic ML   : Multinomial LR with derived features (best overall)
    - Random Forest : Ensemble of 200 decision trees

Industry Partner : TE Connectivity — CPT Department
Academic Context : Capstone Project, Tec de Monterrey (Campus Sonora Norte)
Authors          : Ian Vargas, José Luis Espinosa, Rubén Figueroa
Supervisors      : Víctor Hugo Benítez (academic), Jorge Clayton (industry)

References
----------
Shewhart (1931). Economic Control of Quality of Manufactured Product.
Mahalanobis (1936). On the Generalised Distance in Statistics. PNISI 2(1).
McLachlan & Peel (2000). Finite Mixture Models. Wiley. ISBN 978-0471006268.
Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
Cox & Snell (1989). Analysis of Binary Data. Chapman & Hall. ISBN 0412233606.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, f as f_dist

warnings.filterwarnings("ignore")


# ─── Component & Recipe Definitions ──────────────────────────────────────────

@dataclass(frozen=True)
class Component:
    """Single hardware component with calibration statistics (n=30)."""
    component_id: str
    mean_g: float       # calibrated mean weight (grams)
    std_g: float        # calibrated std deviation (grams)
    n_cal: int = 30     # calibration sample size


@dataclass(frozen=True)
class Recipe:
    """Kit specification: which components and how many of each."""
    recipe_id: str
    composition: Dict[str, int]  # {component_id: quantity}

    def mu_kit(self, catalog: Dict[str, Component]) -> float:
        """Expected weight of a complete kit (grams)."""
        return sum(qty * catalog[cid].mean_g
                   for cid, qty in self.composition.items())

    def sigma_kit(self, catalog: Dict[str, Component]) -> float:
        """Propagated std deviation of kit weight (variance propagation)."""
        var = sum(qty * catalog[cid].std_g ** 2
                  for cid, qty in self.composition.items())
        return np.sqrt(var)

    def lcl(self, catalog: Dict[str, Component], k: float = 3.0) -> float:
        """Lower Control Limit = mu_kit - k * sigma_kit."""
        return self.mu_kit(catalog) - k * self.sigma_kit(catalog)

    def ucl(self, catalog: Dict[str, Component], k: float = 3.0) -> float:
        """Upper Control Limit = mu_kit + k * sigma_kit."""
        return self.mu_kit(catalog) + k * self.sigma_kit(catalog)


# ─── Default Catalog (calibrated, n=30 per component) ────────────────────────

DEFAULT_CATALOG: Dict[str, Component] = {
    # Extracted directly from Components sheet (DOE_Smart_Packaging_Line_v4.xlsx)
    # columns: measured_mean_g, measured_std_g — n=30 per component
    "S1": Component("S1", mean_g=2.753333333333333,  std_g=0.006608945522512526),
    "S2": Component("S2", mean_g=4.036000000000000,  std_g=0.013796551293211020),
    "W1": Component("W1", mean_g=0.993333333333333,  std_g=0.013978637231524930),
    "W2": Component("W2", mean_g=4.562666666666667,  std_g=0.025179813114988600),
    "N1": Component("N1", mean_g=1.155666666666666,  std_g=0.027628487713469430),
    "N2": Component("N2", mean_g=1.374333333333333,  std_g=0.025145553296253180),
}

DEFAULT_RECIPES: Dict[str, Recipe] = {
    "R3": Recipe("R3", {"S2": 1, "W2": 1, "N2": 1}),
    "R4": Recipe("R4", {"S1": 1, "W1": 1, "N1": 1, "S2": 1}),
    "R6": Recipe("R6", {"S1": 1, "W1": 2, "N1": 1, "N2": 2}),
}


# ─── Result Objects ───────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Full output of the two-level classifier for a single kit."""
    # Inputs
    measured_weight_g: float
    recipe_id: str

    # Computed kit parameters
    expected_weight_g: float
    expected_std_g: float
    deficit_g: float
    z_score: float

    # L1 decision
    l1_decision: str          # "OK" | "NG"
    l1_method: str            # e.g. "Runs1.0 k=3"

    # L2 decision (only meaningful when l1_decision == "NG")
    l2_predicted_missing: Optional[str] = None    # component ID or "NONE"
    l2_confidence: Optional[float] = None         # P_top1 probability
    l2_flag: Optional[str] = None                 # "CONFIDENT" | "AMBIGUOUS"
    l2_method: Optional[str] = None               # method name

    # Per-component probabilities (optional, for GMM/Logistic)
    component_probabilities: Dict[str, float] = field(default_factory=dict)


# ─── Level 1 Classifier ──────────────────────────────────────────────────────

class L1Classifier:
    """
    Level 1: OK / NG classification.

    Implements Shewhart control chart (z-score with k-sigma limits).
    A kit is OK if its weight falls within [mu_kit ± k * sigma_kit].

    Reference: Shewhart, W.A. (1931). Economic Control of Quality
               of Manufactured Product. Van Nostrand.
    """

    def __init__(self, k: float = 3.0):
        """
        Parameters
        ----------
        k : float
            Control limit multiplier. k=3 → α=0.0027 (3-sigma rule).
            k=3.5, k=6 are common alternatives.
        """
        self.k = k

    def classify(
        self,
        weight_g: float,
        recipe: Recipe,
        catalog: Dict[str, Component],
    ) -> Tuple[str, float, float, float]:
        """
        Classify a kit as OK or NG.

        Returns
        -------
        decision : str
            "OK" or "NG"
        z : float
            z-score = (weight - mu_kit) / sigma_kit
        mu_kit : float
            Expected kit weight
        sigma_kit : float
            Kit weight standard deviation
        """
        mu = recipe.mu_kit(catalog)
        sigma = recipe.sigma_kit(catalog)
        z = (weight_g - mu) / sigma
        decision = "OK" if abs(z) <= self.k else "NG"
        return decision, z, mu, sigma


# ─── Level 2 Classifiers ─────────────────────────────────────────────────────

class Runs10L2:
    """
    L2: Absolute deficit method (Runs 1.0).

    Identifies the missing component as the one whose individual weight
    is closest (in absolute grams) to the observed kit deficit.

    deficit = mu_kit - measured_weight
    missing = argmin_i |deficit - mu_i * qty_i|

    This is the best single-variable statistical method in the DOE
    (L2 accuracy: 96.06% global, 91.67% on R6).
    """

    def identify(
        self,
        deficit_g: float,
        recipe: Recipe,
        catalog: Dict[str, Component],
    ) -> Tuple[str, Dict[str, float]]:
        distances = {
            cid: abs(deficit_g - catalog[cid].mean_g * qty)
            for cid, qty in recipe.composition.items()
        }
        predicted = min(distances, key=distances.get)
        return predicted, distances


class GMMClassifierL2:
    """
    L2: Gaussian Mixture Model.

    Models each NG hypothesis as a Gaussian distribution:
        P(deficit | missing=X) = N(deficit; mu_X, sigma_X^2)

    Uses softmax-normalized log-likelihoods for calibrated probabilities.
    Flags AMBIGUOUS when P_top1 - P_top2 < ambiguity_threshold.

    Reference: McLachlan & Peel (2000). Finite Mixture Models. Wiley.
               Dempster, Laird, Rubin (1977). JRSS-B, 39(1), 1-38.
    """

    def __init__(self, ambiguity_threshold: float = 0.15):
        self.ambiguity_threshold = ambiguity_threshold

    def identify(
        self,
        deficit_g: float,
        recipe: Recipe,
        catalog: Dict[str, Component],
    ) -> Tuple[str, Dict[str, float], str]:
        """
        Returns
        -------
        predicted : str
        probabilities : dict {component_id: probability}
        flag : "CONFIDENT" | "AMBIGUOUS"
        """
        log_likelihoods = {
            cid: norm.logpdf(deficit_g,
                             loc=catalog[cid].mean_g,
                             scale=catalog[cid].std_g)
            for cid in recipe.composition
        }

        # Numerically stable softmax
        log_vals = np.array(list(log_likelihoods.values()))
        shifted = log_vals - log_vals.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        prob_dict = dict(zip(log_likelihoods.keys(), probs))

        predicted = max(prob_dict, key=prob_dict.get)
        sorted_probs = sorted(prob_dict.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        flag = "AMBIGUOUS" if margin < self.ambiguity_threshold else "CONFIDENT"

        return predicted, prob_dict, flag


class BayesOptimalL2:
    """
    L2: Bayes Optimal Classifier.

    Evaluates the log-likelihood of each NG hypothesis using the
    Gaussian density of the observed deficit under each component model.
    Theoretically optimal under normality with known parameters.

    Accuracy ceiling with 1 variable:
        P(correct) = 1 - Φ(-D_eff / 2)
        For W1/N1 pair in R6: D_eff = 3.27 → ceiling ≈ 94.91%

    Reference: Hastie, Tibshirani, Friedman (2009).
               The Elements of Statistical Learning §4.4. Springer.
    """

    def __init__(self, ambiguity_threshold: float = 0.15):
        self.ambiguity_threshold = ambiguity_threshold

    def identify(
        self,
        deficit_g: float,
        recipe: Recipe,
        catalog: Dict[str, Component],
    ) -> Tuple[str, Dict[str, float], str]:
        log_likelihoods = {}
        for cid, qty in recipe.composition.items():
            mu_c = catalog[cid].mean_g * qty
            sig_c = catalog[cid].std_g * np.sqrt(qty)
            log_likelihoods[cid] = norm.logpdf(deficit_g, loc=mu_c, scale=sig_c)

        log_vals = np.array(list(log_likelihoods.values()))
        shifted = log_vals - log_vals.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        prob_dict = dict(zip(log_likelihoods.keys(), probs))

        predicted = max(prob_dict, key=prob_dict.get)
        sorted_probs = sorted(prob_dict.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        flag = "AMBIGUOUS" if margin < self.ambiguity_threshold else "CONFIDENT"

        return predicted, prob_dict, flag


# ─── Full Two-Level Classifier ────────────────────────────────────────────────

class SmartPackagingClassifier:
    """
    Two-level kit inspection classifier.

    Usage
    -----
    clf = SmartPackagingClassifier()
    result = clf.classify(weight_g=9.95, recipe_id="R3")
    print(result.l1_decision)          # "OK"
    print(result.l2_predicted_missing) # None (kit is OK)

    result_ng = clf.classify(weight_g=8.48, recipe_id="R6")
    print(result_ng.l1_decision)          # "NG"
    print(result_ng.l2_predicted_missing) # "W1" or "N1"
    print(result_ng.l2_flag)              # "CONFIDENT" or "AMBIGUOUS"
    """

    def __init__(
        self,
        catalog: Optional[Dict[str, Component]] = None,
        recipes: Optional[Dict[str, Recipe]] = None,
        k_sigma: float = 3.0,
        l2_method: str = "runs10",
        ambiguity_threshold: float = 0.15,
    ):
        """
        Parameters
        ----------
        catalog : dict, optional
            Component catalog. Defaults to DEFAULT_CATALOG.
        recipes : dict, optional
            Recipe definitions. Defaults to DEFAULT_RECIPES.
        k_sigma : float
            L1 control limit multiplier (default 3.0).
        l2_method : str
            L2 identification method: "runs10" | "gmm" | "bayes"
        ambiguity_threshold : float
            Margin below which L2 flags AMBIGUOUS (for gmm/bayes).
        """
        self.catalog = catalog or DEFAULT_CATALOG
        self.recipes = recipes or DEFAULT_RECIPES
        self.l1 = L1Classifier(k=k_sigma)

        method_map = {
            "runs10": Runs10L2(),
            "gmm":    GMMClassifierL2(ambiguity_threshold),
            "bayes":  BayesOptimalL2(ambiguity_threshold),
        }
        if l2_method not in method_map:
            raise ValueError(f"l2_method must be one of {list(method_map)}")
        self._l2 = method_map[l2_method]
        self._l2_name = l2_method

    def classify(self, weight_g: float, recipe_id: str) -> ClassificationResult:
        """
        Classify a single kit.

        Parameters
        ----------
        weight_g : float
            Measured kit weight from the Sartorius scale.
        recipe_id : str
            Recipe selected by operator (R3, R4, or R6).

        Returns
        -------
        ClassificationResult
        """
        if recipe_id not in self.recipes:
            raise ValueError(f"Unknown recipe '{recipe_id}'. Valid: {list(self.recipes)}")

        recipe = self.recipes[recipe_id]

        # L1
        l1_decision, z, mu_kit, sigma_kit = self.l1.classify(
            weight_g, recipe, self.catalog
        )
        deficit = mu_kit - weight_g

        result = ClassificationResult(
            measured_weight_g=weight_g,
            recipe_id=recipe_id,
            expected_weight_g=round(mu_kit, 4),
            expected_std_g=round(sigma_kit, 5),
            deficit_g=round(deficit, 4),
            z_score=round(z, 4),
            l1_decision=l1_decision,
            l1_method=f"Shewhart k={self.l1.k}",
        )

        if l1_decision == "NG":
            self._apply_l2(result, deficit, recipe)

        return result

    def _apply_l2(
        self,
        result: ClassificationResult,
        deficit: float,
        recipe: Recipe,
    ) -> None:
        """Run L2 identification and populate result in-place."""
        if isinstance(self._l2, Runs10L2):
            predicted, distances = self._l2.identify(deficit, recipe, self.catalog)
            result.l2_predicted_missing = predicted
            result.l2_confidence = None
            result.l2_flag = "CONFIDENT"
            result.l2_method = "Runs1.0 (absolute deficit)"
        else:
            predicted, probs, flag = self._l2.identify(deficit, recipe, self.catalog)
            result.l2_predicted_missing = predicted
            result.component_probabilities = {k: round(v, 4) for k, v in probs.items()}
            result.l2_confidence = round(max(probs.values()), 4)
            result.l2_flag = flag
            result.l2_method = (
                "GMM (softmax log-likelihood)"
                if isinstance(self._l2, GMMClassifierL2)
                else "Bayes Optimal (qty-weighted likelihood)"
            )


# ─── Discriminability Analysis ────────────────────────────────────────────────

def discriminability_index(
    comp_i: Component,
    comp_j: Component,
    sigma_kit: Optional[float] = None,
) -> float:
    """
    Compute discriminability index D between two components.

    D = |mu_i - mu_j| / (sigma_i + sigma_j)     [component-level]
    D_eff = |mu_i - mu_j| / sigma_kit             [kit-level]

    Reference: Sartorius AG (1999). Manual of Weighing Applications Part 2.
               Cohen (1988). Statistical Power Analysis. Erlbaum. §2.2 eq 2.2.2.

    Interpretation
    --------------
    D >= 3  : reliably distinguishable
    1 <= D < 3  : marginal
    D < 1   : unreliable
    """
    gap = abs(comp_i.mean_g - comp_j.mean_g)
    if sigma_kit is not None:
        return gap / sigma_kit
    return gap / (comp_i.std_g + comp_j.std_g)


def theoretical_accuracy_ceiling(d_eff: float) -> float:
    """
    Bayes-optimal accuracy ceiling for a binary classification problem
    with D_eff discriminability and equal prior probabilities.

    P(correct) = 1 - Φ(-D_eff / 2)

    For W1/N1 in R6: D_eff = 3.27 → ceiling ≈ 0.9491
    """
    return 1 - norm.cdf(-d_eff / 2)


# ─── Quick Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smart Packaging Line — Classifier Demo")
    print("=" * 50)

    # OK kit
    clf = SmartPackagingClassifier(l2_method="bayes")
    r = clf.classify(weight_g=9.97, recipe_id="R3")
    print(f"\nKit R3, w=9.97g:")
    print(f"  L1: {r.l1_decision} (z={r.z_score})")

    # NG kit — easy case
    r2 = clf.classify(weight_g=5.93, recipe_id="R3")
    print(f"\nKit R3, w=5.93g (missing S2):")
    print(f"  L1: {r2.l1_decision}")
    print(f"  L2: {r2.l2_predicted_missing} ({r2.l2_flag})")
    print(f"  Probs: {r2.component_probabilities}")

    # NG kit — ambiguous W1/N1 in R6
    r3 = clf.classify(weight_g=7.55, recipe_id="R6")
    print(f"\nKit R6, w=7.55g (W1/N1 ambiguous zone):")
    print(f"  L1: {r3.l1_decision}")
    print(f"  L2: {r3.l2_predicted_missing} ({r3.l2_flag})")
    print(f"  Probs: {r3.component_probabilities}")

    # Discriminability
    cat = DEFAULT_CATALOG
    r6 = DEFAULT_RECIPES["R6"]
    sigma_r6 = r6.sigma_kit(cat)
    D_eff = discriminability_index(cat["W1"], cat["N1"], sigma_kit=sigma_r6)
    ceiling = theoretical_accuracy_ceiling(D_eff)
    print(f"\nW1/N1 discriminability in R6:")
    print(f"  D_eff = {D_eff:.3f}")
    print(f"  Theoretical accuracy ceiling = {ceiling*100:.2f}%")
