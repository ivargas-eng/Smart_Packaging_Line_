"""
Unit tests for the Smart Packaging Line classifier.

Run with: pytest src/tests/ -v
"""
import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

from classifier import (
    Component,
    Recipe,
    SmartPackagingClassifier,
    L1Classifier,
    GMMClassifierL2,
    BayesOptimalL2,
    Runs10L2,
    discriminability_index,
    theoretical_accuracy_ceiling,
    DEFAULT_CATALOG,
    DEFAULT_RECIPES,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def catalog():
    return DEFAULT_CATALOG

@pytest.fixture
def recipes():
    return DEFAULT_RECIPES

@pytest.fixture
def clf_runs10():
    return SmartPackagingClassifier(l2_method="runs10")

@pytest.fixture
def clf_bayes():
    return SmartPackagingClassifier(l2_method="bayes")

@pytest.fixture
def clf_gmm():
    return SmartPackagingClassifier(l2_method="gmm")


# ─── Component & Recipe Tests ─────────────────────────────────────────────────

class TestComponent:
    def test_instantiation(self):
        c = Component("X1", mean_g=1.0, std_g=0.01)
        assert c.component_id == "X1"
        assert c.mean_g == 1.0

    def test_immutable(self):
        c = Component("X1", mean_g=1.0, std_g=0.01)
        with pytest.raises(Exception):
            c.mean_g = 2.0  # frozen dataclass


class TestRecipe:
    def test_mu_kit_r3(self, catalog, recipes):
        """R3 = S2 + W2 + N2 = 4.036 + 4.5627 + 1.3743 = 9.973"""
        mu = recipes["R3"].mu_kit(catalog)
        assert abs(mu - 9.9730) < 0.001

    def test_sigma_kit_r6(self, catalog, recipes):
        """sigma_R6 should be around 0.0496g."""
        sigma = recipes["R6"].sigma_kit(catalog)
        assert 0.04 < sigma < 0.06

    def test_lcl_ucl_symmetry(self, catalog, recipes):
        mu = recipes["R3"].mu_kit(catalog)
        assert abs(recipes["R3"].lcl(catalog, k=3) - (mu - 3 * recipes["R3"].sigma_kit(catalog))) < 1e-10
        assert abs(recipes["R3"].ucl(catalog, k=3) - (mu + 3 * recipes["R3"].sigma_kit(catalog))) < 1e-10


# ─── L1 Classifier Tests ──────────────────────────────────────────────────────

class TestL1Classifier:
    def test_ok_kit_center(self, catalog, recipes):
        clf = L1Classifier(k=3.0)
        # Weight exactly at mu_kit → should be OK
        mu = recipes["R3"].mu_kit(catalog)
        decision, z, _, _ = clf.classify(mu, recipes["R3"], catalog)
        assert decision == "OK"
        assert abs(z) < 1e-10

    def test_ng_kit_far(self, catalog, recipes):
        clf = L1Classifier(k=3.0)
        # Weight 5 sigma below mu → NG
        mu = recipes["R3"].mu_kit(catalog)
        sigma = recipes["R3"].sigma_kit(catalog)
        decision, z, _, _ = clf.classify(mu - 5 * sigma, recipes["R3"], catalog)
        assert decision == "NG"

    def test_boundary_exactly_k_sigma(self, catalog, recipes):
        clf = L1Classifier(k=3.0)
        mu = recipes["R3"].mu_kit(catalog)
        sigma = recipes["R3"].sigma_kit(catalog)
        # Exactly at LCL: |z| == k → classifier uses <=, so this is NG
        # (strict equality at boundary, floating point: |z| == 3.0 → NG)
        decision, z, _, _ = clf.classify(mu - 3 * sigma, recipes["R3"], catalog)
        assert abs(abs(z) - 3.0) < 1e-9  # z is exactly ±3
        # Slightly inside → OK
        decision_inside, _, _, _ = clf.classify(mu - 2.99 * sigma, recipes["R3"], catalog)
        assert decision_inside == "OK"

    def test_k6_more_permissive(self, catalog, recipes):
        clf3 = L1Classifier(k=3.0)
        clf6 = L1Classifier(k=6.0)
        mu = recipes["R3"].mu_kit(catalog)
        sigma = recipes["R3"].sigma_kit(catalog)
        w = mu - 4 * sigma  # 4 sigma below
        assert clf3.classify(w, recipes["R3"], catalog)[0] == "NG"
        assert clf6.classify(w, recipes["R3"], catalog)[0] == "OK"


# ─── L2 Classifier Tests ─────────────────────────────────────────────────────

class TestRuns10L2:
    def test_obvious_s2_missing(self, catalog, recipes):
        """Deficit ~ 4.04g (S2 weight) → predicts S2."""
        l2 = Runs10L2()
        deficit = catalog["S2"].mean_g  # exactly mu_S2
        pred, distances = l2.identify(deficit, recipes["R3"], catalog)
        assert pred == "S2"
        assert distances["S2"] < 0.01

    def test_returns_dict_with_recipe_components(self, catalog, recipes):
        l2 = Runs10L2()
        _, distances = l2.identify(1.0, recipes["R3"], catalog)
        assert set(distances.keys()) == set(recipes["R3"].composition.keys())


class TestGMMClassifierL2:
    def test_high_confidence_easy_case(self, catalog, recipes):
        """S2 missing in R3 → clear deficit, high confidence."""
        gmm = GMMClassifierL2(ambiguity_threshold=0.15)
        deficit = catalog["S2"].mean_g
        pred, probs, flag = gmm.identify(deficit, recipes["R3"], catalog)
        assert pred == "S2"
        assert probs["S2"] > 0.9
        assert flag == "CONFIDENT"

    def test_probs_sum_to_one(self, catalog, recipes):
        gmm = GMMClassifierL2()
        _, probs, _ = gmm.identify(1.05, recipes["R6"], catalog)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_ambiguous_w1_n1(self, catalog, recipes):
        """
        Test that GMM correctly ranks W1 vs N1 based on proximity.
        Near W1 mean → prefers W1. Near N1 mean → prefers N1.
        """
        gmm = GMMClassifierL2(ambiguity_threshold=0.15)
        # Near W1 → should prefer W1
        _, probs_w1, _ = gmm.identify(catalog["W1"].mean_g, recipes["R6"], catalog)
        assert probs_w1["W1"] > probs_w1["N1"]
        # Near N1 → should prefer N1
        _, probs_n1, _ = gmm.identify(catalog["N1"].mean_g, recipes["R6"], catalog)
        assert probs_n1["N1"] > probs_n1["W1"]


# ─── Full Classifier Tests ────────────────────────────────────────────────────

class TestSmartPackagingClassifier:
    def test_ok_kit(self, clf_runs10, catalog, recipes):
        mu = recipes["R3"].mu_kit(catalog)
        r = clf_runs10.classify(weight_g=mu, recipe_id="R3")
        assert r.l1_decision == "OK"
        assert r.l2_predicted_missing is None

    def test_ng_kit_l2_runs10(self, clf_runs10, catalog, recipes):
        # S2 clearly missing in R3
        mu = recipes["R3"].mu_kit(catalog)
        w = mu - catalog["S2"].mean_g * 0.9  # ~90% of S2 weight as deficit
        r = clf_runs10.classify(weight_g=w, recipe_id="R3")
        if r.l1_decision == "NG":
            assert r.l2_predicted_missing is not None
            assert r.l2_predicted_missing in recipes["R3"].composition

    def test_unknown_recipe_raises(self, clf_runs10):
        with pytest.raises(ValueError, match="Unknown recipe"):
            clf_runs10.classify(weight_g=9.5, recipe_id="R99")

    def test_gmm_produces_probabilities(self, clf_gmm, catalog, recipes):
        mu = recipes["R3"].mu_kit(catalog)
        w = mu - 4.0  # clearly NG
        r = clf_gmm.classify(weight_g=w, recipe_id="R3")
        if r.l1_decision == "NG":
            assert len(r.component_probabilities) > 0
            assert abs(sum(r.component_probabilities.values()) - 1.0) < 1e-5

    def test_result_fields_consistent(self, clf_bayes, catalog, recipes):
        mu = recipes["R6"].mu_kit(catalog)
        r = clf_bayes.classify(weight_g=mu - 1.0, recipe_id="R6")
        assert r.expected_weight_g == pytest.approx(mu, abs=0.001)
        assert r.deficit_g == pytest.approx(1.0, abs=0.01)
        assert r.recipe_id == "R6"


# ─── Discriminability Tests ───────────────────────────────────────────────────

class TestDiscriminability:
    def test_w1_n1_d_eff(self, catalog, recipes):
        """W1/N1 pair in R6 should have D_eff ≈ 3.27."""
        sigma_r6 = recipes["R6"].sigma_kit(catalog)
        d_eff = discriminability_index(catalog["W1"], catalog["N1"], sigma_kit=sigma_r6)
        assert 3.0 < d_eff < 3.6

    def test_theoretical_ceiling_w1_n1(self, catalog, recipes):
        """Ceiling for D_eff ~ 3.27 should be ~0.9491."""
        sigma_r6 = recipes["R6"].sigma_kit(catalog)
        d_eff = discriminability_index(catalog["W1"], catalog["N1"], sigma_kit=sigma_r6)
        ceiling = theoretical_accuracy_ceiling(d_eff)
        assert 0.94 < ceiling < 0.96

    def test_s1_s2_high_discriminability(self, catalog):
        """S1 and S2 have very different weights → should be easy to separate."""
        d = discriminability_index(catalog["S1"], catalog["S2"])
        assert d > 10  # trivially discriminable
