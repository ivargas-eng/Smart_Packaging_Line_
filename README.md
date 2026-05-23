# Smart Packaging Line
### Weight-Based Automated Kit Inspection — 12-Method Benchmark — Tecnológico de Monterrey x TE Connectivity

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?logo=scikit-learn)
![Methods](https://img.shields.io/badge/Methods-12_Benchmarked-purple)
![Best](https://img.shields.io/badge/Best_L2-97.58%25_(QDA)-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This repository contains the full statistical and machine learning pipeline for an automated final inspection system for a two-level classification pipeline at **TE Connectivity (CPT Department)**.
The system uses precision weight measurement (Sartorius scale, 0.01g resolution, RS-232) to:

- **L1 — OK / NG**: Detect whether the kit is complete
- **L2 — Component ID**: When NG, identify which component is missing

Twelve methods are benchmarked: 8 statistical/probabilistic + 5 machine learning, with full 5-fold cross-validation on the real DOE dataset (420 kits).

The work is part of a **Mechatronics Engineering Capstone** at [Tec de Monterrrey](https://tec.mx), in collaboration with [TE Connectivity](https://www.te.com).

---

## Results — All 12 Methods

| Rank | Method | Type | L2 Global | R6 | W1 |
|------|--------|------|-----------|-----|-----|
| 1 | **QDA** | ML | **97.58%** | **95.0%** | **91.7%** |
| 2 | Logistic Regression | ML | 97.27% | 93.3% | 88.3% |
| 3 | SVM (RBF) | ML | 96.97% | 93.3% | 88.3% |
| 4 | LDA | ML | 96.67% | 92.5% | 86.7% |
| 5 | Random Forest | ML | 96.67% | 92.5% | 91.7% |
| 6 | Mahalanobis | Statistical | 96.06% | 91.7% | 85.0% |
| 7 | Runs 1.0 (k=3) | Statistical | 96.06% | 91.7% | 85.0% |
| 8-9 | Runs 1.0 (+3.5σ/+6σ) | Statistical | 96.06% | 91.7% | 85.0% |
| 10 | Bayes Optimal | Probabilistic | 94.24% | 88.3% | 73.3% |
| 11 | GMM | Probabilistic | 93.64% | 88.3% | 73.3% |
| 12 | Runs Z-score | Statistical | 93.64% | 88.3% | 73.3% |

> **Zero false negatives across all 12 methods.** No defective kit is ever classified as OK.

---

## The Scientific Story

**Phase 1 — Classical (Methods 1-9):** Runs 1.0 with absolute deficit (96.06%) is the best classical method. Mahalanobis adds rigor with F-Hotelling threshold and bias correction (eq. 4.11/4.12 of the 1936 paper) but produces the same L2 because **with 1 variable, Mahalanobis reduces to z-score**.

**Phase 2 — Probabilistic (Methods 10-11):** GMM and Bayes Optimal underperform because σ-weighting penalizes components with small sigma (W1, σ=0.014g) too heavily.

**Phase 3 — Machine Learning (Methods 9-13):** All ML methods use a 16-dimensional feature vector derived from the single weight measurement:

```
4 global features:  weight, deficit, z-score, weight_norm
12 per-component:   dist_X = (deficit - mu_X) / sigma_X        [signed normalized]
                    absdist_X = |deficit - mu_X|                [raw absolute]
```

QDA wins because it learns a separate covariance matrix per class — capturing different variances of W1 vs N1 in feature space.

---

## The Physical Ceiling

Hardest pair: **W1 (0.993g) vs N1 (1.156g)** in recipe R6:

```
Weight gap:       0.163 g
Kit-level σ (R6): 0.050 g
D_eff:            3.27

Bayes theoretical ceiling (1 variable) = 1 - Phi(-D_eff/2) ≈ 94.91%
```

QDA and Logistic **surpass this 1-variable ceiling (94.91%)** because their 16-feature space exploits geometric structure invisible to the single-variable bound.

---

## Dataset

| Parameter | Value |
|-----------|-------|
| Total kits | 420 (90 OK + 330 NG) |
| Recipes | R3 (3 components), R4 (4), R6 (6) |
| Components | S1, S2, W1, W2, N1, N2 |
| Calibration | n = 30 per component |
| Replicas per NG condition | 30 |

**Component catalog** (extracted from `Components` sheet):

| ID | Mean (g) | Std (g) |
|----|----------|---------|
| S1 | 2.75333 | 0.00661 |
| S2 | 4.03600 | 0.01380 |
| W1 | 0.99333 | 0.01398 |
| W2 | 4.56267 | 0.02518 |
| N1 | 1.15567 | 0.02763 |
| N2 | 1.37433 | 0.02515 |

---

## Repository Structure

```
smart-packaging-line/
├── src/
│   ├── classifier.py              # Object-oriented API (statistical)
│   ├── ml_classifiers.py          # ML methods
│   ├── all_methods_benchmark.py   # Runs all 12 methods
│   ├── extract_doe_data.py        # Excel → CSV
│   └── tests/test_classifier.py   # Unit tests (22 passing)
├── data/
│   ├── doe_data.csv               # 420 kits from Excel
│   └── predictions_all_methods.csv
├── notebooks/
│   └── Smart_Packaging_8Models_Analysis.ipynb
├── results/
│   └── DOE_Smart_Packaging_Line_v7.xlsx
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/ivargas-eng/smart-packaging-line.git
cd smart-packaging-line
pip install -r requirements.txt

# Run the 12-method benchmark
python src/all_methods_benchmark.py

# Run unit tests
pytest src/tests/ -v
```

---

## Validation

- **Statistical methods**: Deterministic formulas evaluated on all 420 kits.
- **ML methods**: 5-fold Stratified Cross-Validation per recipe.
- Each kit predicted by a model trained on data that excluded that kit (no data leakage).
- `random_state=42`, `class_weight='balanced'` throughout.
- Full hyperparameter documentation in `src/all_methods_benchmark.py:get_ml_models()`.

---

## References

1. Shewhart, W.A. (1931). *Economic Control of Quality of Manufactured Product*. Van Nostrand.
2. Mahalanobis, P.C. (1936). On the Generalised Distance in Statistics. *PNISI*, 2(1), 49–55.
3. Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. *Ann. Eugenics*, 7(2), 179–188.
4. Cox, D.R., Snell, E.J. (1989). *Analysis of Binary Data* (2nd ed.). Chapman & Hall. ISBN 0412233606.
5. Vapnik, V.N. (1995). *The Nature of Statistical Learning Theory*. Springer. ISBN 978-0387987804.
6. McLachlan, G.J., Peel, D. (2000). *Finite Mixture Models*. Wiley. ISBN 978-0471006268.
7. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324.
8. Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. ISBN 978-0387848587.
9. Agresti, A. (2002). *Categorical Data Analysis* (2nd ed.). Wiley. ISBN 978-0471360933.
10. Montgomery, D.C. (2009). *Introduction to Statistical Quality Control* (6th ed.). Wiley.
11. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Erlbaum.
12. Sartorius AG (1999). *Manual of Weighing Applications, Part 2: Counting*. Göttingen.

---

## Team

| Name | Role |
|------|------|
| Ian Vargas | Statistical engine, software, methodology, ML benchmark |
| José Luis Espinosa Mora | Mechanical design, JAKA cobot integration |
| Rubén Figueroa Fuentes | DOE validation, statistical testing |

**Academic Advisor**: Víctor Hugo Benítez  
**Industry Advisor**: Jorge Clayton — Sr. Manager, TE Connectivity CPT MX

---

## License

MIT License. See [LICENSE](LICENSE).

> *"Chakalitos Engineering Solutions minus 2."*
