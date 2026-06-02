# Black Swan HiggsML — Master Class on Machine Learning for Particle Physics

## Overview

This project is a **hands-on introduction to Machine Learning** built around a real particle physics problem: measuring the Higgs boson signal strength in the $H \to \tau\tau$ (Higgs to two tau leptons) decay channel using simulated data from the ATLAS experiment at CERN's Large Hadron Collider.

The challenge: given 2 million simulated proton-proton collision events, each described by ~28 features, build a complete analysis pipeline that estimates **$\mu$** — the ratio of observed Higgs production to the Standard Model prediction ($\mu = 1$ means exactly the Standard Model).

## Two Repositories

```
CentralSupelec_course/
├── black_swan_pkg/           ← Competition infrastructure (read-only)
│   └── HiggsML/
│       ├── datasets.py       ← Data loading & download
│       ├── ingestion.py      ← Competition pipeline
│       ├── systematics.py    ← Physics systematics engine
│       ├── derived_quantities.py ← Feature engineering
│       ├── score.py          ← Competition metrics
│       └── visualization.py  ← Plotting tools
│
├── Higgs_collaborations/     ← Your workspace
│   ├── StartingKit_Black_Swan_HiggsML.ipynb  ← Main notebook
│   └── sample_code_submission/
│       ├── model.py                     ← Orchestrator (runs your code)
│       ├── boosted_decision_tree.py     ← GROUP 1
│       ├── neural_network.py            ← GROUP 2
│       ├── feature_analysis.py          ← GROUP 3
│       ├── statistical_analysis.py      ← GROUP 4
│       ├── systematic_analysis.py       ← GROUP 5
│       ├── sample_model.py              ← Reference / inspiration
│       └── utils.py                     ← Shared utilities
│
└── README.md                 ← This document
```

---

## Physics Context

### What are we measuring?

The Large Hadron Collider smashes protons together at 13 TeV. Among the debris, a Higgs boson can be produced and decay into two tau leptons ($H \to \tau\tau$). One tau decays to a lepton (electron or muon), the other to hadrons. This gives a distinctive signature:

- **1 charged lepton** (electron or muon)
- **1 hadronic tau** (narrow jet of hadrons)
- **Missing transverse energy** (MET) from undetected neutrinos
- **0–2 additional jets** from the proton collision

### The data

Each event is one row with **~28 features**:

| Category | Features | Physics meaning |
|---|---|---|
| **PRI_** (primary) | `PRI_lep_pt`, `PRI_lep_eta`, `PRI_lep_phi`, `PRI_had_pt`, `PRI_had_eta`, `PRI_had_phi`, `PRI_met`, `PRI_met_phi`, `PRI_jet_leading_pt/eta/phi`, `PRI_jet_subleading_pt/eta/phi`, `PRI_jet_all_pt`, `PRI_n_jets` | Raw detector measurements |
| **DER_** (derived) | `DER_mass_vis`, `DER_mass_transverse_met_lep`, `DER_mass_jet_jet`, `DER_deltaeta_jet_jet`, `DER_pt_h`, `DER_pt_tot`, `DER_sum_pt`, `DER_pt_ratio_lep_had`, `DER_deltar_had_lep`, `DER_met_phi_centrality`, `DER_lep_eta_centrality`, `DER_prodeta_jet_jet` | Physically meaningful combinations computed from PRI features |

### Composition — extreme class imbalance

| Physics process | Role | Events (train) | Average weight |
|---|---|---|---|
| $H \to \tau\tau$ | **Signal** | 463,000 | 0.0015 |
| $Z \to \tau\tau$ | Main background | 894,000 | 0.112 |
| $t\bar{t}$ | Background | 39,000 | 0.112 |
| Diboson (WW/ZZ/WZ) | Background | 3,400 | 0.112 |

**Signal is only 0.15% of the total by weight.** The weights encode production cross-sections — the signal process is intrinsically much rarer than backgrounds. A classifier that ignores weights and optimizes for raw accuracy will fail completely.

### What is $\mu$?

$\mu$ is the **signal strength modifier** — it multiplies the expected number of Higgs events:

$$N_{\text{expected}} = \mu \times N_{\text{signal}}^{\text{SM}} + N_{\text{background}}$$

- $\mu = 1$ → Standard Model prediction
- $\mu = 0$ → No Higgs signal (background-only)
- $\mu > 1$ → More Higgs than predicted (possible new physics)

The challenge is to estimate $\mu$ and its uncertainty from the data, handling both **statistical uncertainty** (finite data) and **systematic uncertainty** (detector calibration, background modeling).

---

## Class Organization — 5 Groups, 5 Scripts

The class is organized into **5 groups**, each responsible for one script in `sample_code_submission/`. These scripts are composed together by `model.py` (which you should **not modify** — it handles evaluation). The only exception is `model.py` lines 5–6, where you can toggle between `BDT = True` and `NN = True` to select which classifier to use.

Two more files are provided for reference:

| File | Role | Who modifies it |
|---|---|---|
| `model.py` | Orchestrator — loads data, calls your code, returns predictions | Do NOT modify (except lines 5-6 to toggle BDT/NN) |
| `utils.py` | Visualization helpers (histograms, ROC curves) | Shared utility |
| `sample_model.py` | Minimal baseline — returns `DER_mass_vis` as the score | Reference only |

### Group 1: `boosted_decision_tree.py` — The XGBoost Classifier

> **Physics goal:** Build a classifier that distinguishes Higgs signal from background events using gradient-boosted decision trees.

**Current state:** A basic `XGBClassifier` with default hyperparameters. It scales features with `StandardScaler` and calls `predict_proba` to output a score between 0 and 1.

**What to improve:**
- Optimize hyperparameters (learning rate, max depth, number of estimators, regularization)
- Add early stopping with a validation set
- **Fix the weight bug:** Line 22 passes `weights=weights` but XGBoost expects `sample_weight=weights` — currently weights are silently ignored
- Experiment with `scale_pos_weight` as an alternative to manual rebalancing
- Consider feature importance analysis to understand which features drive the classification

### Group 2: `neural_network.py` — The Keras/TensorFlow Classifier

> **Physics goal:** Build a deep neural network classifier, exploring architecture choices and training strategies for this physics problem.

**Current state:** A small feedforward network (10→10→1) with ReLU activations, trained for 5 epochs with binary cross-entropy loss. Weights are correctly passed via `sample_weight`.

**What to improve:**
- Design a better architecture (more layers, different widths, dropout, batch normalization)
- Add a validation split for monitoring overfitting
- Tune learning rate, optimizer, and number of epochs
- Replace `metrics=["accuracy"]` (meaningless for 99.9% background) with AUC or a custom metric
- Consider 1D-CNNs or attention mechanisms for jet structure
- Save and load trained models to avoid retraining

### Group 3: `feature_analysis.py` — Feature Engineering

> **Physics goal:** Identify the most discriminating features for Higgs detection and create new derived quantities that improve sensitivity.

**Current state:** Three stub functions that do almost nothing:
- `feature_correlations(data)` — empty
- `systematics_dependence(data)` — empty
- `minimal_dependent_features(data)` — just returns all columns

**What to implement:**
- Compute feature correlations with the signal/background labels
- Identify which features are most sensitive to systematic uncertainties (TES, JES)
- Create new physics-motivated features beyond the provided DER quantities
- Implement feature selection to reduce dimensionality
- Study how features behave differently for events with 0, 1, or 2 jets
- Handle the **-25 sentinel value** (meaning "no jet") intelligently — it should not be treated as a physical value

**Hint:** The `derived_quantities.py` module in `black_swan_pkg/HiggsML/` shows how the existing DER features are computed from 4-momentum conservation laws.

### Group 4: `statistical_analysis.py` — The Mu Estimator

> **Physics goal:** Convert classifier scores into a measurement of $\mu$ with proper statistical uncertainty. This is the final step of the analysis — everything else is preparation for this.

**Current state:** A simple **counting experiment**:
1. Apply a hard threshold at score > 0.5 (event is "signal-like" or "background-like")
2. Compute $s = \sum w_i \cdot \mathbb{1}[\text{score}_i > 0.5] \cdot \mathbb{1}[\text{label}_i = 1]$ (weighted signal count)
3. Compute $b = \sum w_i \cdot \mathbb{1}[\text{score}_i > 0.5] \cdot \mathbb{1}[\text{label}_i = 0]$ (weighted background count)
4. $\hat{\mu} = (s - b_{\text{holdout}}) / s_{\text{holdout}}$ (subtract background estimated from holdout set)
5. Uncertainty: $\Delta\mu_{\text{stat}} = \sqrt{b_{\text{holdout}} + s_{\text{holdout}}} / s_{\text{holdout}}$, systematic uncertainty is **identically zero**

**What to improve (3 levels of increasing sophistication):**

*Level 1 — Better counting:* Use a continuous score rather than a hard threshold. Optimize the threshold to minimize expected uncertainty.

*Level 2 — Profile likelihood fit:* Construct a binned likelihood function where each bin has Poisson statistics:
$$\mathcal{L}(\mu, \theta) = \prod_{\text{bins}} \text{Poisson}(n_i \mid \mu s_i(\theta) + b_i(\theta)) \cdot \text{constraint}(\theta)$$
Use `iminuit` (already installed) to find $\hat{\mu}$ by minimizing $-\ln\mathcal{L}$, and extract uncertainties from the likelihood profile. This is the standard method in particle physics.

*Level 3 — Systematic uncertainties:* Extend the likelihood to include nuisance parameters $\theta$ (TES, JES, etc.) and their constraints. The fit then profiles over both $\mu$ and $\theta$, automatically propagating systematic uncertainties.

**Key function signatures:**
- `compute_mu(score, weight, saved_info)` → `{"mu_hat", "del_mu_stat", "del_mu_sys", "del_mu_tot"}`
- `calculate_saved_info(model, holdout_set)` → `{"beta", "gamma", "tes_fit", "jes_fit"}`

### Group 5: `systematic_analysis.py` — Systematic Uncertainty Modeling

> **Physics goal:** Quantify how detector calibration uncertainties affect the classifier output, enabling Group 4 to include systematic uncertainties in the final $\mu$ measurement.

**Current state:** Stub functions `tes_fitter()` and `jes_fitter()` that return dummy fit functions (one of which would crash if called because `f` is undefined).

**What to implement:**

*Task 1 — Characterize systematic shifts:*
1. Loop over different values of the nuisance parameter (e.g., TES from 0.99 to 1.01)
2. Apply the systematic shift using `systematics(train_set, tes=value)`
3. Get the classifier score distribution for each value
4. Create histograms showing how scores shift with the nuisance parameter

*Task 2 — Build interpolation functions:*
1. For each bin of the score histogram, fit how the bin content changes as a function of the nuisance parameter
2. Return a function that can predict the histogram for any TES/JES value
3. These fit functions are stored in `saved_info` and used by Group 4 in the likelihood fit

**Why this matters:** In real ATLAS analyses, the tau energy scale is known to 0.1% and the jet energy scale to ~1%. When we shift these calibrations, events move between bins in our analysis, changing the measured $\mu$. The systematic uncertainty quantifies "how much would $\mu$ change if our calibrations are wrong?"

---

## How the Pipeline Works

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  datasets   │───▶│  ingestion   │───▶│  model.fit()     │
│  (2M events)│    │  (orchestrates)│   │  trains classifier│
└─────────────┘    └──────────────┘    └──────────┬───────┘
                                                  │
                    ┌─────────────────────────────┘
                    ▼
          ┌────────────────────┐
          │ model.predict()    │
          │ runs on N sets ×   │
          │ M pseudo-experiments│
          └────────┬───────────┘
                   ▼
          ┌────────────────────┐    ┌──────────────┐
          │ statistical_analysis│◀───│  systematic_ │
          │ compute μ, Δμ      │    │  analysis    │
          └────────┬───────────┘    └──────────────┘
                   ▼
          ┌────────────────────┐
          │  score.py          │
          │  RMSE, MAE,        │
          │  Quantiles Score   │
          └────────────────────┘
```

### Pseudo-experiments

The evaluation doesn't test on one dataset — it creates **N sets × M pseudo-experiments**:
- Each **set** has a different true $\mu$ value (random in [0.1, 3])
- Each **pseudo-experiment** randomizes:
  - Poisson fluctuations on event counts
  - Systematic nuisance parameters (drawn from their priors, if enabled)
  - Applied detector response shifts (4-momentum transformations)

This tests whether your $\mu$ estimate is accurate AND whether your uncertainty (p16 to p84 interval) is well-calibrated — it should cover the true $\mu$ about 68% of the time.

---

## Git Workflow — Collaborating as 25 People

With 25 people (5 groups of ~5) working on the same codebase, a disciplined git workflow is essential to avoid chaos. This project uses **feature branches + pull requests**, with a **Git Guru** who reviews and merges everything.

### The Golden Rules

1. **Never push directly to `main`.** All changes go through a pull request.
2. **One branch per feature/fix.** Keep branches small and focused — a few files, one logical change.
3. **Pull before you push.** Always `git pull` the latest `main` before opening a PR.
4. **Don't commit data files.** The 2M-event Parquet files are handled by `HiggsML.datasets` — never commit them.

### Roles

| Role | Who | Responsibilities |
|---|---|---|
| **Git Guru** | 1 designated person | Reviews all PRs, enforces code quality, resolves merge conflicts, is the only person who merges to `main` |
| **Group Lead** | 1 per group (5 total) | Reviews group members' PRs first, ensures the group's script works end-to-end before passing to the Git Guru |
| **Contributor** | Everyone else (~19 people) | Works on their assigned task in a feature branch, opens a PR, responds to review comments |

### Branch Strategy

```
main ────────────────────────────────────────────────────── (protected, merge-only)
  │
  ├── group1/xgboost-hyperparameters ───► PR ──► merge
  ├── group1/fix-sample-weights        ───► PR ──► merge
  ├── group2/neural-architecture       ───► PR ──► merge
  ├── group2/add-batch-normalization   ───► PR ──► merge
  ├── group3/feature-correlations      ───► PR ──► merge
  ├── group3/systematics-dependence    ───► PR ──► merge
  ├── group4/profile-likelihood        ───► PR ──► merge
  ├── group5/tes-fitter                ───► PR ──► merge
  └── ...
```

Each group works on its own script, so merge conflicts between groups are rare. Within a group, coordinate who works on which function to avoid editing the same lines simultaneously.

### Essential Git Commands (Cheat Sheet)

#### One-time setup

```bash
# Clone the repository
git clone https://github.com/blackSwanCS/Higgs_collaborations.git
cd Higgs_collaborations

# Tell git who you are
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

#### Daily workflow — the 6 commands you'll use 90% of the time

```bash
# 1. Get the latest main
git checkout main
git pull

# 2. Create a branch for your task (name it: group<N>/<what-you-are-doing>)
git checkout -b group1/fix-sample-weights

# 3. ... do your work (edit code, test it) ...

# 4. Stage and commit your changes
git add sample_code_submission/boosted_decision_tree.py
git commit -m "Fix XGBoost sample_weight parameter name"

# 5. Push your branch to GitHub
git push -u origin group1/fix-sample-weights

# 6. Open a Pull Request on GitHub (browser), then notify the Git Guru
```

#### Keeping your branch up to date

```bash
# If main has moved forward while you were working:
git checkout main
git pull                          # Get latest main
git checkout group1/fix-sample-weights
git merge main                    # Bring main's changes into your branch
# Fix any conflicts, then:
git add .
git commit -m "Merge latest main"
git push
```

#### Handling merge conflicts

```bash
# When git merge main shows CONFLICT:
# 1. Open the file(s) — look for <<<<<<<, =======, >>>>>>>
# 2. Decide which version to keep (or combine both), remove the markers
# 3. Tell git the conflict is resolved:
git add conflicted_file.py
git commit -m "Resolve merge conflict"
```

#### Undoing mistakes

```bash
# "I staged the wrong file"
git reset HEAD file.py

# "I want to throw away my uncommitted changes to one file"
git checkout -- file.py

# "My last commit message is wrong" (haven't pushed yet)
git commit --amend -m "Correct message"

# "I committed to main by accident" (haven't pushed yet)
git checkout -b group1/my-fix    # Create a branch from the mistake
git checkout main
git reset --hard origin/main     # Reset main to match remote
```

### Pull Request Checklist

Before opening a PR, verify:

- [ ] Your branch is up to date with `main` (`git merge main` first)
- [ ] Your code runs without errors (`python -m HiggsML.run_competition --submission sample_code_submission --model-type BDT --num-of-sets 1 --num-pseudo-experiments 10`)
- [ ] You only changed files in your group's script (don't touch other groups' files or `model.py`)
- [ ] No data files, `.pyc` files, or notebook checkpoints are committed
- [ ] Your commit messages describe **what** and **why**, not just "update" or "fix"

### The Git Guru's Job

The Git Guru is the single point of merge to `main`. Their responsibilities:

1. **Review every PR** — check for correctness, style, and that only the intended files are changed
2. **Run the full pipeline** on the merged code before accepting
3. **Resolve inter-group conflicts** if two groups accidentally touch the same code
4. **Protect `main`** — enable branch protection rules on GitHub (require PR reviews, require status checks)
5. **Tag releases** (`v1.0`, `v1.1`, ...) so everyone can roll back if something breaks

The Git Guru is **not** responsible for making everyone's code work — each group owns the quality of their script. The Guru's role is integration and gatekeeping.

### GitHub Branch Protection (Setup for Git Guru)

On the GitHub repository → Settings → Branches → Add rule:

| Setting | Value |
|---|---|
| Branch name pattern | `main` |
| Require a pull request before merging | ✅ |
| Require approvals | 1 |
| Dismiss stale pull request approvals | ✅ |
| Require status checks to pass | ✅ |

This prevents anyone (even the Guru) from accidentally pushing directly to `main`.

---

## Scoring Metrics

The competition evaluates three aspects of your analysis:

| Metric | What it measures | Good value |
|---|---|---|
| **RMSE** | Accuracy of $\hat{\mu}$ + calibration of $\Delta\hat{\mu}$ | As low as possible |
| **MAE** | Same, but in absolute (L1) sense | As low as possible |
| **Quantiles Score** | Interval tightness × coverage quality | As high as possible |

The Quantiles Score rewards intervals that are **narrow** (precise) while maintaining **68% coverage** (well-calibrated). An interval that is too narrow misses the truth; an interval that is too wide is imprecise. Both are penalized.

---

## Files Reference

### You modify (5 groups):
| File | Group | Lines | Key functions |
|---|---|---|---|
| `boosted_decision_tree.py` | 1 | 27 | `__init__`, `fit`, `predict` |
| `neural_network.py` | 2 | 39 | `__init__`, `fit`, `predict` |
| `feature_analysis.py` | 3 | 11 | `feature_correlations`, `systematics_dependence`, `minimal_dependent_features` |
| `statistical_analysis.py` | 4 | 87 | `compute_mu`, `calculate_saved_info` |
| `systematic_analysis.py` | 5 | 65 | `tes_fitter`, `jes_fitter` |

### You don't modify:
| File | Role |
|---|---|
| `model.py` | Orchestrator — composes all group scripts together |
| `utils.py` | Visualization utilities (histograms, ROC curves) |
| `sample_model.py` | Minimal working example for reference |

### Infrastructure (black_swan_pkg):
| Module | Role |
|---|---|
| `HiggsML.datasets` | Download and load Parquet data |
| `HiggsML.ingestion` | Competition pipeline (init → fit → predict → score) |
| `HiggsML.systematics` | Physics systematics engine (4-vectors, detector response) |
| `HiggsML.derived_quantities` | Compute DER features from primary quantities |
| `HiggsML.score` | Scoring metrics and HTML report generation |
| `HiggsML.visualization` | Data visualization tools |

---

## Authors

- **Ragansu Chakkappai** — Université Paris-Saclay
- **David Rousseau** — IJCLab, CNRS/IN2P3
- **Victor Estrade** — CentraleSupélec
- **Ihsan Ullah** — Université Paris-Saclay
