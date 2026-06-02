# 🔍 Diagnostic : Pourquoi les scores sont autour de 100 au lieu de (0, 1)

## Le coupable

Le problème vient du fichier **`sample_model.py`**, ligne 20 :

```python
def predict(self, test_data):
    return np.array(test_data["DER_mass_vis"])
```

`DER_mass_vis` est une variable physique — **la masse invariante du système di-tau** — qui s'exprime en **GeV**. Ses valeurs typiques sont entre **~50 et ~150 GeV**. Le modèle se contente de renvoyer cette colonne brute, sans aucune transformation.

## Le chemin du bug

```
hist_plot.py
  → Model(model_type="sample_model")     # utilise SampleModel
    → SampleModel.predict()              # renvoie DER_mass_vis (~50-150 GeV)
      → scores bruts = ~100              # AUCUNE normalisation n'est appliquée
        → plt.hist(scores)               # affiche des valeurs autour de 100
        → roc_curve(labels, scores)      # les courbes ROC sont faussées
```

À aucun moment une transformation (sigmoïde, calibration, division) n'est appliquée pour ramener ces valeurs dans l'intervalle **[0, 1]**.

## Pourquoi les deux autres modèles fonctionnent correctement

| Modèle | Ce que `predict()` renvoie | Intervalle |
|---|---|---|
| **`BoostedDecisionTree`** (`BDT`) | `XGBClassifier.predict_proba(...)[:, 1]` | [0, 1] ✅ |
| **`NeuralNetwork`** (`NN`) | Couche de sortie `sigmoid` → probabilité | [0, 1] ✅ |
| **`SampleModel`** (utilisé ici) | Colonne brute `DER_mass_vis` | ~50–150 ❌ |

Le **BDT** utilise `predict_proba()` qui renvoie la probabilité d'appartenir à la classe signal — une valeur entre 0 et 1.
Le **NN** a une couche de sortie avec activation `sigmoid`, qui contraint la sortie entre 0 et 1.
Le **SampleModel** n'a ni l'un ni l'autre — il renvoie une colonne de données brutes.

## Conséquences en cascade

Le bug ne se limite pas à l'affichage des histogrammes :

1. **`systematic_analysis.py:25`** — `tes_fitter` crée un histogramme avec `range=(0, 1)`. Toutes les valeurs (~100) tombent **hors de l'intervalle** → histogramme vide.

2. **`statistical_analysis.py:35`** — `compute_mu` applique un seuil `score > 0.5`. Comme toutes les masses sont >> 0.5, **100% des événements sont classés comme signal** → estimation de 𝜇 complètement fausse.

## La correction

Dans **`hist_plot.py`**, ligne 127, remplacer :

```python
model_type="sample_model"
```

par :

```python
model_type="BDT"
```

Cela utilisera XGBoost avec `predict_proba()`, qui renvoie de vraies probabilités entre 0 et 1.

---

> **Note :** `SampleModel` est un **placeholder** intentionnel. Le commentaire dans son code le dit explicitement :
> *"This Dummy class implements a decision tree classifier — change the code in the fit method to implement a decision tree classifier."*
> Il est fait pour être remplacé par un vrai modèle, pas pour être utilisé tel quel.
