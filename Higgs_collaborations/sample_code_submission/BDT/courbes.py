import os
import time
from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from boosted_decision_tree import BoostedDecisionTree

np.random.seed(31415)
datapath = ""
filename = os.path.join(datapath, "dataWW_d1_600k.csv.gz")

print("Lecture du fichier :", filename)
dfall = pd.read_csv(filename)

# Mélange de sécurité et hack du mcWeight (notations originales)
dfall = dfall.sample(frac=1).reset_index(drop=True)
dfall.mcWeight *= 4

# --- SELECTION DES ÉVÉNEMENTS ---
# Uniquement les événements avec exactement deux leptons et un poids positif
fulldata = dfall[(dfall.lep_n == 2) & (dfall.mcWeight > 0)]
print("Shape du dataset après sélection primaire :", fulldata.shape)

# Séparation des features, des targets et des poids physiques
target = fulldata["label"]
weights = fulldata["mcWeight"]

# Choix des features d'entraînement
data = pd.DataFrame(
    fulldata,
    columns=[
        "met_et",
        "met_phi",
        "lep_pt_0",
        "lep_pt_1",
        "lep_phi_0",
        "lep_phi_1",
    ],
)

#Calcul de la signification maximale

def amsasimov(s_in, b_in):
    """Calcule la signification d'Asimov (arXiv:1007.1727 eq. 97)"""
    s = np.copy(s_in)
    b = np.copy(b_in)
    s = np.where((b_in == 0), 0.0, s_in)
    b = np.where((b_in == 0), 1.0, b)

    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    ams = np.where((s < 0) | (b < 0), np.nan, ams)
    if np.isscalar(s_in):
        return float(ams)
    else:
        return ams


def significance_vscore(y_true, y_score, sample_weight=None):
    """Calcule le vecteur de signification Z selon le seuil de coupure"""
    if sample_weight is None:
        sample_weight = np.full(len(y_true), 1.0)

    bins = np.linspace(0, 1.0, 101)
    s_hist, _ = np.histogram(
        y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
    )
    b_hist, _ = np.histogram(
        y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
    )

    # Sommes cumulées inversées (du plus haut score au plus bas)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    return amsasimov(s_cumul, b_cumul)


def significance_score(y_true, y_score, sample_weight=None):
    """Retourne la signification maximale trouvée (Z max)"""
    vsig = significance_vscore(y_true, y_score, sample_weight)
    return np.max(vsig)


train_size = 0.75
X_train, X_test, y_train, y_test, weights_train, weights_test = (
    train_test_split(data, target, weights, train_size=train_size)
)

# Conversion immédiate des poids de test en array numpy pour éviter les indexations désalignées
weights_test_arr = np.array(weights_test)

# Correction du poids du lot de test pour compenser l'échantillonnage global (25% restants)
for i in [0, 1]:
    weights_test_arr[y_test == i] *= 1 / (1 - train_size)


# 
print("\nDébut de l'entraînement du modèle autonome...")
bdt_model = BoostedDecisionTree()

# On passe le lot d'entraînement brut (la classe gère sa validation interne et son scaling)
bdt_model.fit(X_train, y_train, weights=weights_train)
print("Modèle entraîné avec succès (Early Stopping appliqué en interne).")

# Évaluation des scores de probabilité
y_pred_test = bdt_model.predict(X_test)
y_pred_train = bdt_model.predict(X_train)

