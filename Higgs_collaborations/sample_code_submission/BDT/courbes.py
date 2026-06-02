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