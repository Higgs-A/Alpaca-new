import pandas as pd
import numpy as np

# Importations de vos fichiers
from model import Model
from HiggsML.systematics import systematics
from systematic_analysis import tes_fitter, jes_fitter # Votre code

# 1. Chargement des données
print("Chargement des données...")
# Remplacez par votre vrai chemin vers le fichier .parquet
df_complet = pd.read_parquet('/home/beta/Documents/school/Higgs/Alpaca-new/blackSwan_data/blackSwan_data.parquet')

# 2. Fonction requise par la classe Model
def get_train_set_custom(selected_indices):
    return df_complet.iloc[selected_indices].copy()

# 3. Initialisation et entraînement du Modèle
print("Initialisation du modèle...")
mon_wrapper = Model(
    get_train_set=get_train_set_custom,
    systematics=systematics,
    model_type="sample_model" # ou "BDT", "NN"
)

print("Entraînement en cours...")
mon_wrapper.fit()

# 4. recherche de tes_fitter
print("Calcul de la paramétrisation TES...")
# La fonction renvoie directement le "transformateur" mathématique
transformateur_tes = tes_fitter(
    model=mon_wrapper.model,
    train_set=mon_wrapper.training_set
)
print(" Modélisation TES terminée !")

# ==========================================
# EXEMPLE POUR L'ÉQUIPE STATISTIQUES :
# ==========================================
print("\n--- Test d'utilisation par l'équipe STAT ---")

# A. Ils doivent d'abord générer l'histogramme nominal (TES = 1.0)
scores_nominaux = mon_wrapper.model.predict(mon_wrapper.training_set["data"])
histogramme_nominal, _ = np.histogram(scores_nominaux, bins=100, range=(0, 1))

# B. Ils demandent un nouvel histogramme pour TES = 1.04 (variation de 4%)
valeur_tes_testee = 1.04
nouvel_histogramme = transformateur_tes(array=histogramme_nominal, tes=valeur_tes_testee)

print(f"Somme des événements nominaux : {np.sum(histogramme_nominal):.2f}")
print(f"Somme des événements après altération TES (+4%) : {np.sum(nouvel_histogramme):.2f}")