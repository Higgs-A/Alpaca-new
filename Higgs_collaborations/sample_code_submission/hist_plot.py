import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Importations de vos fichiers locaux
from model import Model
from HiggsML.systematics import systematics
from systematic_analysis import tes_fitter, jes_fitter 

# ==============================================================================
# ZONE 1 : DÉFINITION DE LA FONCTION DE VISUALISATION (VOTRE DEMANDE)
# ==============================================================================

def visualiser_impact_tes(model, training_dict):
    """
    Affiche 3 subplots sur une seule fenêtre :
    1. Histogramme Signal pour les 7 valeurs de TES
    2. Histogramme Bruit pour les 7 valeurs de TES
    3. Courbes ROC pour les 7 valeurs de TES
    """
    # On définit les 7 valeurs (Nominal 1.0 + 6 autres entre 0.97 et 1.03)
    tes_values = np.linspace(0.97, 1.03, 7)
    
    # Création de la figure avec 3 colonnes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Palette de couleurs "Plasma" pour bien distinguer les shifts
    colors = plt.cm.plasma(np.linspace(0, 1, len(tes_values)))

    print("Calcul des shifts et génération des graphiques...")

    for i, val in enumerate(tes_values):
        # 1. Appliquer le shift physique
        set_shift = systematics(training_dict, tes=val)
        
        # 2. Prédire les scores
        scores_bruts = model.predict(set_shift["data"])
        
        # 3. Conversion en pur Numpy
        scores = np.array(scores_bruts).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()

        # 4. Le masque basé sur vos VRAIES données (0. et 1.)
        is_signal = (labels == 1.0)
        is_bkg = (labels == 0.0)

        # 5. Filtrage
        s_scores = scores[is_signal]
        b_scores = scores[is_bkg]
        s_w = weights[is_signal]
        b_w = weights[is_bkg]
        
        # Style visuel (NOMINAL)
        is_nominal = np.isclose(val, 1.0)
        lw = 3 if is_nominal else 1.2
        alpha = 1.0 if is_nominal else 0.7
        label_tag = " (NOMINAL)" if is_nominal else ""

        # --- SUBPLOT 1 : SIGNAL ---
        ax1.hist(s_scores, bins=40, weights=s_w, histtype='step', 
                 color=colors[i], linewidth=lw, alpha=alpha, label=f"TES={val:.2f}{label_tag}")
        
        # --- SUBPLOT 2 : BRUIT ---
        ax2.hist(b_scores, bins=40, weights=b_w, histtype='step', 
                 color=colors[i], linewidth=lw, alpha=alpha, label=f"TES={val:.2f}")

        # --- SUBPLOT 3 : COURBE ROC ---
        # On utilise les vrais labels (0. et 1.)
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color=colors[i], lw=lw, alpha=alpha, 
                 label=f"TES={val:.2f} (AUC={roc_auc:.3f})")
    
    # Mise en forme du Plot Signal
    ax1.set_title("Distribution du SIGNAL (Higgs)", fontweight='bold')
    ax1.set_xlabel("Score du modèle")
    ax1.set_ylabel("Somme des poids")
    ax1.legend(fontsize='x-small', ncol=2)

    # Mise en forme du Plot Bruit
    ax2.set_title("Distribution du BRUIT (Background)", fontweight='bold')
    ax2.set_xlabel("Score du modèle")
    ax2.legend(fontsize='x-small', ncol=2)

    # Mise en forme de la Courbe ROC
    ax3.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1)
    ax3.set_title("Courbes ROC (Performance)", fontweight='bold')
    ax3.set_xlabel("Taux de Faux Positifs")
    ax3.set_ylabel("Taux de Vrais Positifs")
    ax3.legend(fontsize='x-small')

    plt.suptitle("Étude de l'Incertitude Systématique : Échelle d'énergie du Tau (TES)", fontsize=16)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# ZONE 2 : CHARGEMENT DES DONNÉES (VOTRE STRUCTURE DE DOSSIERS)
# ==============================================================================

print("Localisation et chargement des données...")
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
data_path = os.path.join(root_dir, "blackSwan_data", "blackSwan_data.parquet")

df_complet = pd.read_parquet(data_path)

def get_train_set_custom(selected_indices):
    return df_complet.iloc[selected_indices].copy()

# ==============================================================================
# ZONE 3 : INITIALISATION ET EXÉCUTION
# ==============================================================================

print("Initialisation du pipeline...")
mon_wrapper = Model(
    get_train_set=get_train_set_custom,
    systematics=systematics,
    model_type="sample_model"
)

print("Entraînement du modèle...")
mon_wrapper.fit()

# LANCEMENT DE VOTRE VISUALISATION
print("Génération des graphiques multi-shifts...")
visualiser_impact_tes(mon_wrapper.model, mon_wrapper.training_set)

# APPEL DE VOS FITTERS (POUR L'ÉQUIPE STAT)
print("Calcul de la paramétrisation TES...")
transformateur_tes = tes_fitter(
    model=mon_wrapper.model,
    train_set=mon_wrapper.training_set
)
print("Analyse terminée.")