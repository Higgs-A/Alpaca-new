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
# ZONE 1 : DASHBOARD SYSTÉMATIQUE (DISTRIBUTIONS + DIFFÉRENCES)
# ==============================================================================

def visualiser_impact_tes(model, training_dict):
    tes_values = np.linspace(0.97, 1.03, 7)
    
    # Grille 2x3 : 3 plots en haut, 2 en bas pour les différences
    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(231) # Dist Signal
    ax2 = plt.subplot(232) # Dist Bruit
    ax3 = plt.subplot(233) # ROC
    ax4 = plt.subplot(234) # Delta Signal
    ax5 = plt.subplot(235) # Delta Bruit

    colors = plt.cm.plasma(np.linspace(0, 1, len(tes_values)))

    # --- ÉTAPE 1 : Référence Nominale ---
    set_nom = systematics(training_dict, tes=1.0)
    scores_nom = np.array(model.predict(set_nom["data"])).ravel()
    labels_nom = np.array(set_nom["labels"]).ravel()
    weights_nom = np.array(set_nom["weights"]).ravel()

    is_sig_nom, is_bkg_nom = (labels_nom == 1.0), (labels_nom == 0.0)
    
    # On définit 10 bins fixes et on calcule le centre des bins pour le plot
    bins_fixes = np.linspace(np.min(scores_nom), np.max(scores_nom), 11)
    bin_centers = (bins_fixes[:-1] + bins_fixes[1:]) / 2

    # Comptage Nominal
    nom_s_counts, _ = np.histogram(scores_nom[is_sig_nom], bins=bins_fixes, weights=weights_nom[is_sig_nom])
    nom_b_counts, _ = np.histogram(scores_nom[is_bkg_nom], bins=bins_fixes, weights=weights_nom[is_bkg_nom])

    # --- ÉTAPE 2 : Boucle de calcul des Shifts ---
    for i, val in enumerate(tes_values):
        set_shift = systematics(training_dict, tes=val)
        scores = np.array(model.predict(set_shift["data"])).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()

        is_sig, is_bkg = (labels == 1.0), (labels == 0.0)
        
        # Comptage actuel
        curr_s_counts, _ = np.histogram(scores[is_sig], bins=bins_fixes, weights=weights[is_sig])
        curr_b_counts, _ = np.histogram(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg])

        # CALCUL DES DIFFÉRENCES
        diff_s = curr_s_counts - nom_s_counts
        diff_b = curr_b_counts - nom_b_counts

        # Style
        is_nom = np.isclose(val, 1.0)
        lw, alpha, tag = (3, 1.0, " (NOMINAL)") if is_nom else (1.5, 0.8, "")

        # --- PLOTS LIGNE 1 (Distributions) ---
        ax1.hist(scores[is_sig], bins=bins_fixes, weights=weights[is_sig], histtype='step', 
                 color=colors[i], linewidth=lw, alpha=alpha, label=f"TES={val:.2f}{tag}")
        ax2.hist(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg], histtype='step', 
                 color=colors[i], linewidth=lw, alpha=alpha, label=f"TES={val:.2f}")
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        ax3.plot(fpr, tpr, color=colors[i], lw=lw, alpha=alpha, label=f"TES={val:.2f} (AUC={auc(fpr, tpr):.3f})")

        # --- PLOTS LIGNE 2 (Différences) ---
        # On utilise une ligne avec des points pour bien voir la dérive par bin
        ax4.plot(bin_centers, diff_s, 'o-', color=colors[i], lw=lw, alpha=alpha, label=f"TES={val:.2f}")
        ax5.plot(bin_centers, diff_b, 'o-', color=colors[i], lw=lw, alpha=alpha, label=f"TES={val:.2f}")

    # --- MISE EN FORME ---
    ax1.set_title("Distribution SIGNAL", fontweight='bold'); ax1.legend(fontsize='x-small', ncol=2)
    ax2.set_title("Distribution BRUIT", fontweight='bold'); ax2.legend(fontsize='x-small', ncol=2)
    ax3.set_title("Performance ROC", fontweight='bold'); ax3.plot([0,1],[0,1],'--',color='grey'); ax3.legend(fontsize='x-small')
    
    # Axes des différences
    ax4.axhline(0, color='black', lw=1, linestyle='--') # Ligne 0 de référence
    ax4.set_title("Δ Signal (Valeur - Nominal)", color='red', fontweight='bold')
    ax4.set_xlabel("Score du BDT")
    ax4.set_ylabel("Différence de poids")

    ax5.axhline(0, color='black', lw=1, linestyle='--') # Ligne 0 de référence
    ax5.set_title("Δ Bruit (Valeur - Nominal)", color='red', fontweight='bold')
    ax5.set_xlabel("Score du BDT")

    plt.suptitle("Analyse Systématique TES : Dérive des comptages par bin", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    # model_type="sample_model"
    model_type = "BDT"
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

