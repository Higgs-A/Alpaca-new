import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import copy

# Importations de vos fichiers locaux
from model import Model
from HiggsML.systematics import systematics
from systematic_analysis import tes_fitter, jes_fitter 

# ==============================================================================
# ZONE 1.B : ÉTUDE DE L'INCERTITUDE DE NORMALISATION DU BRUIT (BKG_SCALE)
# ==============================================================================

def visualiser_impact_bkg(model, training_dict):
    """
    Étude spécifique pour la systématique de normalisation du background.
    On fait varier le bkg_scale (généralement entre 0.9 et 1.1).
    """
    # On étudie une variation de +/- 10% sur la quantité de bruit
    bkg_values = np.linspace(0.9, 1.1, 7)
    num_bins = 6
    
    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(231) # Dist Signal (ne doit pas bouger)
    ax2 = plt.subplot(232) # Dist Bruit (doit monter/descendre)
    ax3 = plt.subplot(233) # ROC (doit varier car le rapport S/B change)
    ax4 = plt.subplot(234) # Delta Signal vs Bkg_Scale (doit être plat à zéro)
    ax5 = plt.subplot(235) # Delta Bruit vs Bkg_Scale (doit être une droite)

    colors_bkg = plt.cm.coolwarm(np.linspace(0, 1, len(bkg_values)))
    colors_bins = plt.cm.viridis(np.linspace(0, 1, num_bins))

    # --- RÉFÉRENCE NOMINALE (bkg_scale = 1.0) ---
    # On injecte bkg_scale=1.0 dans la fonction systematics
    set_nom = systematics(training_dict, bkg_scale=1.0) 
    scores_nom = np.array(model.predict(set_nom["data"])).ravel()
    labels_nom = np.array(set_nom["labels"]).ravel()
    weights_nom = np.array(set_nom["weights"]).ravel()

    is_sig_nom, is_bkg_nom = (labels_nom == 1.0), (labels_nom == 0.0)
    bins_fixes = np.linspace(np.min(scores_nom), np.max(scores_nom), num_bins + 1)
    
    nom_s_counts, _ = np.histogram(scores_nom[is_sig_nom], bins=bins_fixes, weights=weights_nom[is_sig_nom])
    nom_b_counts, _ = np.histogram(scores_nom[is_bkg_nom], bins=bins_fixes, weights=weights_nom[is_bkg_nom])

    hist_diff_s, hist_diff_b = [], []

    print("Analyse de l'impact de la normalisation du bruit (Bkg Scale)...")
    # Utilise .copy() ou recrée le dictionnaire pour protéger les données originales
    set_nom = systematics(copy.deepcopy(training_dict), bkg_scale=1.0)

    for i, val in enumerate(bkg_values):
        # APPLICATION DE LA SYSTÉMATIQUE : On change uniquement bkg_scale
        # Envoie une copie du dictionnaire à la fonction
        set_shift = systematics(copy.deepcopy(training_dict), bkg_scale=val)
        
        scores = np.array(model.predict(set_shift["data"])).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()

        is_sig, is_bkg = (labels == 1.0), (labels == 0.0)
        
        # Comptage
        curr_s_counts, _ = np.histogram(scores[is_sig], bins=bins_fixes, weights=weights[is_sig])
        curr_b_counts, _ = np.histogram(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg])

        hist_diff_s.append(curr_s_counts - nom_s_counts)
        hist_diff_b.append(curr_b_counts - nom_b_counts)

        # Tracé des distributions
        is_nom = np.isclose(val, 1.0)
        lw, alpha = (3, 1.0) if is_nom else (1.5, 0.7)

        ax1.hist(scores[is_sig], bins=bins_fixes, weights=weights[is_sig], histtype='step', 
                 color=colors_bkg[i], linewidth=lw, alpha=alpha, label=f"Scale={val:.2f}")
        ax2.hist(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg], histtype='step', 
                 color=colors_bkg[i], linewidth=lw, alpha=alpha, label=f"Scale={val:.2f}")
        
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        ax3.plot(fpr, tpr, color=colors_bkg[i], lw=lw, alpha=alpha, label=f"AUC={auc(fpr, tpr):.3f}")

    # --- TRACÉ DES DÉRIVES vs BKG_SCALE ---
    hist_diff_s = np.array(hist_diff_s)
    hist_diff_b = np.array(hist_diff_b)

    for b in range(num_bins):
        # Δ Signal vs Bkg_Scale
        ax4.plot(bkg_values, hist_diff_s[:, b], 'o-', color=colors_bins[b], label=f"Bin {b+1}")
        # Δ Bruit vs Bkg_Scale
        ax5.plot(bkg_values, hist_diff_b[:, b], 'o-', color=colors_bins[b], label=f"Bin {b+1}")

    # --- MISE EN FORME ---
    ax1.set_title("Distribution SIGNAL", fontweight='bold'); ax1.legend(fontsize='x-small', ncol=2)
    ax2.set_title("Distribution BRUIT (Normalisation)", fontweight='bold'); ax2.legend(fontsize='x-small', ncol=2)
    ax3.set_title("Impact sur la ROC", fontweight='bold'); ax3.plot([0,1],[0,1],'--',color='grey')
    
    for ax in [ax4, ax5]:
        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.axvline(1.0, color='black', lw=1, linestyle='--')
        ax.set_xlabel("Valeur du BKG_SCALE")
        ax.legend(fontsize='x-small', ncol=2)

    ax4.set_title("Δ Signal vs Bkg Scale (Doit être nul)", color='blue', fontweight='bold')
    ax5.set_title("Δ Bruit vs Bkg Scale (Évolution linéaire)", color='red', fontweight='bold')

    plt.suptitle("Étude de l'Incertitude de Normalisation du Background", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==============================================================================
# ZONE 2 : CHARGEMENT DES DONNÉES 
# ==============================================================================

print("Localisation et chargement des données...")
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
data_path = os.path.join(root_dir, "blackSwan_data", "blackSwan_data.parquet")

# 1. On charge tout
df_complet = pd.read_parquet(data_path)

# 2. On divise le dataset en Train (80%) et Eval (20%)
# L'utilisation d'un random_state garantit que la séparation sera toujours la même
df_train, df_eval = train_test_split(df_complet, test_size=0.20, random_state=42) #on est pas obligés de garder le random state, mais ça peut aider à la reproductibilité et au debug 

# 3. Fonction pour le wrapper (qui ne tire QUE dans df_train)
def get_train_set_custom(selected_indices):
    return df_train.iloc[selected_indices].copy()

# 4. On prépare le dictionnaire d'évaluation que tu vas envoyer à tes fitters
# La fonction systematics() s'attend souvent à un dictionnaire avec "data", "labels", "weights"
eval_dict = {
    "data": df_eval.drop(columns=["labels", "weights", "detailed_labels"], errors='ignore'),
    "labels": df_eval["labels"].values,
    "weights": df_eval["weights"].values
}

# ==============================================================================
# ZONE 3 : INITIALISATION ET EXÉCUTION
# ==============================================================================

print("Initialisation du pipeline...")
mon_wrapper = Model(
    get_train_set=get_train_set_custom,
    systematics=systematics,
    model_type="BDT"
)

print("Entraînement du modèle")
mon_wrapper.fit()

# LANCEMENT DE LA VISUALISATION (SUR LES 20% RESTANTS)
print("Génération des graphiques multi-shifts (sur l'Eval Set)...")
visualiser_impact_bkg(mon_wrapper.model, eval_dict)

# APPEL DES FITTERS (SUR LES 20% RESTANTS)
print("Calcul de la paramétrisation BKG (sur l'Eval Set)...")
transformateur_bkg = bkg_fitter(
    model=mon_wrapper.model,
    train_set=eval_dict  
)
print("Analyse terminée.")

