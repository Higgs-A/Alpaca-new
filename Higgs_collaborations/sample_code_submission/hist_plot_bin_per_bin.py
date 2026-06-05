import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Importations de vos fichiers locaux
from model import Model
from HiggsML.systematics import systematics
from systematic_analysis import tes_fitter, jes_fitter

# Configuration des valeurs nominales par défaut pour chaque systématique
NOMINALS = {
    "tes": 1.0,
    "jes": 1.0,
    "bkg_scale": 1.0,
    "soft_met": 0.0  
}

# ==============================================================================
# ZONE 1 : DASHBOARD SYSTÉMATIQUE GÉNÉRALISÉ (AVEC SÉCURITÉ BKG_SCALE + MET 0)
# ==============================================================================

def visualiser_impact_systematique(model, training_dict, syst_name="tes", param_min=0.97, param_max=1.03, step=0.01):
    """
    Visualise l'impact d'une incertitude systématique ('tes', 'jes', 'bkg_scale' ou 'soft_met')
    sur les distributions, la courbe ROC et les dérives par bin.
    """
    syst_name_upper = syst_name.upper()
    val_nominale = NOMINALS.get(syst_name, 1.0)
   
    # Calcul dynamique du nombre de points pour respecter scrupuleusement le pas indiqué
    num_points = int(round((param_max - param_min) / step)) + 1
    param_values = np.linspace(param_min, param_max, num_points)
   
    print(f"\n--- Scan {syst_name_upper} configuré : de {param_min} à {param_max} avec un pas de {step} ({num_points} points) ---")
   
    num_bins = 15
   
    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(231) # Dist Signal
    ax2 = plt.subplot(232) # Dist Bruit
    ax3 = plt.subplot(233) # ROC
    ax4 = plt.subplot(234) # Delta Signal vs Systématique (par bin)
    ax5 = plt.subplot(235) # Delta Bruit vs Systématique (par bin)

    # Couleurs : Plasma pour les variations (en haut), Viridis pour les Bins (en bas)
    colors_syst = plt.cm.plasma(np.linspace(0, 1, len(param_values)))
    colors_bins = plt.cm.viridis(np.linspace(0, 1, num_bins))

    print(f"Calcul de la référence Nominale ({syst_name_upper} = {val_nominale})...")
   
    # Sécurité : Copie locale pour éviter les effets de bord "in-place"
    dict_nom_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
    set_nom = systematics(dict_nom_input, **NOMINALS)
   
    scores_nom = np.array(model.predict(set_nom["data"])).ravel()
    labels_nom = np.array(set_nom["labels"]).ravel()
    weights_nom = np.array(set_nom["weights"]).ravel()

    is_sig_nom, is_bkg_nom = (labels_nom == 1.0), (labels_nom == 0.0)
   
    # Bins fixes basés sur les scores nominaux
    bins_fixes = np.linspace(np.min(scores_nom), np.max(scores_nom), num_bins + 1)
   
    # Comptage Nominal
    nom_s_counts, _ = np.histogram(scores_nom[is_sig_nom], bins=bins_fixes, weights=weights_nom[is_sig_nom])
    nom_b_counts, _ = np.histogram(scores_nom[is_bkg_nom], bins=bins_fixes, weights=weights_nom[is_bkg_nom])

    # Tableaux pour stocker l'historique des différences
    historique_diff_s = []
    historique_diff_b = []

    print(f"Boucle sur les valeurs de {syst_name_upper}...")
    for i, val in enumerate(param_values):
       
        # Sécurité : On recrée un dictionnaire propre pour chaque itération
        dict_loop_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
       
        # Configuration des arguments
        syst_kwargs = NOMINALS.copy()
        syst_kwargs[syst_name] = val
       
        set_shift = systematics(dict_loop_input, **syst_kwargs)
       
        scores = np.array(model.predict(set_shift["data"])).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()

        is_sig, is_bkg = (labels == 1.0), (labels == 0.0)
       
        # 🛡️ SÉCURITÉ MANUELLE (FALLBACK) : Si bkg_scale est ignoré par le package, on force l'application ici
        if syst_name == "bkg_scale":
            weights[is_bkg] = weights[is_bkg] * val
       
        # Comptage actuel
        curr_s_counts, _ = np.histogram(scores[is_sig], bins=bins_fixes, weights=weights[is_sig])
        curr_b_counts, _ = np.histogram(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg])

        # Différence stockée
        historique_diff_s.append(curr_s_counts - nom_s_counts)
        historique_diff_b.append(curr_b_counts - nom_b_counts)

        # Style des plots
        is_nom = np.isclose(val, val_nominale, atol=step/2)
        lw, alpha, tag = (3, 1.0, " (NOMINAL)") if is_nom else (1.5, 0.8, "")

        # --- PLOTS LIGNE 1 (Distributions) ---
        ax1.hist(scores[is_sig], bins=bins_fixes, weights=weights[is_sig], histtype='step',
                 color=colors_syst[i], linewidth=lw, alpha=alpha, label=f"{syst_name_upper}={val:.2f}{tag}")
        ax2.hist(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg], histtype='step',
                 color=colors_syst[i], linewidth=lw, alpha=alpha, label=f"{syst_name_upper}={val:.2f}")
       
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        ax3.plot(fpr, tpr, color=colors_syst[i], lw=lw, alpha=alpha, label=f"{syst_name_upper}={val:.2f} (AUC={auc(fpr, tpr):.3f})")

    # --- ÉTAPE 3 : TRACÉ DES DÉRIVES (LIGNE 2) ---
    historique_diff_s = np.array(historique_diff_s)
    historique_diff_b = np.array(historique_diff_b)

    for b in range(int(num_bins/2), num_bins):  
        evolution_bin_s = historique_diff_s[:, b]
        evolution_bin_b = historique_diff_b[:, b]

        ax4.plot(param_values, evolution_bin_s, 'o-', color=colors_bins[b], lw=2, label=f"Bin {b+1}")
        ax5.plot(param_values, evolution_bin_b, 'o-', color=colors_bins[b], lw=2, label=f"Bin {b+1}")

    # --- MISE EN FORME ---
    ax1.set_title("Distribution SIGNAL", fontweight='bold'); ax1.legend(fontsize='x-small', ncol=2)
    ax2.set_title("Distribution BRUIT", fontweight='bold'); ax2.legend(fontsize='x-small', ncol=2)
    ax3.set_title("Performance ROC", fontweight='bold'); ax3.plot([0,1],[0,1],'--',color='grey'); ax3.legend(fontsize='x-small')
   
    for ax in [ax4, ax5]:
        ax.axhline(0, color='black', lw=1, linestyle='--')  
        ax.axvline(val_nominale, color='black', lw=1, linestyle='--')
        ax.set_xlabel(f"Valeur de {syst_name_upper}")
        ax.legend(fontsize='x-small', ncol=2)

    ax4.set_title(f"Évolution du Signal par Bin vs {syst_name_upper}", color='red', fontweight='bold')
    ax4.set_ylabel("Différence d'événements (Δ)")
    ax5.set_title(f"Évolution du Bruit par Bin vs {syst_name_upper}", color='red', fontweight='bold')

    plt.suptitle(f"Analyse Systématique : Impact du {syst_name_upper} (Template Morphing)", fontsize=18)
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
df_train, df_eval = train_test_split(df_complet, test_size=0.20, random_state=42)

def get_train_set_custom(selected_indices):
    return df_train.iloc[selected_indices].copy()

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

print("Génération des graphiques multi-shifts (sur l'Eval Set)...")

# 1. TES (Scan de 0.7 à 1.3)
visualiser_impact_systematique(mon_wrapper.model, eval_dict, syst_name="tes", param_min=0.97, param_max=1.03, step=0.01)

# 2. JES (Scan de 0.7 à 1.3)
visualiser_impact_systematique(mon_wrapper.model, eval_dict, syst_name="jes", param_min=0.97, param_max=1.03, step=0.01)

# 3. bkg_scale (Scan de 0.95 à 1.05 avec sécurité manuelle active)
visualiser_impact_systematique(mon_wrapper.model, eval_dict, syst_name="bkg_scale", param_min=0.95, param_max=1.05, step=0.01)

# 4. Soft MET (Scan de -3 à +3 GeV autour de 0)
visualiser_impact_systematique(mon_wrapper.model, eval_dict, syst_name="soft_met", param_min=-3.0, param_max=3.0, step=0.5)


# --- APPEL DES FITTERS ---
print("\nCalcul de la paramétrisation TES (sur l'Eval Set)...")
transformateur_tes = tes_fitter(model=mon_wrapper.model, train_set=eval_dict)

print("Calcul de la paramétrisation JES (sur l'Eval Set)...")
transformateur_jes = jes_fitter(model=mon_wrapper.model, train_set=eval_dict)

print("Analyse complète terminée.")