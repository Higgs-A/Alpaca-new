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

# Configuration des valeurs nominales par défaut
NOMINALS = {
    "tes": 1.0,
    "jes": 1.0,
    "bkg_scale": 1.0,
    "soft_met": 0.0  
}

# Définition des valeurs à +1σ et -1σ pour chaque systématique (selon les énoncés)
# Les clés correspondent EXACTEMENT aux noms demandés par votre professeur
SIGMA_SHIFTS = {
    "tes": {"plus": 1.03, "minus": 0.97},
    "jes": {"plus": 1.03, "minus": 0.97},
    "bnorm": {"plus": 1.05, "minus": 0.95}, # Correspond à bkg_scale (5%)
    "smet": {"plus": 3.0, "minus": -3.0}     # Correspond à soft_met (3 GeV)
}

# Correspondance entre le nom du prof (clé de sortie) et le nom de l'argument du code
PARAM_MAPPING = {
    "tes": "tes",
    "jes": "jes",
    "bnorm": "bkg_scale",
    "smet": "soft_met"
}

# ==============================================================================
# FONCTION DEMANDEE PAR LE PROFESSEUR : GENERATION DE SAVED_INFO
# ==============================================================================

def generer_saved_info(model, training_dict, num_bins=15):
    """
    Calcule les deltas d'événements par bin à +1σ et -1σ pour chaque systématique
    et retourne le dictionnaire structuré pour le groupe STAT.
    """
    print("\n==================================================")
    print("GÉNÉRATION DU DICTIONNAIRE 'saved_info' POUR STAT")
    print("==================================================")
   
    # 1. Calcul de la référence nominale commune
    dict_nom_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
    set_nom = systematics(dict_nom_input, **NOMINALS)
   
    scores_nom = np.array(model.predict(set_nom["data"])).ravel()
    labels_nom = np.array(set_nom["labels"]).ravel()
    weights_nom = np.array(set_nom["weights"]).ravel()

    is_sig_nom, is_bkg_nom = (labels_nom == 1.0), (labels_nom == 0.0)
    bins_fixes = np.linspace(np.min(scores_nom), np.max(scores_nom), num_bins + 1)
   
    nom_s_counts, _ = np.histogram(scores_nom[is_sig_nom], bins=bins_fixes, weights=weights_nom[is_sig_nom])
    nom_b_counts, _ = np.histogram(scores_nom[is_bkg_nom], bins=bins_fixes, weights=weights_nom[is_bkg_nom])

    # Structure initiale requise par le professeur
    saved_info = {
        "S": {},
        "B": {}
    }

    # Helper interne pour obtenir les deltas d'une configuration spécifique
    def obtenir_deltas(syst_key, direction):
        val_shift = SIGMA_SHIFTS[syst_key][direction]
        arg_name = PARAM_MAPPING[syst_key]
       
        # Recréation d'un dictionnaire propre pour l'itération
        dict_loop_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
       
        # Configuration des arguments
        syst_kwargs = NOMINALS.copy()
        if syst_key in ["tes", "jes"]:
            syst_kwargs[arg_name] = val_shift
           
        set_shift = systematics(dict_loop_input, **syst_kwargs)
       
        scores = np.array(model.predict(set_shift["data"])).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()
       
        is_sig, is_bkg = (labels == 1.0), (labels == 0.0)
       
        # Sécurité manuelle active pour le bnorm (bkg_scale)
        if syst_key == "bnorm":
            weights[is_bkg] = weights[is_bkg] * val_shift
           
        curr_s_counts, _ = np.histogram(scores[is_sig], bins=bins_fixes, weights=weights[is_sig])
        curr_b_counts, _ = np.histogram(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg])
       
        return (curr_s_counts - nom_s_counts), (curr_b_counts - nom_b_counts)

    # 2. Boucle sur chaque systématique pour extraire les paires [+1σ, -1σ]
    for syst in SIGMA_SHIFTS.keys():
        print(f"Calcul des dérives ±1σ pour : {syst.upper()}...")
       
        delta_plus_s, delta_plus_b = obtenir_deltas(syst, "plus")
        delta_minus_s, delta_minus_b = obtenir_deltas(syst, "minus")
       
        # Structuration sous forme de liste de paires [Delta(+1σ), Delta(-1σ)] pour chaque bin
        saved_info["S"][syst] = [[delta_plus_s[i], delta_minus_s[i]] for i in range(num_bins)]
        saved_info["B"][syst] = [[delta_plus_b[i], delta_minus_b[i]] for i in range(num_bins)]

    print("Génération terminée avec succès.")
    return saved_info


# ==============================================================================
# ZONE 1 : DASHBOARD VISUEL (POUR VOS VERIFICATIONS VISUELLES)
# ==============================================================================

def visualiser_impact_systematique(model, training_dict, syst_name="tes", param_min=0.7, param_max=1.3, step=0.01):
    syst_name_upper = syst_name.upper()
    val_nominale = NOMINALS.get(syst_name, 1.0)
   
    num_points = int(round((param_max - param_min) / step)) + 1
    param_values = np.linspace(param_min, param_max, num_points)
   
    num_bins = 15
    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(231); ax2 = plt.subplot(232); ax3 = plt.subplot(233)
    ax4 = plt.subplot(234); ax5 = plt.subplot(235)

    colors_syst = plt.cm.plasma(np.linspace(0, 1, len(param_values)))
    colors_bins = plt.cm.viridis(np.linspace(0, 1, num_bins))

    dict_nom_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
    set_nom = systematics(dict_nom_input, **NOMINALS)
   
    scores_nom = np.array(model.predict(set_nom["data"])).ravel()
    labels_nom = np.array(set_nom["labels"]).ravel()
    weights_nom = np.array(set_nom["weights"]).ravel()
    is_sig_nom, is_bkg_nom = (labels_nom == 1.0), (labels_nom == 0.0)
   
    bins_fixes = np.linspace(np.min(scores_nom), np.max(scores_nom), num_bins + 1)
    nom_s_counts, _ = np.histogram(scores_nom[is_sig_nom], bins=bins_fixes, weights=weights_nom[is_sig_nom])
    nom_b_counts, _ = np.histogram(scores_nom[is_bkg_nom], bins=bins_fixes, weights=weights_nom[is_bkg_nom])

    historique_diff_s = []; historique_diff_b = []

    for i, val in enumerate(param_values):
        dict_loop_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
        syst_kwargs = NOMINALS.copy()
        syst_kwargs[syst_name] = val
       
        set_shift = systematics(dict_loop_input, **syst_kwargs)
        scores = np.array(model.predict(set_shift["data"])).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()
        is_sig, is_bkg = (labels == 1.0), (labels == 0.0)
       
        if syst_name == "bkg_scale":
            weights[is_bkg] = weights[is_bkg] * val
       
        curr_s_counts, _ = np.histogram(scores[is_sig], bins=bins_fixes, weights=weights[is_sig])
        curr_b_counts, _ = np.histogram(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg])

        historique_diff_s.append(curr_s_counts - nom_s_counts)
        historique_diff_b.append(curr_b_counts - nom_b_counts)

        is_nom = np.isclose(val, val_nominale, atol=step/2)
        lw, alpha, tag = (3, 1.0, " (NOMINAL)") if is_nom else (1.5, 0.8, "")

        ax1.hist(scores[is_sig], bins=bins_fixes, weights=weights[is_sig], histtype='step', color=colors_syst[i], linewidth=lw, alpha=alpha, label=f"{syst_name_upper}={val:.2f}{tag}")
        ax2.hist(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg], histtype='step', color=colors_syst[i], linewidth=lw, alpha=alpha, label=f"{syst_name_upper}={val:.2f}")
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        ax3.plot(fpr, tpr, color=colors_syst[i], lw=lw, alpha=alpha, label=f"{syst_name_upper}={val:.2f} (AUC={auc(fpr, tpr):.3f})")

    historique_diff_s = np.array(historique_diff_s); historique_diff_b = np.array(historique_diff_b)
    for b in range(int(num_bins/2), num_bins):  
        ax4.plot(param_values, historique_diff_s[:, b], 'o-', color=colors_bins[b], lw=2, label=f"Bin {b+1}")
        ax5.plot(param_values, historique_diff_b[:, b], 'o-', color=colors_bins[b], lw=2, label=f"Bin {b+1}")

    ax1.set_title("Distribution SIGNAL", fontweight='bold'); ax1.legend(fontsize='x-small', ncol=2)
    ax2.set_title("Distribution BRUIT", fontweight='bold'); ax2.legend(fontsize='x-small', ncol=2)
    ax3.set_title("Performance ROC", fontweight='bold'); ax3.plot([0,1],[0,1],'--',color='grey'); ax3.legend(fontsize='x-small')
    for ax in [ax4, ax5]:
        ax.axhline(0, color='black', lw=1, linestyle='--'); ax.axvline(val_nominale, color='black', lw=1, linestyle='--')
        ax.set_xlabel(f"Valeur de {syst_name_upper}"); ax.legend(fontsize='x-small', ncol=2)
    ax4.set_title(f"Évolution du Signal par Bin vs {syst_name_upper}", color='red', fontweight='bold'); ax4.set_ylabel("Différence d'événements (Δ)")
    ax5.set_title(f"Évolution du Bruit par Bin vs {syst_name_upper}", color='red', fontweight='bold')
    plt.suptitle(f"Analyse Systématique : Impact du {syst_name_upper}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# ==============================================================================
# ZONE 2 : CHARGEMENT DES DONNÉES
# ==============================================================================

print("Localisation et chargement des données...")
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
data_path = os.path.join(root_dir, "blackSwan_data", "blackSwan_data.parquet")

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
mon_wrapper = Model(get_train_set=get_train_set_custom, systematics=systematics, model_type="BDT")
print("Entraînement du modèle...")
mon_wrapper.fit()

# --- ÉTAPE A : Vos Visualisations de contrôle (Optionnel, vous pouvez les commenter) ---
print("Génération des graphiques multi-shifts de contrôle...")
visualiser_impact_systematique(mon_wrapper.model, eval_dict, syst_name="bkg_scale", param_min=0.95, param_max=1.05, step=0.01)
visualiser_impact_systematique(mon_wrapper.model, eval_dict, syst_name="soft_met", param_min=-3.0, param_max=3.0, step=0.5)

# --- ÉTAPE B : Génération du dictionnaire demandé par le professeur ---
info_pour_stat = generer_saved_info(mon_wrapper.model, eval_dict, num_bins=6)

# C'est cette variable 'info_pour_stat' que vous devez transmettre à l'équipe STAT !
# Exemple pour voir la structure obtenue :
# print(info_pour_stat["S"]["tes"])

print("\nAnalyse et structuration terminées.")
print(info_pour_stat)