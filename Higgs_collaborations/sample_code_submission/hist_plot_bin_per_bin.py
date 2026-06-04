import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Importations de vos fichiers locaux
from model import Model
from HiggsML.systematics import systematics
from systematic_analysis import tes_fitter, jes_fitter, met_fitter, signal_bck, get_data


###########################################################################by beta

def visualise_tes(transformateur, data_set, model, bin_index: int, n_bins: int = 100):
    """
    Visualise l'évolution d'un bin spécifique en fonction de la valeur de TES.
    Compare la vérité terrain (scan complet) et la prédiction du transformateur.
    """
    print(f"=== Génération de la visualisation pour le bin n°{bin_index} ===")
    
    # 1. Définir une plage de TES fine pour tracer une belle courbe prédictive continue
    tes_fine_range = np.linspace(0.9, 1.1, 200)
    
    # 2. Récupérer l'histogramme nominal de validation (20%) comme base de calcul
    val_data, labels_val, weights_val = get_data(data_set)[3:]
    score_val = model.predict(val_data)
    s_sc_v, s_w_v, b_sc_v, b_w_v = signal_bck(score_val, labels_val.to_numpy(), weights_val.to_numpy())
    
    s_hist_val, _ = np.histogram(s_sc_v, bins=n_bins, range=(0, 1), weights=s_w_v, density=False)
    b_hist_val, _ = np.histogram(b_sc_v, bins=n_bins, range=(0, 1), weights=b_w_v, density=False)
    
    # 3. Calculer les prédictions du transformateur pour toute la plage de TES
    pred_signal_bin = []
    pred_bkg_bin = []
    
    for t in tes_fine_range:
        comp_s, comp_b = transformateur((s_hist_val, b_hist_val), t)
        pred_signal_bin.append(comp_s[bin_index])
        pred_bkg_bin.append(comp_b[bin_index])
        
    # 4. Échantillonner quelques "vrais" points réels pour la comparaison (ex: 15 points du détecteur)
    tes_scan_points = np.linspace(0.97, 1.03, 15)
    true_signal_bin = []
    true_bkg_bin = []
    
    print("Échantillonnage des données réelles du détecteur...")
    for t in tes_scan_points:
        syst_reel = systematics(data_set, tes=t)
        data_rl, labels_rl, weights_rl = get_data(syst_reel)[3:] # toujours les 20% validation
        
        score_rl = model.predict(data_rl)
        s_sc_rl, s_w_rl, b_sc_rl, b_w_rl = signal_bck(score_rl, labels_rl.to_numpy(), weights_rl.to_numpy())
        
        vrai_s, _ = np.histogram(s_sc_rl, bins=n_bins, range=(0, 1), weights=s_w_rl, density=False)
        vrai_b, _ = np.histogram(b_sc_rl, bins=n_bins, range=(0, 1), weights=b_w_rl, density=False)
        
        true_signal_bin.append(vrai_s[bin_index])
        true_bkg_bin.append(vrai_b[bin_index])

    # 5. Création des graphiques (côte à côte : Signal et Background)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Graphique 1 : Signal ---
    axs[0].plot(tes_fine_range, pred_signal_bin, color='blue', label='Prédiction du transformateur (Fit)', linewidth=2)
    axs[0].scatter(tes_scan_points, true_signal_bin, color='black', marker='o', s=40, zorder=3, label='Vérité Terrain (Validation Set)')
    axs[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Cas Nominal (TES=1.0)')
    axs[0].set_title(f"Signal - Évolution du Bin n°{bin_index}")
    axs[0].set_xlabel("Valeur de la TES")
    axs[0].set_ylabel("Contenu du bin (Densité)")
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend()
    
    # --- Graphique 2 : Background ---
    axs[1].plot(tes_fine_range, pred_bkg_bin, color='red', label='Prédiction du transformateur (Fit)', linewidth=2)
    axs[1].scatter(tes_scan_points, true_bkg_bin, color='black', marker='o', s=40, zorder=3, label='Vérité Terrain (Validation Set)')
    axs[1].axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Cas Nominal (TES=1.0)')
    axs[1].set_title(f"Background - Évolution du Bin n°{bin_index}")
    axs[1].set_xlabel("Valeur de la TES")
    axs[1].set_ylabel("Contenu du bin (Densité)")
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()



############################################################################by beta

def visualise_met(transformateur, data_set, model, bin_index: int, n_bins: int = 100):
    """
    Visualise l'évolution d'un bin spécifique en fonction de la valeur de MET.
    Compare la vérité terrain (scan complet) et la prédiction du transformateur.
    """
    print(f"=== Génération de la visualisation pour le bin n°{bin_index} ===")
    
    # 1. Définir une plage de MET fine pour tracer une belle courbe prédictive continue
    met_fine_range = np.linspace(-10, 10, 200)
    
    # 2. Récupérer l'histogramme nominal de validation (20%) comme base de calcul
    val_data, labels_val, weights_val = get_data(data_set)[3:]
    score_val = model.predict(val_data)
    s_sc_v, s_w_v, b_sc_v, b_w_v = signal_bck(score_val, labels_val.to_numpy(), weights_val.to_numpy())
    
    s_hist_val, _ = np.histogram(s_sc_v, bins=n_bins, range=(0, 1), weights=s_w_v, density=False)
    b_hist_val, _ = np.histogram(b_sc_v, bins=n_bins, range=(0, 1), weights=b_w_v, density=False)
    
    # 3. Calculer les prédictions du transformateur pour toute la plage de MET
    pred_signal_bin = []
    pred_bkg_bin = []
    
    for m in met_fine_range:
        comp_s, comp_b = transformateur((s_hist_val, b_hist_val), m)
        pred_signal_bin.append(comp_s[bin_index])
        pred_bkg_bin.append(comp_b[bin_index])
        
    # 4. Échantillonner quelques "vrais" points réels pour la comparaison (ex: 15 points du détecteur)
    met_scan_points = np.linspace(-10, 10, 15)
    true_signal_bin = []
    true_bkg_bin = []
    
    print("Échantillonnage des données réelles du détecteur...")
    for m in met_scan_points:
        syst_reel = systematics(data_set, soft_met=m)
        data_rl, labels_rl, weights_rl = get_data(syst_reel)[3:] # toujours les 20% validation
        
        score_rl = model.predict(data_rl)
        s_sc_rl, s_w_rl, b_sc_rl, b_w_rl = signal_bck(score_rl, labels_rl.to_numpy(), weights_rl.to_numpy())
        
        vrai_s, _ = np.histogram(s_sc_rl, bins=n_bins, range=(0, 1), weights=s_w_rl, density=False)
        vrai_b, _ = np.histogram(b_sc_rl, bins=n_bins, range=(0, 1), weights=b_w_rl, density=False)
        
        true_signal_bin.append(vrai_s[bin_index])
        true_bkg_bin.append(vrai_b[bin_index])

    # 5. Création des graphiques (côte à côte : Signal et Background)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Graphique 1 : Signal ---
    axs[0].plot(met_fine_range, pred_signal_bin, color='blue', label='Prédiction du transformateur (Fit)', linewidth=2)
    axs[0].scatter(met_scan_points, true_signal_bin, color='black', marker='o', s=40, zorder=3, label='Vérité Terrain (Validation Set)')
    axs[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Cas Nominal (MET=1.0)')
    axs[0].set_title(f"Signal - Évolution du Bin n°{bin_index}")
    axs[0].set_xlabel("Valeur de la MET")
    axs[0].set_ylabel("Contenu du bin (Densité)")
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend()
    
    # --- Graphique 2 : Background ---
    axs[1].plot(met_fine_range, pred_bkg_bin, color='red', label='Prédiction du transformateur (Fit)', linewidth=2)
    axs[1].scatter(met_scan_points, true_bkg_bin, color='black', marker='o', s=40, zorder=3, label='Vérité Terrain (Validation Set)')
    axs[1].axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Cas Nominal (MET=1.0)')
    axs[1].set_title(f"Background - Évolution du Bin n°{bin_index}")
    axs[1].set_xlabel("Valeur de la MET")
    axs[1].set_ylabel("Contenu du bin (Densité)")
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

#################################################################################################




# ==============================================================================
# ZONE 1 : DASHBOARD SYSTÉMATIQUE (DISTRIBUTIONS + DIFFÉRENCES)
# ==============================================================================

def visualiser_impact_tes(model, training_dict):
    tes_values = np.linspace(0.9, 1.0, 10)
    
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
    # ==============================================================================
# ZONE 1 : DASHBOARD SYSTÉMATIQUE (ÉVOLUTION PAR BIN vs TES)
# ==============================================================================

def visualiser_impact_tes(model, training_dict):
    tes_values = np.linspace(0.9, 1.1, 25)  # Plus de points pour une courbe plus lisse
    num_bins = 6
    
    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(231) # Dist Signal
    ax2 = plt.subplot(232) # Dist Bruit
    ax3 = plt.subplot(233) # ROC
    ax4 = plt.subplot(234) # Delta Signal vs TES (par bin)
    ax5 = plt.subplot(235) # Delta Bruit vs TES (par bin)

    # Couleurs : Plasma pour les TES (en haut), Viridis pour les Bins (en bas)
    colors_tes = plt.cm.plasma(np.linspace(0, 1, len(tes_values)))
    colors_bins = plt.cm.viridis(np.linspace(0, 1, num_bins))

    print("Calcul de la référence Nominale (TES = 1.0)...")
    set_nom = systematics(training_dict, tes=1.0)
    scores_nom = np.array(model.predict(set_nom["data"])).ravel()
    labels_nom = np.array(set_nom["labels"]).ravel()
    weights_nom = np.array(set_nom["weights"]).ravel()

    is_sig_nom, is_bkg_nom = (labels_nom == 1.0), (labels_nom == 0.0)
    
    # 10 bins fixes basés sur les scores nominaux
    bins_fixes = np.linspace(np.min(scores_nom), np.max(scores_nom), num_bins + 1)
    
    # Comptage Nominal
    nom_s_counts, _ = np.histogram(scores_nom[is_sig_nom], bins=bins_fixes, weights=weights_nom[is_sig_nom])
    nom_b_counts, _ = np.histogram(scores_nom[is_bkg_nom], bins=bins_fixes, weights=weights_nom[is_bkg_nom])

    # Tableaux pour stocker l'historique des différences pour chaque valeur de TES
    historique_diff_s = []
    historique_diff_b = []

    print("Boucle sur les valeurs de TES...")
    for i, val in enumerate(tes_values):
        set_shift = systematics(training_dict, tes=val)
        scores = np.array(model.predict(set_shift["data"])).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()

        is_sig, is_bkg = (labels == 1.0), (labels == 0.0)
        
        # Comptage actuel
        curr_s_counts, _ = np.histogram(scores[is_sig], bins=bins_fixes, weights=weights[is_sig])
        curr_b_counts, _ = np.histogram(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg])

        # Différence stockée pour la tracer plus tard
        historique_diff_s.append(curr_s_counts - nom_s_counts)
        historique_diff_b.append(curr_b_counts - nom_b_counts)

        # Style des plots de la ligne 1
        is_nom = np.isclose(val, 1.0)
        lw, alpha, tag = (3, 1.0, " (NOMINAL)") if is_nom else (1.5, 0.8, "")

        # --- PLOTS LIGNE 1 (Distributions) ---
        ax1.hist(scores[is_sig], bins=bins_fixes, weights=weights[is_sig], histtype='step', 
                 color=colors_tes[i], linewidth=lw, alpha=alpha, label=f"TES={val:.2f}{tag}")
        ax2.hist(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg], histtype='step', 
                 color=colors_tes[i], linewidth=lw, alpha=alpha, label=f"TES={val:.2f}")
        
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        ax3.plot(fpr, tpr, color=colors_tes[i], lw=lw, alpha=alpha, label=f"TES={val:.2f} (AUC={auc(fpr, tpr):.3f})")

    # --- ÉTAPE 3 : TRACÉ DES DÉRIVES (LIGNE 2) ---
    # On transforme nos listes en tableaux Numpy pour pouvoir lire colonne par colonne (bin par bin)
    historique_diff_s = np.array(historique_diff_s)
    historique_diff_b = np.array(historique_diff_b)

    for b in range(int(num_bins/2),num_bins):   # on ne fait qu'a partir de la moitié des bins pour éviter les bins de trop faibles score. 
        # On extrait la colonne 'b', qui contient l'évolution du Bin 'b' pour les 7 valeurs de TES
        evolution_bin_s = historique_diff_s[:, b]
        evolution_bin_b = historique_diff_b[:, b]

        ax4.plot(tes_values, evolution_bin_s, 'o-', color=colors_bins[b], lw=2, label=f"Bin {b+1}")
        ax5.plot(tes_values, evolution_bin_b, 'o-', color=colors_bins[b], lw=2, label=f"Bin {b+1}")

    # --- MISE EN FORME ---
    ax1.set_title("Distribution SIGNAL", fontweight='bold'); ax1.legend(fontsize='x-small', ncol=2)
    ax2.set_title("Distribution BRUIT", fontweight='bold'); ax2.legend(fontsize='x-small', ncol=2)
    ax3.set_title("Performance ROC", fontweight='bold'); ax3.plot([0,1],[0,1],'--',color='grey'); ax3.legend(fontsize='x-small')
    
    # Axes des différences vs TES
    for ax in [ax4, ax5]:
        ax.axhline(0, color='black', lw=1, linestyle='--') # Zéro (pas de différence)
        ax.axvline(1.0, color='black', lw=1, linestyle='--') # TES nominal (doit croiser zéro ici)
        ax.set_xlabel("Valeur de TES")
        ax.legend(fontsize='x-small', ncol=2)

    ax4.set_title("Évolution du Signal par Bin vs TES", color='red', fontweight='bold')
    ax4.set_ylabel("Différence d'événements (Δ)")

    ax5.set_title("Évolution du Bruit par Bin vs TES", color='red', fontweight='bold')

    plt.suptitle("Analyse Systématique : Interpolation (Template Morphing)", fontsize=18)
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
# visualiser_impact_tes(mon_wrapper.model, eval_dict)

# APPEL DES FITTERS (SUR LES 20% RESTANTS)
print("Calcul de la paramétrisation TES (sur l'Eval Set)...")

transformateur_tes = tes_fitter(
    model=mon_wrapper.model,
    train_set=mon_wrapper.training_set,
    n_bins=100
)

transformateur_met = met_fitter(
    model=mon_wrapper.model,
    train_set=mon_wrapper.training_set,
    n_bins=100
)

visualise_tes(
    transformateur=transformateur_tes,
    data_set=mon_wrapper.training_set,
    model=mon_wrapper.model,
    bin_index=26,
    n_bins=100
)

visualise_met(
    transformateur=transformateur_met,
    data_set=mon_wrapper.training_set,
    model=mon_wrapper.model,
    bin_index=26,
    n_bins=100
)

print("Analyse terminée.")

