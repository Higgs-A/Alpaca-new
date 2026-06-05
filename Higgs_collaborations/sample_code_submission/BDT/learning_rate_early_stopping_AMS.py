import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from boosted_decision_tree import XGBoost_BDT
from get_data import get_clean_splits


def amsasimov(s_in, b_in):
    s = np.copy(s_in)
    b = np.copy(b_in)
    s = np.where((b_in == 0), 0.0, s_in)
    b = np.where((b_in == 0), 1.0, b)
    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    return np.where((s < 0) | (b < 0), np.nan, ams)

def get_best_significance_and_threshold(y_true, y_score, sample_weight):
    #Calcule le vecteur de signification et retourne :
    #- Le Z-score maximum
    # - Le seuil (threshold) optimal associé
    bins = np.linspace(0, 1.0, 101)
    s_hist, _ = np.histogram(y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1])
    b_hist, _ = np.histogram(y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0])

    # Sommes cumulées inversées (du plus haut score au plus bas)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    v_ams = amsasimov(s_cumul, b_cumul)
    
    # Trouver le max et son index (le bin correspondant)
    idx_max = np.nanargmax(v_ams)
    best_z = v_ams[idx_max]
    best_threshold = bins[idx_max] # Donne une approximation du seuil optimal entre 0 et 1
    
    return best_z, best_threshold


def grid_search_ams_and_threshold(X_train, y_train, weights_train):
    # Exécute une Grid Search pour optimiser l'AMS (Signification Z)
    # et cartographier le comportement du seuil optimal de coupure.
    learning_rates = [0.005,0.01]
    stopping_rounds = [5,10,15]
    
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # On prépare deux matrices : une pour le score Z, une pour le seuil (threshold)
    z_matrix = np.zeros((len(learning_rates), len(stopping_rounds)))
    threshold_matrix = np.zeros((len(learning_rates), len(stopping_rounds)))
    
    print(f"--- Grid Search Axée sur la Signification d'Asimov ({n_splits} plis) ---")

    for i, lr in enumerate(learning_rates):
        for j, stopping in enumerate(stopping_rounds):
            print(f"\n[Test] LR: {lr} | Early Stopping: {stopping}")
            fold_z = []
            fold_thresh = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_tr_fold = X_train.iloc[train_idx]
                y_tr_fold = y_train.iloc[train_idx]
                w_tr_fold = weights_train.iloc[train_idx]
                
                X_val_fold = X_train.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                w_val_fold = weights_train.iloc[val_idx].to_numpy().copy()
                w_val_fold *= n_splits
                
                # Modèle
                test_model = XGBoost_BDT()
                test_model.model.set_params(learning_rate=lr, early_stopping_rounds=stopping)
                test_model.fit(X_tr_fold, y_tr_fold, weights=w_tr_fold)
                
                # Prédiction 
                y_pred_val = test_model.predict(X_val_fold)
                
                # Calcul AMS et seuil 
                z_max, thresh_opt = get_best_significance_and_threshold(
                    y_true=y_val_fold.to_numpy(), 
                    y_score=y_pred_val, 
                    sample_weight=w_val_fold
                )
                fold_z.append(z_max)
                fold_thresh.append(thresh_opt)
                
            mean_z = np.mean(fold_z)
            mean_thresh = np.mean(fold_thresh)
            
            z_matrix[i, j] = mean_z
            threshold_matrix[i, j] = mean_thresh
            print(f"==> Résultat moyen : Z Max = {mean_z:.2f} sigma (Seuil optimal à {mean_thresh:.2f})")

    # DataFrames pour l'affichage
    df_z = pd.DataFrame(z_matrix, index=learning_rates, columns=stopping_rounds)
    df_thresh = pd.DataFrame(threshold_matrix, index=learning_rates, columns=stopping_rounds)
    
    # Affichage de la heatmap des scores Z (AMS)
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_z, annot=True, fmt=".2f", cmap="Purples", cbar_kws={'label': "Signification d'Asimov (Z)"})
    plt.title("Grid Search : Signification d'Asimov Maximale (Z)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Early Stopping Rounds")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.show()

    # Affichage de la heatmap des seuils optimaux
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_thresh, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={'label': "Seuil de Coupure Optimal (0 à 1)"})
    plt.title("Evolution du Seuil de Coupure Optimal selon les paramètres", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Early Stopping Rounds")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(31415)
    
    # Chargement
    X_train, _, y_train, _, weights_train, _ = get_clean_splits()
    
    # Échantillonnage à 100k ou 150k pour la vitesse
    n_sample = 100000
    X_tr_opti = X_train.iloc[:n_sample]
    y_tr_opti = y_train.iloc[:n_sample]
    w_tr_opti = weights_train.iloc[:n_sample]

    # Lancement
    grid_search_ams_and_threshold(X_tr_opti, y_tr_opti, w_tr_opti)