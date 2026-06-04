
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from boosted_decision_tree import XGBoost_BDT
from get_data import get_clean_splits

def grid_search_lr_and_stopping(X_train, y_train, weights_train):
    """
    Exécute une recherche sur grille complète croisant chaque Learning Rate
    avec chaque valeur d'Early Stopping via une validation croisée à 3 plis.
    """
    # Définition des listes de paramètres à croiser de manière exhaustive
    learning_rates = [0.01, 0.03, 0.05, 0.1, 0.2]
    stopping_rounds = [15, 30, 50, 80]
    
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Matrice pour stocker les scores moyens (index: LR, colonnes: Stopping)
    score_matrix = np.zeros((len(learning_rates), len(stopping_rounds)))
    
    print(f"--- Début de la Grid Search Exhaustive ({n_splits} plis) ---")
    print(f"Total de combinaisons à tester : {len(learning_rates) * len(stopping_rounds)}")

    # Boucle sur la grille
    for i, lr in enumerate(learning_rates):
        for j, stopping in enumerate(stopping_rounds):
            print(f"\n[Test] LR: {lr} | Early Stopping: {stopping} rounds")
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                # Découpage des plis
                X_tr_fold = X_train.iloc[train_idx]
                y_tr_fold = y_train.iloc[train_idx]
                w_tr_fold = weights_train.iloc[train_idx]
                
                X_val_fold = X_train.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                # Réalignement des poids de validation du pli
                w_val_fold = weights_train.iloc[val_idx].to_numpy().copy()
                w_val_fold *= n_splits
                
                # Modèle vierge
                test_model = XGBoost_BDT()
                test_model.model.set_params(
                    learning_rate=lr,
                    early_stopping_rounds=stopping
                )
                
                # Entraînement
                test_model.fit(X_tr_fold, y_tr_fold, weights=w_tr_fold)
                
                # Évaluation
                y_pred_val = test_model.predict(X_val_fold)
                auc_score = roc_auc_score(y_val_fold, y_pred_val, sample_weight=w_val_fold)
                fold_scores.append(auc_score)
                
            mean_auc = np.mean(fold_scores)
            score_matrix[i, j] = mean_auc
            print(f"==> AUC Moyen pour (LR={lr}, Stop={stopping}) : {mean_auc:.4f}")

    # Transformation en DataFrame pour l'affichage graphique
    df_scores = pd.DataFrame(score_matrix, index=learning_rates, columns=stopping_rounds)
    
    # Trouve les coordonnées du maximum absolu
    best_lr = df_scores.max(axis=1).idxmax()
    best_stop = df_scores.max(axis=0).idxmax()
    best_auc = df_scores.values.max()
    
    print("\n" + "="*45)
    print(" RÉSULTAT FINAL DE LA DE LA RECHERCHE SUR GRILLE ")
    print("="*45)
    print(f"Meilleur compromis trouvé au global :")
    print(f" -> Learning Rate (lr)       : {best_lr}")
    print(f" -> Early Stopping (rounds)  : {best_stop}")
    print(f" -> Validation ROC AUC Moyen : {best_auc:.4f}")
    print("="*45)
    
    # AFFICHAGE DE LA HEATMAP 
    plt.figure(figsize=(9, 6))
    sns.heatmap(df_scores, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Validation ROC AUC'})
    plt.title("Grid Search : Interaction Learning Rate vs Early Stopping", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Early Stopping Rounds", fontsize=10)
    plt.ylabel("Learning Rate", fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(31415)
    
    # 1. Chargement des données globales
    X_train, _, y_train, _, weights_train, _ = get_clean_splits()
    
    # 2. Sous-échantillon pour garder un temps de calcul raisonnable 
    n_sample = 150000
    print(f"Sélection de {n_sample} événements pour la Grid Search...")
    X_tr_opti = X_train.iloc[:n_sample]
    y_tr_opti = y_train.iloc[:n_sample]
    w_tr_opti = weights_train.iloc[:n_sample]

    # 3. Lancement de la grille exhaustive
    grid_search_lr_and_stopping(X_tr_opti, y_tr_opti, w_tr_opti)