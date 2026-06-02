import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# Imports de tes classes et fonctions locales
from boosted_decision_tree import BoostedDecisionTree
from get_data import get_clean_splits

# ==========================================
# 1. FONCTION D'OPTIMISATION DU LEARNING RATE
# ==========================================

def optimize_learning_rate(X_train, y_train, weights_train):
    """
    Teste plusieurs valeurs de learning rate via une validation croisée
    et affiche le graphique de performance pour choisir le meilleur.
    """
    # Liste des learning rates à tester
    learning_rates_to_test = [0.01, 0.03, 0.05, 0.1, 0.2]
    
    # Validation croisée à 3 plis pour ne pas surcharger les calculs
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Dictionnaire pour stocker les scores de chaque pli
    lr_results = {lr: [] for lr in learning_rates_to_test}
    
    print(f"\n--- Début de l'optimisation du Learning Rate ({n_splits} plis) ---")
    
    # On boucle sur chaque paramètre
    for lr in learning_rates_to_test:
        print(f"\n[Test] Évaluation avec learning_rate = {lr}...")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # Découpage des plis (conversion numpy synchrone)
            X_tr_fold = X_train.iloc[train_idx]
            y_tr_fold = y_train.iloc[train_idx]
            w_tr_fold = weights_train.iloc[train_idx]
            
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # Ajustement d'échelle des poids pour le pli de validation
            w_val_fold = weights_train.iloc[val_idx].to_numpy().copy()
            w_val_fold *= n_splits 
            
            # Instanciation d'un modèle vierge
            test_model = BoostedDecisionTree()
            
            # Injection dynamique du learning rate dans le sous-modèle XGBoost
            test_model.model.set_params(learning_rate=lr)
            
            # Entraînement sur le pli
            start_time = time.time()
            test_model.fit(X_tr_fold, y_tr_fold, weights=w_tr_fold)
            duration = time.time() - start_time
            
            # Évaluation
            y_pred_val = test_model.predict(X_val_fold)
            auc_score = roc_auc_score(y_val_fold, y_pred_val, sample_weight=w_val_fold)
            fold_scores.append(auc_score)
            
            print(f"  -> Pli {fold + 1}/{n_splits} | AUC: {auc_score:.4f} | Temps: {duration:.1f}s")
            
        lr_results[lr] = fold_scores
        print(f"==> Score moyen pour lr={lr} : {np.mean(fold_scores):.4f}")

    # --- ANALYSE DES RÉSULTATS & TRACÉ ---
    best_lr = max(lr_results, key=lambda k: np.mean(lr_results[k]))
    print(f"\n[RÉSULTAT] Le meilleur learning_rate est {best_lr} (AUC moyen = {np.mean(lr_results[best_lr]):.4f})")
    
    means = [np.mean(lr_results[lr]) for lr in learning_rates_to_test]
    stds = [np.std(lr_results[lr]) for lr in learning_rates_to_test]
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(learning_rates_to_test, means, yerr=stds, fmt='o-', color='darkgreen', ecolor='red', capsize=5, lw=2)
    plt.title("Optimisation du Learning Rate (Validation Croisée K-Fold)")
    plt.xlabel("Learning Rate (Échelle Log)")
    plt.ylabel("Validation ROC AUC Moyen")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xscale('log')
    plt.xticks(learning_rates_to_test, labels=[str(lr) for lr in learning_rates_to_test])
    plt.show()


# ==========================================
# 2. BLOC EXÉCUTABLE PRINCIPAL
# ==========================================

if __name__ == "__main__":
    np.random.seed(31415)
    
    # 1. Récupération directe des données via ton module externe 'get_data.py'
    X_train, X_test, y_train, y_test, weights_train, weights_test = get_clean_splits()
    
    # Optionnel de sécurité : Si l'entraînement complet à 1.4 million de lignes prend trop de temps,
    # tu peux décommenter les 3 lignes ci-dessous pour faire ton étude de LR sur un échantillon plus rapide (ex: 150k lignes)
    # n_sample = 150000
    # X_train, y_train, weights_train = X_train.iloc[:n_sample], y_train.iloc[:n_sample], weights_train.iloc[:n_sample]

    # 2. Lancement direct de la recherche du meilleur learning rate
    optimize_learning_rate(X_train, y_train, weights_train)