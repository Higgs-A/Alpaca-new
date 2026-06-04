import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Imports de vos fichiers d'origine
from get_data import get_clean_splits
from boosted_decision_tree import BoostedDecisionTree
from courbes import significance_score, significance_vscore, roc_auc_score, roc_curve

def evaluate_fold_parameters(X_train_full, y_train_full, w_train_full, params, n_splits=3):
    """Cross-Validation en version ultra-légère pour accélérer Optuna."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_signif = []
    
    X_arr = np.array(X_train_full)
    y_arr = np.array(y_train_full)
    w_arr = np.array(w_train_full) if w_train_full is not None else None

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
        
        if w_arr is not None:
            w_tr = np.copy(w_arr[train_idx])
            w_val = np.copy(w_arr[val_idx])
            
            class_weights_tr = (w_tr[y_tr == 0].sum(), w_tr[y_tr == 1].sum())
            for i in [0, 1]:
                if class_weights_tr[i] > 0:
                    w_tr[y_tr == i] *= max(class_weights_tr) / class_weights_tr[i]
            for i in [0, 1]:
                w_val[y_val == i] *= n_splits
        else:
            w_tr = w_val = None
            
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        
        model = XGBClassifier(
            **params,
            n_estimators=50,            
            early_stopping_rounds=8,    
            eval_metric="logloss",
            random_state=42,
        )
        
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val] if w_val is not None else None,
            verbose=False
        )
        
        preds_val = model.predict_proba(X_val)[:, 1]
        fold_sig = significance_score(y_val, preds_val, sample_weight=w_val)
        scores_signif.append(fold_sig)
        
    return np.mean(scores_signif)

def objective(trial, X, y, weights):
    """Espace de recherche d'hyperparamètres."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.75, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 1.0),
    }
    return evaluate_fold_parameters(X, y, weights, params, n_splits=3)

def main():
    # 1. Chargement des données
    X_train_full, X_test, y_train_full, y_test, w_train_full, w_test = get_clean_splits()
    
    taille_opt = 100000
    print(f"[*] Extraction de {taille_opt} lignes pour Optuna...")
    
    if isinstance(X_train_full, pd.DataFrame):
        X_train_opt = X_train_full.iloc[:taille_opt]
        y_train_opt = y_train_full.iloc[:taille_opt]
        w_train_opt = w_train_full.iloc[:taille_opt] if w_train_full is not None else None
    else:
        X_train_opt = X_train_full[:taille_opt]
        y_train_opt = y_train_full[:taille_opt]
        w_train_opt = w_train_full[:taille_opt] if w_train_full is not None else None
    
    print("[*] LANCEMENT DE LA CROSS-OPTIMISATION RAPIDE (OPTUNA)")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    n_trials = 10  
    print(f"-> Recherche de la meilleure combinaison sur {n_trials} itérations...")
    study.optimize(lambda trial: objective(trial, X_train_opt, y_train_opt, w_train_opt), n_trials=n_trials)
    
    # 2. RECUPERATION ET AFFICHAGE DES MEILLEURS PARAMETRES
    best_params = study.best_params
    print("\n" + "="*50)
    print("      🏆 MEILLEURS PARAMÈTRES RETENUS POUR LES COURBES 🏆")
    print("="*50)
    print(f"Signification Maximale Attendue (Z moyen) : {study.best_value:.4f}")
    print("-"*50)
    for param_name, param_value in best_params.items():
        if isinstance(param_value, float):
            print(f" 👉 {param_name:<18} : {param_value:.5f}")
        else:
            print(f" 👉 {param_name:<18} : {param_value}")
    print("="*50 + "\n")

    # 3. Entraînement du modèle final (300 000 lignes)
    taille_finale = 300000
    print(f"[*] Entraînement du modèle final sur {taille_finale} lignes...")
    
    if isinstance(X_train_full, pd.DataFrame):
        X_train_f = X_train_full.iloc[:taille_finale]
        y_train_f = y_train_full.iloc[:taille_finale]
        w_train_f = w_train_full.iloc[:taille_finale] if w_train_full is not None else None
    else:
        X_train_f = X_train_full[:taille_finale]
        y_train_f = y_train_full[:taille_finale]
        w_train_f = w_train_full[:taille_finale] if w_train_full is not None else None

    bdt_final = BoostedDecisionTree()
    bdt_final.model.set_params(**best_params)
    bdt_final.model.set_params(n_estimators=200, early_stopping_rounds=20)
    
    bdt_final.fit(X_train_f, y_train_f, weights=w_train_f)
    print("[*] Modèle final entraîné avec succès !")

    # 4. Évaluation sur le jeu de Test
    predictions = bdt_final.predict(X_test)
    weights_test_arr = np.array(w_test)

    # --- GENERATION DES GRAPHES AVEC DETAILS DES PARAMETRES ---
    print("\n[*] Génération immédiate des courbes de performance...")
    
    # Création d'un texte résumé des paramètres pour les graphiques
    texte_params = f"Paramètres optimisés :\n• max_depth: {best_params['max_depth']}\n• learning_rate: {best_params['learning_rate']:.3f}\n• subsample: {best_params['subsample']:.2f}\n• colsample_bytree: {best_params['colsample_bytree']:.2f}"

    # Graphe 1 : Courbe ROC
    plt.figure(figsize=(8, 5))
    auc_test = roc_auc_score(y_true=y_test, y_score=predictions, sample_weight=weights_test_arr)
    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=predictions, sample_weight=weights_test_arr)

    plt.plot(fpr, tpr, color="darkgreen", lw=2, label="XGBoost Optimisé (AUC = {:.3f})".format(auc_test))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    
    # Ajout de l'encadré des paramètres sur le graphique
    plt.text(0.55, 0.15, texte_params, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=9)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Background Efficiency")
    plt.ylabel("Signal Efficiency")
    plt.title("ROC Curve (Modèle après Cross-Optimisation)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Graphe 2 : Courbe de Significativité (AMS)
    plt.figure(figsize=(8, 5))
    vamsasimov_res = significance_vscore(y_true=y_test, y_score=predictions, sample_weight=weights_test_arr)
    significance_max = np.max(vamsasimov_res)
    x_thresholds = np.linspace(0, 1, num=len(vamsasimov_res))

    plt.plot(x_thresholds, vamsasimov_res, color="darkgreen", lw=2, label="XGBoost (Z max = {:.2f})".format(significance_max))
    
    # Ajout de l'encadré des paramètres sur le graphique
    plt.text(0.05, significance_max * 0.4, texte_params, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=9)
    
    plt.title("BDT Significance vs Threshold (Après Cross-Optimisation)")
    plt.xlabel("Threshold")
    plt.ylabel("Significance (Z)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()

if __name__ == "__main__":
    main()