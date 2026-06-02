import os
import time
from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from get_data import get_clean_splits
from boosted_decision_tree import BoostedDecisionTree
from training_tree import training_tree



#Calcul de la signification maximale

def amsasimov(s_in, b_in):
    """Calcule la signification d'Asimov (arXiv:1007.1727 eq. 97)"""
    s = np.copy(s_in)
    b = np.copy(b_in)
    s = np.where((b_in == 0), 0.0, s_in)
    b = np.where((b_in == 0), 1.0, b)

    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    ams = np.where((s < 0) | (b < 0), np.nan, ams)
    if np.isscalar(s_in):
        return float(ams)
    else:
        return ams


def significance_vscore(y_true, y_score, sample_weight=None):
    """Calcule le vecteur de signification Z selon le seuil de coupure"""
    if sample_weight is None:
        sample_weight = np.full(len(y_true), 1.0)

    bins = np.linspace(0, 1.0, 101)
    s_hist, _ = np.histogram(
        y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
    )
    b_hist, _ = np.histogram(
        y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
    )

    # Sommes cumulées inversées (du plus haut score au plus bas)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    return amsasimov(s_cumul, b_cumul)


def significance_score(y_true, y_score, sample_weight=None):
    """Retourne la signification maximale trouvée (Z max)"""
    vsig = significance_vscore(y_true, y_score, sample_weight)
    return np.max(vsig)




# --- 1. COURBE ROC AUC ---
if __name__ == "__main__":
    y_pred_test=training_tree()[1]
    X_train, X_test,y_train, y_test, w_train,weights_test_arr=get_clean_splits()

    plt.figure(figsize=(8, 6))
    auc_test = roc_auc_score(
        y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr
    )
    fpr, tpr, _ = roc_curve(
        y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr
    )

    plt.plot(
        fpr, tpr, color="darkgreen", lw=2, label="XGBoost Classe (AUC = {:.3f})".format(auc_test)
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Background Efficiency")
    plt.ylabel("Signal Efficiency")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # --- 2. COURBE DE SIGNIFICATIVITÉ ---
    plt.figure(figsize=(8, 6))
    vamsasimov_res = significance_vscore(
        y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr
    )
    significance_max = significance_score(
        y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr
    )

    x_thresholds = np.linspace(0, 1, num=len(vamsasimov_res))

    plt.plot(
        x_thresholds,
        vamsasimov_res,
        color="darkgreen",
        lw=2,
        label="XGBoost (Z max = {:.2f})".format(significance_max),
    )
    plt.title("BDT Significance vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Significance (Z)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # --- 3. LEARNING CURVE ---
    # Utilisation de la logique de boucle manuelle sur les tailles pour intégrer votre classe
    train_sizes = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
    ntrains = []
    test_aucs = []
    train_aucs = []

    print("\n--- Génération de la Learning Curve ---")
    for t_size in train_sizes:
        ntrain = int(len(X_train) * t_size)
        print(f"Calcul pour {ntrain} événements...")
        ntrains.append(ntrain)

        # Ré-instanciation d'un modèle vierge à chaque étape de taille pour éviter les effets de mémoire
        lc_model = BoostedDecisionTree()
        
        # Entraînement partiel
        X_tr_sub = X_train.iloc[:ntrain] if isinstance(X_train, pd.DataFrame) else X_train[:ntrain]
        y_tr_sub = y_train.iloc[:ntrain] if isinstance(y_train, pd.Series) else y_train[:ntrain]
        w_tr_sub = w_train.iloc[:ntrain] if isinstance(w_train, pd.Series) else w_train[:ntrain]
        
        lc_model.fit(X_tr_sub, y_tr_sub, weights=w_tr_sub)

        # Évaluation sur l'ensemble de Test fixe
        y_pred_test_lc = lc_model.predict(X_test)
        auc_test_lc = roc_auc_score(
            y_true=y_test, y_score=y_pred_test_lc, sample_weight=weights_test_arr
        )
        test_aucs.append(auc_test_lc)

        # Évaluation sur la portion de Train courante
        y_pred_train_lc = lc_model.predict(X_tr_sub)
        
        # Normalisation locale temporaire des poids de train pour le scoring de la courbe d'apprentissage
        w_tr_sub_arr = np.array(w_tr_sub)
        auc_train_lc = roc_auc_score(
            y_true=y_tr_sub, y_score=y_pred_train_lc, sample_weight=w_tr_sub_arr
        )
        train_aucs.append(auc_train_lc)

    # Tracé final de la Learning Curve
    plt.figure(figsize=(8, 6))
    plt.plot(ntrains, train_aucs, "o-", color="blue", label="Train AUC")
    plt.plot(ntrains, test_aucs, "o-", color="orange", label="Test AUC")
    plt.title("Learning Curve (Modèle Autonome avec Early Stopping)")
    plt.xlabel("Number of Training Events")
    plt.ylabel("ROC AUC Score")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
