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
from training_tree import training_tree
from boosted_decision_tree import XGBoost_BDT
from LightGBM_BDT import LightGBM_BDT
from Gradient_boosting_BDT import Sklearn_GBDT


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

def simple_significance(s_in, b_in):
    """Calcule la significativité Z = S / sqrt(B)"""
    # On ajoute un epsilon pour éviter la division par zéro
    b = np.where(b_in <= 0, 1e-10, b_in)
    return s_in / np.sqrt(b)


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

    return amsasimov(s_cumul, b_cumul), simple_significance(s_cumul, b_cumul)

def significance_score(y_true, y_score, sample_weight=None):
    """Retourne la signification maximale trouvée (Z max)"""
    z_ams, _ = significance_vscore(y_true, y_score, sample_weight)
    return np.max(z_ams)


def get_model_choice():
    print("Sélection du modèle pour les courbes :")
    choix = input("Taper 'xgb' ou 'lgbm' ou 'sklearn' : ").lower()
    if choix == 'xgb': return XGBoost_BDT
    if choix == 'lgbm': return LightGBM_BDT
    if choix == 'sklearn' : return Sklearn_GBDT
    return XGBoost_BDT # Par défaut

model_class = get_model_choice()

# --- 1. COURBE ROC AUC ---
if __name__ == "__main__":

    X_train, X_test,y_train, y_test, w_train,weights_test_arr,bdt, y_pred_test=training_tree(model_class=model_class)

    model_name = model_class.__name__.replace("_BDT", "")

    plt.figure(figsize=(8, 6))
    auc_test = roc_auc_score(
        y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr
    )
    fpr, tpr, _ = roc_curve(
        y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr
    )

    plt.plot(fpr, tpr, color="darkgreen", lw=2, 
             label=f"{model_name} (AUC = {auc_test:.3f})")
    
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
    
    vams, vsimple = significance_vscore(y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr)
    
    significance_max = significance_score(
        y_true=y_test, y_score=y_pred_test, sample_weight=weights_test_arr
    )
    x_thresholds = np.linspace(0, 1, num=len(vams))

    plt.figure(figsize=(8, 6))
    plt.plot(x_thresholds, vams, color="darkgreen", lw=2,label=f"{model_name} (Z max = {significance_max:.2f})"
             )
    plt.plot(x_thresholds, vsimple, color="red", lw=2, linestyle="--", label=f"{model_name} - Z = S/sqrt(B)")
    
    plt.title("Comparaison : AMS vs Z = S/sqrt(B)")
    plt.xlabel("Threshold")
    plt.ylabel("Significance")
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
        lc_model = model_class() 
        
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
