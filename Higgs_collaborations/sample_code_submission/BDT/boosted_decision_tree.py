from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <-- AJOUTÉ : Pour la standardisation
import numpy as np

class BoostedDecisionTree:
    """
    Classificateur XGBoost autonome pour le dataset Higgs.
    Intègre sa propre standardisation, l'early stopping et applique
    strictement la philosophie de gestion des poids du professeur.
    """

    def __init__(self):
        # AJOUTÉ : Le scaler est maintenant un outil interne de la classe
        self.scaler = StandardScaler()
        
        self.model = XGBClassifier(
            n_estimators=1000,        # L'early stopping l'arrêtera bien avant
            learning_rate=0.05,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,            # Régularisation L1 (évite l'overfitting)
            reg_lambda=1.0,           # Régularisation L2
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=60, # S'arrête si la validation ne s'améliore plus pendant 60 arbres
            random_state=42,
        )

    def fit(self, train_data, labels, weights=None):
        # On définit la taille du set de validation interne (10%)
        val_size = 0.1
        
        # 1. Création du set de validation interne pour l'early stopping
        if weights is not None:
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                train_data, labels, weights,
                test_size=val_size, random_state=42, stratify=labels
            )
            
            # --- ALIGNEMENT DES POIDS D'ENTRAÎNEMENT (w_tr) ---
            # On égalise la somme des poids de Signal et de Background sur les 90%
            class_weights_tr = (w_tr[y_tr == 0].sum(), w_tr[y_tr == 1].sum())
            for i in [0, 1]:
                if class_weights_tr[i] > 0:
                    w_tr[y_tr == i] *= max(class_weights_tr) / class_weights_tr[i]
            
            # --- AJOUTÉ : ALIGNEMENT STRICT DE COHÉRENCE POUR LE SET DE VALIDATION (w_val) ---
            # On applique la formule du prof : on compense la perte de taille due à l'échantillonnage.
            # Comme le set de validation ne prend que 10% des données, on multiplie les poids par 1 / 0.1 (soit 10)
            for i in [0, 1]:
                w_val[y_val == i] *= 1 / val_size
                    
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                train_data, labels,
                test_size=val_size, random_state=42, stratify=labels
            )
            w_tr = w_val = None

        # 2. AJOUTÉ : LA STANDARDISATION INTERNE AUTOMATIQUE
        # Le scaler apprend les moyennes/variances UNIQUEMENT sur le set d'entraînement (90%)
        X_tr = self.scaler.fit_transform(X_tr)
        # Il applique ensuite la même transformation sur le set de validation (10%)
        X_val = self.scaler.transform(X_val)

        # 3. Gestion du déséquilibre si aucun poids physique n'est fourni (sécurité)
        if weights is None:
            neg = np.sum(y_tr == 0)
            pos = np.sum(y_tr == 1)
            self.model.set_params(scale_pos_weight=neg / pos if pos > 0 else 1.0)

        # 4. Entraînement final avec Early Stopping
        self.model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val] if w_val is not None else None,
            verbose=False,
        )

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]
