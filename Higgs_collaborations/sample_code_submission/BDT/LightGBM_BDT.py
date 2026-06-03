from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class LightGBM_BDT:
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Configuration spécifique LightGBM
        self.model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=9,
            bagging_fraction=0.8,    # Équivalent du subsample
            feature_fraction=0.8,    # Équivalent du colsample_bytree
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1               # Pour éviter les logs inutiles
        )

    def fit(self, train_data, labels, weights=None):
        val_size = 0.1
        
        # 1. Split avec gestion des poids (Identique à ta logique)
        if weights is not None:
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                train_data, labels, weights,
                test_size=val_size, random_state=42, stratify=labels
            )
            # Normalisation des poids (logique du prof)
            class_weights_tr = (w_tr[y_tr == 0].sum(), w_tr[y_tr == 1].sum())
            for i in [0, 1]:
                if class_weights_tr[i] > 0:
                    w_tr[y_tr == i] *= max(class_weights_tr) / class_weights_tr[i]
            
            for i in [0, 1]:
                w_val[y_val == i] *= 1 / val_size
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                train_data, labels, test_size=val_size, random_state=42, stratify=labels
            )
            w_tr = w_val = None

        # 2. Standardisation
        X_tr = self.scaler.fit_transform(X_tr)
        X_val = self.scaler.transform(X_val)

        # 3. Entraînement avec Early Stopping (Callback moderne)
        self.model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=60, verbose=False)]
        )

        print(f"Entraînement terminé. Meilleure itération : {self.model.best_iteration_}")

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        # Retourne les probabilités pour la classe 1
        return self.model.predict_proba(test_data)[:, 1]