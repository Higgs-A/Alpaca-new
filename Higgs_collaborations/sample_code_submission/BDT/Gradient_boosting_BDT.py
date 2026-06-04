from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class Sklearn_GBDT:
    def __init__(self):
        self.scaler = StandardScaler()
        # Configuration moderne pour le GBDT de Scikit-Learn
        self.model = HistGradientBoostingClassifier(
            max_iter=1000,
            learning_rate=0.05,
            max_depth=9,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42
        )

    def fit(self, train_data, labels, weights=None):
        # Standardisation 
        X_tr = self.scaler.fit_transform(train_data)
        
        # Gestion des poids 
        # On égalise la somme des poids Signal et Background
        if weights is not None:
            w_tr = weights.copy()
            class_weights = (w_tr[labels == 0].sum(), w_tr[labels == 1].sum())
            for i in [0, 1]:
                if class_weights[i] > 0:
                    w_tr[labels == i] *= max(class_weights) / class_weights[i]
        else:
            w_tr = None

        # Entraînement
        self.model.fit(X_tr, labels, sample_weight=w_tr)
        

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]