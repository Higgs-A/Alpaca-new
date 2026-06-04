
from get_data import get_clean_splits
from boosted_decision_tree import XGBoost_BDT
from LightGBM_BDT import LightGBM_BDT
from Gradient_boosting_BDT import Sklearn_GBDT
import os
import joblib


def training_tree(model_class=None, model_path=None):
    """
    Entraîne ou charge un modèle. 
    Si model_path est None, il crée un nom basé sur la classe.
    """
    if model_class is None:
        print("Aucun modèle spécifié. Veuillez choisir :")
        while True:
            choix = input("Modèle (xgb/lgbm/sklearn) : ").lower()
            if choix == 'xgb':
                model_class = XGBoost_BDT
                break
            elif choix == 'lgbm':
                model_class = LightGBM_BDT
                break
            elif choix == 'sklearn':
                model_class = Sklearn_GBDT
                break
            print("Erreur : veuillez taper 'xgb' ou 'lgbm'.")

    if model_path is None:
        model_path = f"{model_class.__name__}_checkpoint.joblib"

    # Chargement si le checkpoint existe
    if os.path.exists(model_path):
        reponse = input(f"Checkpoint trouvé : {model_path}. Charger ? (o/n) : ").lower()
        if reponse == 'o':
            print(f"Chargement du contexte complet depuis {model_path}...")
            cp = joblib.load(model_path)
            return cp['X_train'], cp['X_test'], cp['y_train'], cp['y_test'], \
                   cp['w_train'], cp['w_test'], cp['model'], cp['predictions']

    # Sinon : Entraînement complet
    print("Début de la récupération des données et entraînement...")
    X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()
    
    bdt = model_class()
    bdt.fit(X_train, y_train, weights=w_train)
    predictions = bdt.predict(X_test)

    # Sauvegarde du contexte complet
    checkpoint = {
        'model': bdt,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'w_train': w_train, 'w_test': w_test,
        'predictions': predictions
    }
    joblib.dump(checkpoint, model_path)
    print(f"Checkpoint complet sauvegardé sous : {model_path}")
    
    return X_train, X_test, y_train, y_test, w_train, w_test, bdt, predictions


if __name__ == "__main__":
    training_tree()