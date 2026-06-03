
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
        model_name = model_class.__name__.lower() # ex: 'xgboost_bdt'
        model_path = f"{model_name}_model.joblib"
    
    # 1. Récupération des données
    X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()
    
    # 2. Logique de chargement ou d'entraînement
    if os.path.exists(model_path):
        reponse = input(f"Un modèle de type {model_class.__name__} existe ({model_path}). Voulez-vous l'utiliser ? (o/n) : ").lower()
        if reponse == 'o':
            print(f"Chargement de {model_path}...")
            bdt = joblib.load(model_path)
        else:
            bdt = train_new_model(model_class, X_train, y_train, w_train, model_path)
    else:
        print(f"Aucun modèle trouvé pour {model_class.__name__}.")
        bdt = train_new_model(model_class, X_train, y_train, w_train, model_path)

    # 3. Évaluation
    predictions = bdt.predict(X_test)
    return bdt, predictions, y_test, w_test

def train_new_model(model_class, X_train, y_train, w_train, model_path):
    # Ici, on instancie dynamiquement la classe passée en argument
    bdt = model_class() 
    print(f"[*] Début de l'entraînement avec {model_class.__name__}...")
    bdt.fit(X_train, y_train, weights=w_train)
    joblib.dump(bdt, model_path)
    print(f"Modèle sauvegardé sous : {model_path}")
    return bdt

if __name__ == "__main__":
    training_tree()