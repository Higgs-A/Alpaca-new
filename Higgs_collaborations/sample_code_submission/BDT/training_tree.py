
from get_data import get_clean_splits
from boosted_decision_tree import BoostedDecisionTree
import os
import joblib


def training_tree(model_path='bdt_model.joblib'):
    # On récupère les données quoi qu'il arrive
    X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()
    
    if os.path.exists(model_path):
        reponse = input(f"Un modèle existe déjà ({model_path}). Voulez-vous l'utiliser ? (o/n) : ").lower()
        if reponse == 'o':
            print("Chargement du modèle existant...")
            bdt = joblib.load(model_path)
        else:
            bdt = train_new_model(X_train, y_train, w_train, model_path)
    else:
        bdt = train_new_model(X_train, y_train, w_train, model_path)

    predictions = bdt.predict(X_test)
    return bdt, predictions, y_test, w_test

def train_new_model(X_train, y_train, w_train, model_path):
    bdt = BoostedDecisionTree()
    print("[*] Début de l'entraînement de l'arbre...")
    bdt.fit(X_train, y_train, weights=w_train)
    joblib.dump(bdt, model_path)
    print(f"Modèle sauvegardé sous : {model_path}")
    return bdt

if __name__ == "__main__":
    training_tree()