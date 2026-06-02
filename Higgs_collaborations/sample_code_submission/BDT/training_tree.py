
from get_data import get_clean_splits
from boosted_decision_tree import BoostedDecisionTree

X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()

bdt = BoostedDecisionTree()


def training_tree():
    """
    Entraîne le modèle BDT et retourne toutes les données 
    nécessaires pour l'évaluation et la création de graphiques.
    """

    print("[*] Début de l'entraînement de l'arbre...")
    bdt.fit(X_train, y_train, weights=w_train)
    print("[*] Modèle entraîné avec succès !")

    predictions = bdt.predict(X_test)
    
    return bdt, predictions, X_train, y_test, w_test,