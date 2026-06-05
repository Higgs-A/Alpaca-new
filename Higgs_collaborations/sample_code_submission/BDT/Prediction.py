from boosted_decision_tree import XGBoost_BDT
from training_tree import training_tree
import pandas as pd
from get_data import get_clean_splits


def predict_with_threshold(input_data=None, output_data=None, output_filename="predictions.csv"):
    """
    Récupère le modèle et les données, prédit les classes, 
    et sauvegarde le résultat en CSV.
    """
    # On récupère tout en un seul appel pour éviter les double-inputs
    _, X_test, _, y_test, _, _, bdt, _ = training_tree(model_class=XGBoost_BDT)
    
    # Utilisation des données par défaut si rien n'est fourni
    data_to_predict = input_data if input_data is not None else X_test
    labels= output_data if input_data is not None else y_test
    
    # Prédiction des probabilités
    probas = bdt.predict(data_to_predict)
    
    # Création du DataFrame
    df = pd.DataFrame({
        'y_true': labels,
        'proba': probas,
    })
    
    # Sauvegarde en CSV
    df.to_csv(output_filename, index=False)
    print(f"[*] Prédictions sauvegardées dans {output_filename}")
    
    return df

predict_with_threshold()