from sklearn.model_selection import train_test_split
from HiggsML.datasets import download_dataset
import joblib

def get_clean_splits():
    """
    Télécharge et prépare les données HiggsML.
    Découpe en 80% Train (pour le fit) et 20% Test (pour le predict).
    
    Retourne:
        X_train, X_test, y_train, y_test, w_train, w_test
    """
    # 1. Chargement des données via la librairie officielle
    data = download_dataset("blackSwan_data")
    data.load_train_set()
    data_set = data.get_train_set()
    # 2. Extraction brute des colonnes cibles et nettoyage de X
    y = data_set["labels"]
    w = data_set["weights"]
    X = data_set.drop(columns=["labels", "weights", "detailed_labels"], errors="ignore")

    # 3. Découpage macro (80% Train / 20% Test local)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.20, random_state=42, stratify=y
    )
    # On renvoie les 6 blocs de données d'un coup
    return X_train, X_test, y_train, y_test, w_train, w_test