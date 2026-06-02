
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import black_swan_pkg_main.HiggsML.datasets as datasets
from tabulate import tabulate
import numpy as np
import utils
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier

    """

    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        self.model.add(Dense(127, input_dim=n_dim, activation="swich"))
        self.model.add(Dense(127, activation="swish"))
        self.model.add(Dense(127, activation="swish"))
        self.model.add(Dense(127, activation="swish"))
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.scaler = StandardScaler()

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)
        history = self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=100, verbose=2
        )
        self.history = history  # permet de conserver des données pour le tracé des courbes d'apprentissage

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()


    def save(self,model_str ="model.keras",scaler_str ="scaler.pkl"):
        joblib.dump(self.scaler, scaler_str)
        self.model.save(model_str)

    def load(self,model_str ="model.keras",scaler_str ="scaler.pkl"):
        self.model = load_model(model_str)
        self.scaler = joblib.load(scaler_str)
    
    def plot_learning_curves(self):
        if not hasattr(self, "history"):
            raise ValueError("Le modèle doit être entraîné avant de tracer les courbes.")   #permet de tracer les courbes uniquement pour un modèle entrainé
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["loss"], label="Loss (train)")
        plt.plot(self.history.history["val_loss"], label="Loss (val)")
        plt.plot(self.history.history["accuracy"], label="Accuracy (train)")
        plt.plot(self.history.history["val_accuracy"], label="Accuracy (val)")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title("Courbes d'apprentissage du réseau de neurones")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_score_distribution(self, X_test, y_test):
        preds = self.model.predict(self.scaler.transform(X_test)).flatten()
        plt.figure(figsize=(8, 5))
        plt.hist(preds[y_test == 1], bins=50, alpha=0.6, label="Signal")
        plt.hist(preds[y_test == 0], bins=50, alpha=0.6, label="Bruit de fond")
        plt.xlabel("Score NN")
        plt.ylabel("Fréquence")
        plt.title("Distribution du score du réseau de neurones")
        plt.legend()
        plt.show()


def data_set():
    data =datasets.download_dataset("blackSwan_data") 
    data.load_train_set()
    data_set = data.get_train_set()

def visualize_data(data_set):
    

    target = data_set["labels"]
    weights = data_set["weights"]
    detailed_label = data_set["detailed_labels"]
    keys = np.unique(detailed_label)


    weight_keys = {}
    average_weights = {}
    for key in keys:
        weight_keys[key] = weights[detailed_label == key]

    table_data = []
    for key in keys:
        table_data.append(
            [
                key,
                np.sum(weight_keys[key]),
                len(weight_keys[key]),
                np.mean(weight_keys[key]),
            ]
        )

    table_data.append(
        [
            "Total Signal",
            np.sum(weights[target == 1]),
            len(weights[target == 1]),
            np.mean(weights[target == 1]),
        ]
    )
    table_data.append(
        [
            "Total Background",
            np.sum(weights[target == 0]),
            len(weights[target == 0]),
            np.mean(weights[target == 0]),
        ]
    )

    print("[*] --- Detailed Label Summary")
    print(
        tabulate(
            table_data,
            headers=[
                "Detailed Label",
                "Total Weight",
                "Number of events",
                "Average Weight",
            ],
            tablefmt="grid",
        )
    )
