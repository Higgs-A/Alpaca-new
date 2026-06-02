
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import black_swan_pkg_main.HiggsML.datasets as datasets
import utils

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
        self.model.fit(
            X_train, y_train, sample_weight=weights_train, epochs=5, verbose=2
        )

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict(test_data).flatten().ravel()


    def save(self,model_str ="model.keras",scaler_str ="scaler.pkl"):
        joblib.dump(self.scaler, scaler_str)
        self.model.save(model_str)

    def load(self,model_str ="model.keras",scaler_str ="scaler.pkl"):
        self.model = load_model(model_str)
        self.scaler = joblib.load(scaler_str)


def test_NN():
    data =datasets.download_dataset("blackSwan_data") 
    data.load_train_set()
    data_set = data.get_train_set()