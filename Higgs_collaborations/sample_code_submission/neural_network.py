from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self, train_data):
        self.model = Sequential()

        n_dim = train_data.shape[1]

        # Architecture rectangulaire : 4 couches cachées de 256 neurones
        # Réf : Melis (2015), Baldi et al. (PRL 2015) - les architectures rectangulaires
        # sont la norme en HEP pour la classification d'événements

        # Première couche cachée — pas de Dropout en entrée (Melis 2015)
        self.model.add(Dense(256, input_dim=n_dim))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        # Couches cachées intermédiaires : Dense → BN → ReLU → Dropout
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.4))

        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.4))

        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.4))

        # Couche de sortie
        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.scaler = StandardScaler()
        

    def fit(self, train_data, y_train, weights_train=None):

        self.scaler.fit_transform(train_data)
        X_train = self.scaler.transform(train_data)

        # Early stopping pour éviter l'overfitting
        # Patience augmentée à 20 vu le dropout et batch norm
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        self.model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            epochs=200,
            batch_size=256,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=2
        )