import os
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:

    def __init__(self, train_data=None):
        self.scaler = StandardScaler()
        self._test_predictions = None
        self._test_labels = None
        self._test_weights = None

        if train_data is not None:
            self._build_model(train_data.shape[1])

    def _build_model(self, n_dim):
        self.model = Sequential()

        self.model.add(Dense(256, input_dim=n_dim))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        for _ in range(3):
            self.model.add(Dense(256))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(Dropout(0.5))

        self.model.add(Dense(1, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.0005),
            metrics=[AUC(name="auc")]
        )

    def fit(self, train_data, y_train, weights_train=None):
        self.scaler.fit(train_data)
        X_train = self.scaler.transform(train_data)

        early_stop = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=20,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-6
        )

        self.model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            epochs=200,
            batch_size=256,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )

    def predict(self, test_data, labels=None, weights=None):
        X_test = self.scaler.transform(test_data)
        predictions = self.model.predict(X_test, verbose=0).flatten()

        self._test_predictions = predictions
        if labels is not None:
            self._test_labels = np.asarray(labels)
        if weights is not None:
            self._test_weights = np.asarray(weights)

        return predictions

    def significance(self, labels=None, weights=None):
        if labels is not None:
            self._test_labels = np.asarray(labels)
        if weights is not None:
            self._test_weights = np.asarray(weights)

        if self._test_predictions is None or self._test_labels is None:
            raise ValueError("Need to call predict with labels and weights first.")

        if self._test_weights is None:
            self._test_weights = np.ones(len(self._test_labels))

        def ams_formula(s, b):
            with np.errstate(invalid="ignore", divide="ignore"):
                b_safe = np.where(b <= 0, 1.0, b)
                val = np.sqrt(2 * ((s + b_safe) * np.log(1 + s / b_safe) - s))
                val = np.where((s < 0) | (b < 0), np.nan, val)
                val = np.where(b <= 0, 0.0, val)
            return val

        bins = np.linspace(0, 1, 101)
        sig_mask = self._test_labels == 1
        bkg_mask = self._test_labels == 0

        s_hist, _ = np.histogram(
            self._test_predictions[sig_mask],
            bins=bins,
            weights=self._test_weights[sig_mask]
        )
        b_hist, _ = np.histogram(
            self._test_predictions[bkg_mask],
            bins=bins,
            weights=self._test_weights[bkg_mask]
        )

        s_cumul = np.cumsum(s_hist[::-1])[::-1]
        b_cumul = np.cumsum(b_hist[::-1])[::-1]

        ams_values = ams_formula(s_cumul, b_cumul)
        max_ams = float(np.nanmax(ams_values))
        best_thr = float(bins[:-1][np.nanargmax(ams_values)])

        return {"max_ams": max_ams, "best_threshold": best_thr}

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        print(f"Model saved to {path}")

    def load(self, path):
        self.model = keras_load_model(os.path.join(path, "model.keras"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        print(f"Model loaded from {path}")