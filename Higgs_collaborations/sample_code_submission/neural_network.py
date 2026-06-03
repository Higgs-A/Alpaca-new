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
    """
    Neural network classifier for Higgs (H -> tau tau) signal detection.

    Implementation grounded in HEP literature:
    - Rectangular architecture (Baldi et al. PRL 2015, Melis 2015)
    - Dropout 0.5 on hidden layers only (Melis 2015)
    - BatchNormalization between Dense and ReLU (modern HEP practice)
    - Adam with learning rate 5e-4 + ReduceLROnPlateau
      (inspired by Melis 2015 LR decay 0.97/epoch)
    - AUC monitored instead of accuracy (avoid 99.85% background trap)
    - Early stopping with restore_best_weights
    - Model save/load for reuse
    - AMS computation (HiggsML Challenge 2014 metric)
    """

    def __init__(self, train_data=None):
        self.scaler = StandardScaler()
        self._test_predictions = None
        self._test_labels = None
        self._test_weights = None

        if train_data is not None:
            self._build_model(train_data.shape[1])

    def _build_model(self, n_dim):
        self.model = Sequential()

        # Première couche cachée — pas de dropout sur les inputs (Melis 2015)
        self.model.add(Dense(256, input_dim=n_dim))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))

        # 3 couches cachées intermédiaires : Dense -> BN -> ReLU -> Dropout
        for _ in range(3):
            self.model.add(Dense(256))
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(Dropout(0.5))

        # Couche de sortie
        self.model.add(Dense(1, activation="sigmoid"))

        # Axe 5 : learning rate adapté + Axe 6 : AUC au lieu d'accuracy
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=0.0005),
            metrics=[AUC(name="auc")]
        )

    def fit(self, train_data, y_train, weights_train=None):
        self.scaler.fit(train_data)
        X_train = self.scaler.transform(train_data)

        # Early stopping : surveille val_auc (à maximiser)
        early_stop = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=20,
            restore_best_weights=True
        )

        # LR scheduling : divise le LR par 2 quand val_auc plateau
        # (inspiré du LR decay 0.97/epoch de Melis 2015)
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

        # Stocke pour calcul AMS ultérieur (optionnel)
        self._test_predictions = predictions
        if labels is not None:
            self._test_labels = np.asarray(labels)
        if weights is not None:
            self._test_weights = np.asarray(weights)

        return predictions

    def significance(self, labels=None, weights=None):
        """
        Calcule l'Approximate Median Significance (AMS).
        Métrique de référence du challenge HiggsML 2014.
        Record du challenge 2014 : ~3.80.
        """
        if labels is not None:
            self._test_labels = np.asarray(labels)
        if weights is not None:
            self._test_weights = np.asarray(weights)

        if self._test_predictions is None or self._test_labels is None:
            raise ValueError(
                "Appelle d'abord predict(data, labels=..., weights=...)"
            )

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

        # Cumul depuis la droite (événements avec score > seuil)
        s_cumul = np.cumsum(s_hist[::-1])[::-1]
        b_cumul = np.cumsum(b_hist[::-1])[::-1]

        ams_values = ams_formula(s_cumul, b_cumul)
        max_ams = float(np.nanmax(ams_values))
        best_thr = float(bins[:-1][np.nanargmax(ams_values)])

        return {"max_ams": max_ams, "best_threshold": best_thr}

    def save(self, path):
        """Sauvegarde le modèle entraîné et le scaler (Axe 8)."""
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        print(f"Model saved to {path}")

    def load(self, path):
        """Recharge un modèle sauvegardé (Axe 8)."""
        self.model = keras_load_model(os.path.join(path, "model.keras"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        print(f"Model loaded from {path}")