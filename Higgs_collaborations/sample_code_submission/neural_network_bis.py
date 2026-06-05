import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.run_functions_eagerly(False)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:

    def __init__(self, n_features=None):
        self.model  = None
        self.scaler = StandardScaler()

        self._predictions  = None
        self._test_labels  = None
        self._test_weights = None

        if n_features is not None:
            self._initialize_model(n_features)

    def _initialize_model(self, n_features):
        """Initialize the model architecture."""
        self.model = Sequential([
            Dense(256, input_dim=n_features, activation="relu"),
            BatchNormalization(), Dropout(0.3),

            Dense(256, activation="relu"),
            BatchNormalization(), Dropout(0.3),

            Dense(128, activation="relu"),
            BatchNormalization(), Dropout(0.3),

            Dense(128, activation="relu"),
            BatchNormalization(), Dropout(0.3),

            Dense(64, activation="relu"),
            BatchNormalization(), Dropout(0.3),

            Dense(1, activation="sigmoid"),
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[AUC(name="auc")],
        )

    def fit(self, train_data, y_train, weights_train=None):
        """
        By design, this method loads a pretrained model from disk
        rather than training from scratch.
        The training notebook (Melkior_NN.ipynb) handles actual training
        and saves model.keras + scaler.pkl alongside this file.
        Those files are committed to the repo so the submission is self-contained.
        """
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_model(model_dir)

    def predict(self, test_data, labels=None, weights=None):
        self._predictions = self.model.predict(
            self.scaler.transform(test_data), verbose=0
        ).ravel()

        if labels  is not None: self._test_labels  = np.asarray(labels)
        if weights is not None: self._test_weights = np.asarray(weights)

        return self._predictions

    def significance(self, test_labels=None, test_weights=None):
        if test_labels  is not None: self._test_labels  = np.asarray(test_labels)
        if test_weights is not None: self._test_weights = np.asarray(test_weights)

        if self._predictions is None:
            raise ValueError("No predictions found. Call predict() first.")
        if self._test_labels is None:
            raise ValueError(
                "True labels not available. Provide them when calling predict()."
            )

        B_REG = 10.0

        def __ams(s, b):
            s = np.asarray(s, float)
            b = np.asarray(b, float)
            val = np.sqrt(2 * ((s + b + B_REG) * np.log(1 + s / (b + B_REG)) - s))
            val = np.where(s < 0, np.nan, val)
            return val

        def __significance_vscore(y_true, y_score, sample_weight):
            sample_weight = np.asarray(sample_weight)
            bins = np.linspace(0, 1.0, 101)
            s_hist, _ = np.histogram(
                y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
            )
            b_hist, _ = np.histogram(
                y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
            )
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]
            return __ams(s_cumul, b_cumul)

        ams_curve = __significance_vscore(
            y_true=self._test_labels,
            y_score=self._predictions,
            sample_weight=self._test_weights,
        )

        plt.plot(np.linspace(0, 1.0, 100), ams_curve, label="AMS Significance")
        plt.xlabel("Score threshold")
        plt.ylabel("AMS")
        return float(np.nanmax(ams_curve))

    def plot_learning_curves(self, weighted_test_auc=None, unweighted_test_auc=None):
        if not hasattr(self, "history"):
            raise ValueError("Model must be trained before plotting learning curves.")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        l2, = ax2.plot(self.history.history["loss"],    color="tab:orange", label="Loss (train)")
        lines = [l2]
        if "val_auc" in self.history.history:
            l3, = ax1.plot(self.history.history["val_auc"], color="tab:blue", linestyle="--",
                           label="AUC (internal val, unweighted)")
            lines.append(l3)
        if "val_loss" in self.history.history:
            l4, = ax2.plot(self.history.history["val_loss"], color="tab:orange", linestyle="--",
                           label="Loss (internal val)")
            lines.append(l4)
        if hasattr(self, "auc_train_unweighted"):
            l1 = ax1.axhline(self.auc_train_unweighted, color="tab:blue", linestyle="-", linewidth=1.2,
                             label=f"AUC (train, unweighted) = {self.auc_train_unweighted:.4f}")
            lines.append(l1)
        if unweighted_test_auc is not None:
            l5 = ax1.axhline(unweighted_test_auc, color="tab:green", linestyle=":", linewidth=1.5,
                             label=f"AUC (test, unweighted) = {unweighted_test_auc:.4f}")
            lines.append(l5)
        if weighted_test_auc is not None:
            l6 = ax1.axhline(weighted_test_auc, color="tab:red", linestyle=":", linewidth=1.5,
                             label=f"AUC (test, weighted) = {weighted_test_auc:.4f}")
            lines.append(l6)

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("AUC",  color="tab:blue")
        ax2.set_ylabel("Loss", color="tab:orange")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.legend(lines, [l.get_label() for l in lines])
        ax1.grid(True)
        plt.title("Learning Curves — all AUC values unweighted for comparability")
        plt.tight_layout()
        plt.show()

    def plot_score_distribution(self, X_test, y_test):
        y_pred = self.predict(X_test)
        signal_scores = y_pred[y_test == 1]
        bkg_scores    = y_pred[y_test == 0]

        plt.figure(figsize=(8, 6))
        plt.hist(signal_scores, bins=50, alpha=0.5, label='Signal',     color='blue', density=True)
        plt.hist(bkg_scores,    bins=50, alpha=0.5, label='Background', color='red',  density=True)
        plt.title('Score Distribution (Signal vs Background)')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path='weighted_best_model'):
        """Save the trained model and scaler to the specified path."""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        print(f"Model saved to {path}")

    def load_model(self, path='weighted_best_model'):
        """Load the trained model and scaler from the specified path."""
        self.model  = load_model(os.path.join(path, "model.keras"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        print(f"Model loaded from {path}")
