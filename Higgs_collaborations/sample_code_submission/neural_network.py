from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



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

    def __init__(self, n_features=None, train_data = None):
        self.model  = None
        self.scaler = StandardScaler()

        self._predictions  = None
        self._test_labels  = None
        self._test_weights = None
        if n_features is None and train_data is not None :
            n_features = train_data.shape[1]
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
        """Train the model."""
        #model_dir = os.path.dirname(os.path.abspath(__file__))
        #from pathlib import Path
        #model_dir = Path("weighted_best_model")
        '''
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weighted_best_model")
        print(model_dir)
        self.model = load_model(os.path.join(model_dir, "model.keras"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))'''
        import torch
        import joblib
        
        model_dir = "/content/drive/MyDrive/higgs_nn"

        model_path = os.path.join(model_dir, "model.pt")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        print("Model loaded from Drive")

        return self

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
                "True labels for test data are not available. Please provide them when calling predict()."
            )

        def __amsasimov(s_in, b_in):
            s = np.asarray(s_in, float)
            b = np.asarray(b_in, float)
            # Mask low-statistics region: b < 1 means the Poisson approximation
            # underlying AMS has broken down — results there are not meaningful.
            valid = (b >= 1.0) & (s >= 0)
            safe_b = np.where(valid, b, 1.0)
            safe_s = np.where(valid, s, 0.0)
            ams = np.sqrt(2 * ((safe_s + safe_b) * np.log(1 + safe_s / safe_b) - safe_s))
            ams = np.where(valid, ams, np.nan)
            if np.isscalar(s_in):
                return float(ams)
            else:
                return ams

        def __significance_vscore(y_true, y_score, sample_weight=None):
            if sample_weight is None:
                sample_weight = np.full(len(y_true), 1.0)
            else:
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
            significance = __amsasimov(s_cumul, b_cumul)
            return significance

        vamsasimov_xgb = __significance_vscore(
            y_true=self._test_labels,
            y_score=self._predictions,
            sample_weight=self._test_weights,
        )

        plt.plot(np.linspace(0, 1.0, 100), vamsasimov_xgb, label="AMS Significance")
        plt.xlabel("Score")
        plt.ylabel("Significance")
        return float(np.nanmax(vamsasimov_xgb))

    def plot_learning_curves(self, weighted_test_auc=None):
        if not hasattr(self, "history"):
            raise ValueError("Model must be trained before plotting learning curves.")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        l1, = ax1.plot(self.history.history["auc"],     color="tab:blue",   label="AUC (train)")
        # val_auc is computed on the internal validation split WITHOUT event weights
        # — it is optimistic relative to the weighted test AUC.
        l2, = ax2.plot(self.history.history["loss"],    color="tab:orange", label="Loss (train)")
        lines = [l1, l2]
        if "val_auc" in self.history.history:
            l3, = ax1.plot(self.history.history["val_auc"],  color="tab:blue",   linestyle="--", label="AUC (internal val, unweighted)")
            lines.append(l3)
        if "val_loss" in self.history.history:
            l4, = ax2.plot(self.history.history["val_loss"], color="tab:orange", linestyle="--", label="Loss (internal val, unweighted)")
            lines.append(l4)
        if weighted_test_auc is not None:
            l5 = ax1.axhline(weighted_test_auc, color="tab:green", linestyle=":", linewidth=1.5,
                             label=f"AUC (weighted test set) = {weighted_test_auc:.4f}")
            lines.append(l5)

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("AUC",  color="tab:blue")
        ax2.set_ylabel("Loss", color="tab:orange")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.legend(lines, [l.get_label() for l in lines])
        ax1.grid(True)
        plt.title("Learning Curves")
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

    def save_model(self, path = 'weighted_best_model'):
        """Save the trained model and scaler to the specified path."""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        print(f"Model saved to {path}")

    def load_model(self, path = 'weighted_best_model'):
        """Load the trained model and scaler from the specified path."""
        self.model  = load_model(os.path.join(path, "model.keras"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        print(f"Model loaded from {path}")


'''
NN = NeuralNetwork()
NN.load_modet()
predict = NN.predict(test_data)



'''