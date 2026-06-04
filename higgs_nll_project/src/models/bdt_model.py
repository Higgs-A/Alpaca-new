from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd

class BDTModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, filename):
        import joblib
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        import joblib
        self.model = joblib.load(filename)