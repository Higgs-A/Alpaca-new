import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from iminuit import Minuit

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from sample_code_submission.boosted_decision_tree import XGBoost_BDT
from statistical_analysis import prepare_binned, NLL, plot_binned_profile_likelihood

DEFAULT_DATA_DIR = Path.home() / "Downloads" / "blackSwan_data" # à modifier selon l'emplacement de vos données

def find_data_file(data_dir):
    data_dir = Path(data_dir)
    if data_dir.is_file():
        return data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for pattern in ["*.npz", "*.csv", "*.parquet"]:
        files = sorted(data_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No supported data file found in {data_dir}")

def load_black_swan_dataset(path):
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "X" in data and "y" in data:
            X = np.array(data["X"], dtype=np.float32)
            y = np.array(data["y"], dtype=np.int32)
            weights = np.array(data.get("w", data.get("weights", np.ones(len(y)))), dtype=np.float32)
            return X, y, weights
        if "data" in data and "labels" in data:
            X = np.array(data["data"], dtype=np.float32)
            y = np.array(data["labels"], dtype=np.int32)
            weights = np.array(data.get("weights", np.ones(len(y))), dtype=np.float32)
            return X, y, weights
        raise ValueError("npz file must contain X/y or data/labels")
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        label_col = next((c for c in ["label", "target", "y", "truth"] if c in df.columns), None)
        if label_col is None:
            raise ValueError("CSV needs a label column named label/target/y/truth")
        weight_col = next((c for c in ["weight", "weights", "w"] if c in df.columns), None)
        y = df[label_col].to_numpy(dtype=np.int32)
        weights = df[weight_col].to_numpy(dtype=np.float32) if weight_col else np.ones(len(df), dtype=np.float32)
        X = df.drop(columns=[label_col] + ([weight_col] if weight_col else [])).to_numpy(dtype=np.float32)
        return X, y, weights
    raise ValueError(f"Unsupported data file type: {path.suffix}")

def fit_mu_binned(N_obs, S, B, mu0=1.0):
    N_obs = np.asarray(N_obs, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    nll_fn = lambda mu: NLL(mu, N_obs, S, B)
    m = Minuit(nll_fn, mu=mu0)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()
    return float(m.values["mu"]), float(m.errors["mu"]), float(m.fval)

def main(data_dir=DEFAULT_DATA_DIR, n_bins=5, test_size=0.3, random_state=42):
    source_file = find_data_file(data_dir)
    print(f"Loading data from: {source_file}")
    X, y, weights = load_black_swan_dataset(source_file)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Training BDT model...")
    model = XGBoost_BDT( )
    model.fit(X_train, y_train, weight=w_train)
    print("Generating predictions...")
    scores = model.predict(X_test)
    S_scores = scores[y_test == 1]
    S_weights = w_test[y_test == 1]
    B_scores = scores[y_test == 0]
    B_weights = w_test[y_test == 0]

    print(f"Preparing binned data with {n_bins} bins...")
    N_obs, S, B = prepare_binned(n_bins, S_scores, S_weights, B_scores, B_weights, scores, w_test)
    
    print(f"N_obs: {N_obs}")
    print(f"S: {S}")
    print(f"B: {B}")
    
    print("Fitting mu...")
    mu_hat, mu_err, nll_min = fit_mu_binned(N_obs, S, B)

    print(f"Data file: {source_file}")
    print(f"mu_hat: {mu_hat}")
    print(f"mu_err: {mu_err}")
    print(f"NLL_min: {nll_min}")

    print("Plotting profile likelihood...")
    plot_binned_profile_likelihood(N_obs, S, B, mu_hat, NLL, plot_show=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BDT and plot binned NLL scan.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()
    main(data_dir=args.data_dir, n_bins=args.n_bins, test_size=args.test_size)