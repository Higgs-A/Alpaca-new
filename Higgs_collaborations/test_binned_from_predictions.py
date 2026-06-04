from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


def register_stub(module_name: str, *, package: bool = False, attrs: dict | None = None) -> None:
    module = types.ModuleType(module_name)
    if package:
        module.__path__ = []
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules[module_name] = module


register_stub("HiggsML", package=True)
register_stub("HiggsML.systematics", attrs={"systematics": object()})
sys.modules["HiggsML"].systematics = sys.modules["HiggsML.systematics"]

register_stub("iminuit", attrs={"Minuit": type("Minuit", (), {})})
register_stub("systematic_analysis", package=True)
register_stub("systematic_analysis.py")
sys.modules["systematic_analysis"].py = sys.modules["systematic_analysis.py"]

module_path = Path(__file__).resolve().parent / "sample_code_submission" / "statistical_analysis.py"
spec = importlib.util.spec_from_file_location("statistical_analysis_under_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

pred_path = Path(__file__).resolve().parent / "predictions(in).csv"
df = pd.read_csv(pred_path)

if "y_true" not in df.columns or "proba" not in df.columns:
    raise ValueError("Le fichier predictions(in).csv doit contenir les colonnes 'y_true' et 'proba'.")

labels = df["y_true"].to_numpy(dtype=int)
scores = np.clip(df["proba"].to_numpy(dtype=float), 0.0, 1.0)

# Si les poids ne sont pas disponibles dans ce CSV, on prend des poids unitaires.
weights = np.ones_like(scores, dtype=float)

signal_mask = labels == 1
background_mask = labels == 0

N_bins = 10
N_obs, S, B = module.prepare_binned(
    N_bins,
    scores[signal_mask],
    weights[signal_mask],
    scores[background_mask],
    weights[background_mask],
    scores,
    weights,
)


def nll_from_predictions(mu, N, S, B):
    expected = np.clip(mu * S + B, 1e-12, None)
    return float(np.sum(expected - N * np.log(expected)))


mu_grid = np.linspace(0.0, 5.0, 301)
nll_values = np.array([nll_from_predictions(mu, N_obs, S, B) for mu in mu_grid])
mu_hat = float(mu_grid[np.argmin(nll_values)])

output_path = Path(__file__).resolve().parent / "binned_profile_from_predictions.png"
mu_minus, mu_plus = module.plot_binned_profile_likelihood(
    N_obs,
    S,
    B,
    mu_hat,
    nll_from_predictions,
    save_path=str(output_path),
    plot_show=False,
)

print(f"rows={len(df)}")
print(f"mu_hat={mu_hat:.3f}")
print(f"mu_minus={mu_minus:.3f}")
print(f"mu_plus={mu_plus:.3f}")
print(f"saved={output_path}")