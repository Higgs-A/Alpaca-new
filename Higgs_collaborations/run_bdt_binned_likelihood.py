from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

from sample_code_submission.boosted_decision_tree import BoostedDecisionTree


def prepare_binned(n_bins, s_scores, s_weights, b_scores, b_weights, n_scores, n_weights):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_obs, _ = np.histogram(n_scores, bins=bin_edges, weights=n_weights)
    s, _ = np.histogram(s_scores, bins=bin_edges, weights=s_weights)
    b, _ = np.histogram(b_scores, bins=bin_edges, weights=b_weights)
    return n_obs, s, b


def nll(mu, n_obs, s, b):
    expected = np.clip(mu * s + b, 1e-12, None)
    return float(np.sum(expected - n_obs * np.log(expected)))


def estimate_mu_hat(n_obs, s, b, mu_min=0.0, mu_max=5.0, points=2001):
    mu_values = np.linspace(mu_min, mu_max, points)
    nll_values = np.array([nll(mu, n_obs, s, b) for mu in mu_values])
    best_idx = int(np.argmin(nll_values))
    return float(mu_values[best_idx]), mu_values, nll_values


def confidence_interval(mu_hat, mu_values, nll_values):
    delta_nll = nll_values - np.min(nll_values)
    mask = delta_nll < 20
    mu_values = mu_values[mask]
    delta_nll = delta_nll[mask]

    left_mask = mu_values < mu_hat
    right_mask = mu_values > mu_hat

    mu_minus = mu_hat
    mu_plus = mu_hat

    if np.sum(left_mask) > 2 and np.sum(right_mask) > 2:
        left_delta = delta_nll[left_mask]
        left_mu = mu_values[left_mask]
        right_delta = delta_nll[right_mask]
        right_mu = mu_values[right_mask]

        left_order = np.argsort(left_delta)
        right_order = np.argsort(right_delta)

        left_interp = interp1d(
            left_delta[left_order],
            left_mu[left_order],
            bounds_error=False,
            fill_value="extrapolate",
        )
        right_interp = interp1d(
            right_delta[right_order],
            right_mu[right_order],
            bounds_error=False,
            fill_value="extrapolate",
        )

        mu_minus = float(left_interp(0.5))
        mu_plus = float(right_interp(0.5))

    return mu_minus, mu_plus, mu_values, delta_nll


def plot_profile(mu_hat, mu_minus, mu_plus, mu_values, delta_nll, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(mu_values, delta_nll, linewidth=2, label=r"$\Delta$NLL$(\mu)$")
    plt.axvline(mu_hat, color="red", linestyle="--", label=rf"$\hat{{\mu}}={mu_hat:.3f}$")
    plt.axhline(0.5, color="black", linestyle=":", label=r"$\Delta$NLL = 0.5")
    plt.axvline(mu_minus, color="green", linestyle=":")
    plt.axvline(mu_plus, color="green", linestyle=":")
    plt.scatter([mu_minus, mu_plus], [0.5, 0.5], color="green", zorder=5)
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$NLL")
    plt.title("Binned Profile Likelihood (BoostedDecisionTree)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train BDT and estimate mu with binned likelihood.")
    parser.add_argument("--data", type=str, default="blackSwan_data/blackSwan_data.parquet")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300000,
        help="Maximum rows to train on. Use -1 to use the full dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="bdt_binned_profile_likelihood.png")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = root / args.data
    output_path = root / args.output

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)

    if args.max_rows is not None and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed)

    labels = df["labels"].to_numpy(dtype=int)
    weights = df["weights"].to_numpy(dtype=float)

    feature_cols = [
        c
        for c in df.columns
        if c not in {"labels", "weights", "detailed_labels"}
    ]
    x = df[feature_cols].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x,
        labels,
        weights,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    model = BoostedDecisionTree()
    model.fit(x_train, y_train, weights=w_train)
    score_test = model.predict(x_test)

    signal_mask = y_test == 1
    background_mask = y_test == 0

    n_obs, s, b = prepare_binned(
        args.n_bins,
        score_test[signal_mask],
        w_test[signal_mask],
        score_test[background_mask],
        w_test[background_mask],
        score_test,
        w_test,
    )

    mu_hat, mu_values, nll_values = estimate_mu_hat(n_obs, s, b)
    mu_minus, mu_plus, mu_plot, delta_nll = confidence_interval(mu_hat, mu_values, nll_values)
    plot_profile(mu_hat, mu_minus, mu_plus, mu_plot, delta_nll, output_path)

    print(f"rows_used={len(df)}")
    print(f"mu_hat={mu_hat:.4f}")
    print(f"mu_minus={mu_minus:.4f}")
    print(f"mu_plus={mu_plus:.4f}")
    print(f"plot_saved={output_path}")


if __name__ == "__main__":
    main()
