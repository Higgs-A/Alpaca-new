from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

from sample_code_submission.boosted_decision_tree import BoostedDecisionTree


def prepare_binned(n_bins, s_scores, s_weights, b_scores, b_weights, n_scores, n_weights):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_obs, _ = np.histogram(n_scores, bins=bin_edges, weights=n_weights)
    s, _ = np.histogram(s_scores, bins=bin_edges, weights=s_weights)
    b, _ = np.histogram(b_scores, bins=bin_edges, weights=b_weights)
    return n_obs, s, b


def binned_nll(mu, n_obs, s, b):
    expected = np.clip(mu * s + b, 1e-12, None)
    return float(np.sum(expected - n_obs * np.log(expected)))


def unbinned_nll(mu, data_scores, data_weights, pdf_s, pdf_b, n_s_exp, n_b_exp):
    if mu < 0:
        return 1e10

    n_expected_total = mu * n_s_exp + n_b_exp
    f_s = pdf_s(data_scores)
    f_b = pdf_b(data_scores)

    event_likelihood = (mu * n_s_exp * f_s + n_b_exp * f_b) / np.clip(n_expected_total, 1e-12, None)
    event_likelihood = np.clip(event_likelihood, 1e-12, None)

    return float(n_expected_total - np.sum(data_weights * np.log(event_likelihood)))


def profile_from_nll(mu_grid, nll_fn):
    nll_vals = np.array([nll_fn(mu) for mu in mu_grid])
    delta_nll = nll_vals - np.min(nll_vals)
    mu_hat = float(mu_grid[int(np.argmin(nll_vals))])
    return mu_hat, delta_nll


def main():
    parser = argparse.ArgumentParser(description="Compare BDT binned and unbinned profile likelihood curves.")
    parser.add_argument("--data", type=str, default="blackSwan_data/blackSwan_data.parquet")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=120000, help="Use -1 for full dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu-max", type=float, default=5.0)
    parser.add_argument("--mu-points", type=int, default=1001)
    parser.add_argument("--output", type=str, default="bdt_binned_unbinned_compare.png")
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

    feature_cols = [c for c in df.columns if c not in {"labels", "weights", "detailed_labels"}]
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

    s_scores = score_test[signal_mask]
    s_weights = w_test[signal_mask]
    b_scores = score_test[background_mask]
    b_weights = w_test[background_mask]
    n_scores = score_test
    n_weights = w_test

    mu_grid = np.linspace(0.0, args.mu_max, args.mu_points)

    curves = []
    for n_bins in [1, 5, 10, 20]:
        n_obs, s, b = prepare_binned(
            n_bins,
            s_scores,
            s_weights,
            b_scores,
            b_weights,
            n_scores,
            n_weights,
        )
        mu_hat, delta = profile_from_nll(mu_grid, lambda mu, n_obs=n_obs, s=s, b=b: binned_nll(mu, n_obs, s, b))
        curves.append((f"{n_bins} bins (mu_hat={mu_hat:.3f})", delta))

    pdf_s = gaussian_kde(s_scores, weights=s_weights)
    pdf_b = gaussian_kde(b_scores, weights=b_weights)
    n_s_exp = float(np.sum(s_weights))
    n_b_exp = float(np.sum(b_weights))

    mu_hat_unb, delta_unb = profile_from_nll(
        mu_grid,
        lambda mu: unbinned_nll(mu, n_scores, n_weights, pdf_s, pdf_b, n_s_exp, n_b_exp),
    )
    curves.append((f"unbinned (mu_hat={mu_hat_unb:.3f})", delta_unb))

    plt.figure(figsize=(10, 6))
    for label, delta in curves:
        plt.plot(mu_grid, delta, linewidth=2, label=label)

    plt.axhline(0.5, color="black", linestyle=":", linewidth=1.5, label=r"$\Delta$NLL = 0.5")
    plt.xlim(0.0, args.mu_max)
    plt.ylim(0.0, min(20.0, float(np.max([np.max(c[1]) for c in curves])) + 0.5))
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$NLL")
    plt.title("BDT Profile Likelihood: 1, 5, 10, 20 bins vs unbinned")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"rows_used={len(df)}")
    print(f"plot_saved={output_path}")


if __name__ == "__main__":
    main()
