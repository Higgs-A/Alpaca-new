from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

from sample_code_submission.boosted_decision_tree import BoostedDecisionTree


def prepare_binned(n_bins, s_scores, s_weights, b_scores, b_weights, n_scores, n_weights):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_obs, _ = np.histogram(n_scores, bins=bin_edges, weights=n_weights)
    signal, _ = np.histogram(s_scores, bins=bin_edges, weights=s_weights)
    background, _ = np.histogram(b_scores, bins=bin_edges, weights=b_weights)
    return n_obs, signal, background


def binned_nll(mu, n_obs, signal, background):
    expected = np.clip(mu * signal + background, 1e-12, None)
    return float(np.sum(expected - n_obs * np.log(expected)))


def prepare_unbinned(s_scores, s_weights, b_scores, b_weights):
    # Stabilize KDE when BDT scores are very concentrated near 0 or 1.
    eps = 1e-6
    rng = np.random.default_rng(42)

    s_scores = np.clip(s_scores, eps, 1.0 - eps)
    b_scores = np.clip(b_scores, eps, 1.0 - eps)

    s_scores = s_scores + rng.normal(0.0, 1e-6, size=s_scores.shape)
    b_scores = b_scores + rng.normal(0.0, 1e-6, size=b_scores.shape)

    pdf_signal = gaussian_kde(s_scores, weights=s_weights, bw_method="scott")
    pdf_background = gaussian_kde(b_scores, weights=b_weights, bw_method="scott")
    n_signal = float(np.sum(s_weights))
    n_background = float(np.sum(b_weights))
    return pdf_signal, pdf_background, n_signal, n_background


def unbinned_nll(mu, n_scores, n_weights, pdf_signal, pdf_background, n_signal, n_background):
    if mu < 0:
        return 1e10

    n_scores = np.clip(np.asarray(n_scores, dtype=float), 1e-6, 1.0 - 1e-6)
    n_weights = np.asarray(n_weights, dtype=float)

    expected_total = mu * n_signal + n_background
    f_signal = pdf_signal(n_scores)
    f_background = pdf_background(n_scores)
    event_likelihood = (mu * n_signal * f_signal + n_background * f_background) / expected_total
    event_likelihood = np.clip(event_likelihood, 1e-12, None)
    nll_val = float(expected_total - np.sum(n_weights * np.log(event_likelihood)))
    if not np.isfinite(nll_val):
        return 1e10
    return nll_val


def profile_from_nll(nll_callable, mu_min=0.0, mu_max=5.0, points=1000):
    mu_values = np.linspace(mu_min, mu_max, points)
    nll_values = np.array([nll_callable(mu) for mu in mu_values])
    if not np.all(np.isfinite(nll_values)):
        finite_max = np.nanmax(nll_values[np.isfinite(nll_values)]) if np.any(np.isfinite(nll_values)) else 1e12
        nll_values = np.nan_to_num(nll_values, nan=finite_max + 1e6, posinf=finite_max + 1e6, neginf=finite_max + 1e6)
    nll_min = float(np.min(nll_values))
    delta_nll = nll_values - nll_min
    mu_hat = float(mu_values[int(np.argmin(nll_values))])
    return mu_hat, mu_values, delta_nll


def one_sigma_interval(mu_hat, mu_values, delta_nll):
    mask = delta_nll < 20
    mu_values = mu_values[mask]
    delta_nll = delta_nll[mask]

    left_mask = mu_values < mu_hat
    right_mask = mu_values > mu_hat

    mu_minus = mu_hat
    mu_plus = mu_hat

    # Handle each side independently so boundary minima still get one-sided errors.
    if np.sum(left_mask) > 2:
        left_delta = delta_nll[left_mask]
        left_mu = mu_values[left_mask]
        left_order = np.argsort(left_delta)
        left_delta_sorted = left_delta[left_order]
        left_mu_sorted = left_mu[left_order]
        left_delta_unique, left_unique_idx = np.unique(left_delta_sorted, return_index=True)
        if left_delta_unique.size > 1:
            left_interp = interp1d(
                left_delta_unique,
                left_mu_sorted[left_unique_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )
            mu_minus = float(left_interp(0.5))

    if np.sum(right_mask) > 2:
        right_delta = delta_nll[right_mask]
        right_mu = mu_values[right_mask]
        right_order = np.argsort(right_delta)
        right_delta_sorted = right_delta[right_order]
        right_mu_sorted = right_mu[right_order]
        right_delta_unique, right_unique_idx = np.unique(right_delta_sorted, return_index=True)
        if right_delta_unique.size > 1:
            right_interp = interp1d(
                right_delta_unique,
                right_mu_sorted[right_unique_idx],
                bounds_error=False,
                fill_value="extrapolate",
            )
            mu_plus = float(right_interp(0.5))

    return mu_minus, mu_plus, mu_values, delta_nll


def main():
    parser = argparse.ArgumentParser(
        description="Train the BDT and compare profile likelihood curves for 1, 5, 10, 20 bins and unbinned."
    )
    parser.add_argument("--data", type=str, default="blackSwan_data/blackSwan_data.parquet")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu-max", type=float, default=5.0)
    parser.add_argument("--mu-points", type=int, default=1000)
    parser.add_argument("--output", type=str, default="bdt_likelihood_comparison.png")
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
    feature_cols = [column for column in df.columns if column not in {"labels", "weights", "detailed_labels"}]
    data = df[feature_cols].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        data,
        labels,
        weights,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    model = BoostedDecisionTree()
    model.fit(x_train, y_train, weights=w_train)
    scores = model.predict(x_test)

    signal_mask = y_test == 1
    background_mask = y_test == 0

    s_scores = scores[signal_mask]
    s_weights = w_test[signal_mask]
    b_scores = scores[background_mask]
    b_weights = w_test[background_mask]
    n_scores = scores
    n_weights = w_test

    curves = []
    for n_bins in [1, 5, 10, 20]:
        n_obs, signal, background = prepare_binned(
            n_bins,
            s_scores,
            s_weights,
            b_scores,
            b_weights,
            n_scores,
            n_weights,
        )
        nll_callable = lambda mu, n_obs=n_obs, signal=signal, background=background: binned_nll(
            mu,
            n_obs,
            signal,
            background,
        )
        mu_hat, mu_values, delta_nll = profile_from_nll(
            nll_callable,
            mu_max=args.mu_max,
            points=args.mu_points,
        )
        mu_minus, mu_plus, mu_values, delta_nll = one_sigma_interval(mu_hat, mu_values, delta_nll)
        curves.append(
            {
                "label": f"{n_bins} bins",
                "mu_hat": mu_hat,
                "mu_minus": mu_minus,
                "mu_plus": mu_plus,
                "mu_values": mu_values,
                "delta_nll": delta_nll,
            }
        )

    pdf_signal, pdf_background, n_signal, n_background = prepare_unbinned(
        s_scores,
        s_weights,
        b_scores,
        b_weights,
    )
    unbinned_callable = lambda mu: unbinned_nll(
        mu,
        n_scores,
        n_weights,
        pdf_signal,
        pdf_background,
        n_signal,
        n_background,
    )
    mu_hat, mu_values, delta_nll = profile_from_nll(
        unbinned_callable,
        mu_max=args.mu_max,
        points=args.mu_points,
    )
    mu_minus, mu_plus, mu_values, delta_nll = one_sigma_interval(mu_hat, mu_values, delta_nll)
    curves.append(
        {
            "label": "unbinned",
            "mu_hat": mu_hat,
            "mu_minus": mu_minus,
            "mu_plus": mu_plus,
            "mu_values": mu_values,
            "delta_nll": delta_nll,
        }
    )

    if curves[-1]["mu_hat"] <= 1e-9:
        print(
            "warning: unbinned mu_hat is at lower boundary (mu=0). "
            "Try increasing --mu-max or check whether unbinned likelihood truly prefers no signal."
        )

    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#6a3d9a"]
    for curve, color in zip(curves, colors):
        plt.plot(
            curve["mu_values"],
            curve["delta_nll"],
            linewidth=2,
            color=color,
            label=(
                f"{curve['label']} "
                f"($\\hat{{\\mu}}={curve['mu_hat']:.3f}$, "
                f"[$ {curve['mu_minus']:.3f}, {curve['mu_plus']:.3f} $])"
            ),
        )

    plt.axhline(0.5, color="black", linestyle=":", linewidth=1.5, label=r"$\Delta$NLL = 0.5")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$NLL")
    plt.title("BDT Profile Likelihood Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"rows_used={len(df)}")
    for curve in curves:
        print(
            f"{curve['label']}: "
            f"mu_hat={curve['mu_hat']:.4f}, "
            f"mu_minus={curve['mu_minus']:.4f}, "
            f"mu_plus={curve['mu_plus']:.4f}"
        )
    print(f"plot_saved={output_path}")


if __name__ == "__main__":
    main()