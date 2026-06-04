from get_data import get_clean_splits
from boosted_decision_tree import XGBoost_BDT as BoostedDecisionTree

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Choix du test — décommenter UNE seule ligne
# ==========================================================

TEST = "alpha_beta"
# TEST = "subsample_colsample"
# TEST = "depth"

# AMS

def amsasimov(s_in, b_in):

    s = np.copy(s_in)
    b = np.copy(b_in)

    s = np.where((b_in == 0), 0.0, s_in)
    b = np.where((b_in == 0), 1.0, b)

    ams = np.sqrt(
        2 * (
            (s + b) * np.log(1 + s / b)
            - s
        )
    )

    ams = np.where(
        (s < 0) | (b < 0),
        np.nan,
        ams
    )

    if np.isscalar(s_in):
        return float(ams)

    return ams


def significance_vscore(y_true, y_score, sample_weight):

    bins = np.linspace(0, 1, 101)

    s_hist, _ = np.histogram(
        y_score[y_true == 1],
        bins=bins,
        weights=sample_weight[y_true == 1]
    )

    b_hist, _ = np.histogram(
        y_score[y_true == 0],
        bins=bins,
        weights=sample_weight[y_true == 0]
    )

    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    return amsasimov(s_cumul, b_cumul)


def significance_score(y_true, y_score, sample_weight):

    return np.max(
        significance_vscore(y_true, y_score, sample_weight)
    )

# Évaluation d'un modèle

def evaluate_model(X_train, y_train, w_train,
                   X_test, y_test, w_test, **params):

    model = BoostedDecisionTree()
    model.model.set_params(**params)
    model.fit(X_train, y_train, weights=w_train)

    predictions = model.predict(X_test)

    return significance_score(
        y_true=y_test,
        y_score=predictions,
        sample_weight=w_test
    )


#  heatmap 2D

def plot_heatmap(grid, x_values, y_values, xlabel, ylabel, title):

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")

    plt.colorbar(im, ax=ax, label="AMS")

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{v:.3g}" for v in x_values], rotation=45, ha="right")

    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{v:.3g}" for v in y_values])

    for i in range(len(y_values)):
        for j in range(len(x_values)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if val < np.nanmax(grid) * 0.92 else "black"
                )

    idx = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle(
        (idx[1] - 0.5, idx[0] - 0.5), 1, 1,
        fill=False, edgecolor="red", linewidth=2.5
    ))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


# Optimisation profondeur (1D)

def optimize_depth(X_train, y_train, w_train,
                   X_test, y_test, w_test):

    depths = [7, 8, 9, 10, 11, 12]
    scores = []
    best_score = -np.inf
    best_depth = None

    print(f"\nTesting max_depth  ({len(depths)} évaluations)")

    for depth in depths:

        score = evaluate_model(
            X_train, y_train, w_train,
            X_test, y_test, w_test,
            max_depth=depth
        )

        scores.append(score)
        print(f"  max_depth={depth:>2}  AMS={score:.4f}")

        if score > best_score:
            best_score = score
            best_depth = depth

    plt.figure(figsize=(8, 5))
    plt.plot(depths, scores, marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("AMS")
    plt.title("AMS vs max_depth")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nBest → max_depth={best_depth}  AMS={best_score:.4f}")


# Optimisation croisée reg_alpha / reg_lambda

def optimize_alpha_beta(X_train, y_train, w_train,
                        X_test, y_test, w_test,
                        best_depth=10):

    alphas  = [0, 0.01, 0.05, 0.1, 1,]
    lambdas = [0.1, 0.5, 1, 5,]

    n_a = len(alphas)
    n_l = len(lambdas)

    grid  = np.full((n_l, n_a), np.nan)
    total = n_a * n_l
    done  = 0

    best_score  = -np.inf
    best_params = {}

    print(f"\nOptimisation croisée reg_alpha × reg_lambda  ({total} évaluations)")

    for i, lam in enumerate(lambdas):
        for j, alpha in enumerate(alphas):

            done += 1
            print(
                f"  [{done:>3}/{total}]  "
                f"reg_alpha={alpha:<6}  reg_lambda={lam:<6}",
                end="  "
            )

            score = evaluate_model(
                X_train, y_train, w_train,
                X_test, y_test, w_test,
                max_depth=best_depth,
                reg_alpha=alpha,
                reg_lambda=lam
            )

            grid[i, j] = score
            print(f"AMS={score:.4f}")

            if score > best_score:
                best_score  = score
                best_params = {"reg_alpha": alpha, "reg_lambda": lam}

    plot_heatmap(
        grid,
        x_values=alphas,
        y_values=lambdas,
        xlabel="reg_alpha (L1)",
        ylabel="reg_lambda (L2)",
        title=f"AMS — reg_alpha × reg_lambda  (max_depth={best_depth})"
    )

    print(f"\nBest → reg_alpha={best_params['reg_alpha']}  "
          f"reg_lambda={best_params['reg_lambda']}  "
          f"AMS={best_score:.4f}")


# Optimisation croisée subsample / colsample_bytree

def optimize_subsample_colsample(X_train, y_train, w_train,
                                 X_test, y_test, w_test,
                                 best_depth=10):

    subsamples = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    colsamples = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    n_s = len(subsamples)
    n_c = len(colsamples)

    grid  = np.full((n_c, n_s), np.nan)
    total = n_s * n_c
    done  = 0

    best_score  = -np.inf
    best_params = {}

    print(f"\nOptimisation croisée subsample × colsample_bytree  ({total} évaluations)")

    for i, col in enumerate(colsamples):
        for j, sub in enumerate(subsamples):

            done += 1
            print(
                f"  [{done:>3}/{total}]  "
                f"subsample={sub:<5}  colsample_bytree={col:<5}",
                end="  "
            )

            score = evaluate_model(
                X_train, y_train, w_train,
                X_test, y_test, w_test,
                max_depth=best_depth,
                subsample=sub,
                colsample_bytree=col
            )

            grid[i, j] = score
            print(f"AMS={score:.4f}")

            if score > best_score:
                best_score  = score
                best_params = {"subsample": sub, "colsample_bytree": col}

    plot_heatmap(
        grid,
        x_values=subsamples,
        y_values=colsamples,
        xlabel="subsample",
        ylabel="colsample_bytree",
        title=f"AMS — subsample × colsample_bytree  (max_depth={best_depth})"
    )

    print(f"\nBest → subsample={best_params['subsample']}  "
          f"colsample_bytree={best_params['colsample_bytree']}  "
          f"AMS={best_score:.4f}")


# Main

def hyperparameters_optimisation():

    print("Loading data...")

    X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()

    X_train, _, y_train, _, w_train, _ = train_test_split(
        X_train, y_train, w_train,
        train_size=0.50,
        random_state=42,
        stratify=y_train
    )

    print(f"Train size : {len(X_train):,}")
    print(f"Test  size : {len(X_test):,}")
    print(f"Test sélectionné : {TEST}\n")

    best_depth = 10

    if TEST == "depth":
        optimize_depth(
            X_train, y_train, w_train,
            X_test, y_test, w_test
        )

    elif TEST == "alpha_beta":
        optimize_alpha_beta(
            X_train, y_train, w_train,
            X_test, y_test, w_test,
            best_depth=best_depth
        )

    elif TEST == "subsample_colsample":
        optimize_subsample_colsample(
            X_train, y_train, w_train,
            X_test, y_test, w_test,
            best_depth=best_depth
        )

    else:
        raise ValueError(
            f"TEST inconnu : '{TEST}'. "
            "Valeurs autorisées : depth | alpha_beta | subsample_colsample"
        )


if __name__ == "__main__":
    hyperparameters_optimisation()
