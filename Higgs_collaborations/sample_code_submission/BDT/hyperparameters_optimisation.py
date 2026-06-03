from get_data import get_clean_splits
from boosted_decision_tree import BoostedDecisionTree

from sklearn.model_selection import train_test_split
import numpy as np


# ==================================================
# AMS (Approximate Median Significance)
# ==================================================
def ams(s, b, b_reg=10.0):
    """
    AMS standard utilisé en Higgs ML competitions.

    s = somme des poids signal
    b = somme des poids background
    b_reg = régularisation pour éviter instabilités
    """

    return np.sqrt(
        2 * (
            (s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s
        )
    )


# ==================================================
# Significance score basé sur un seuil
# ==================================================
def significance_score(y_true, y_score, sample_weight, threshold=0.5):

    y_pred = (y_score > threshold)

    s = np.sum(sample_weight[(y_true == 1) & (y_pred == 1)])
    b = np.sum(sample_weight[(y_true == 0) & (y_pred == 1)])

    return ams(s, b)


# ==================================================
# MAIN OPTIMIZATION
# ==================================================
def optimize_hyperparameters():

    print("\nChargement des données...")

    X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()

    # ==================================================
    # SOUS-ECHANTILLONNAGE 10%
    # ==================================================
    X_train, _, y_train, _, w_train, _ = train_test_split(
        X_train,
        y_train,
        w_train,
        train_size=0.10,
        random_state=42,
        stratify=y_train
    )

    X_test, _, y_test, _, w_test, _ = train_test_split(
        X_test,
        y_test,
        w_test,
        train_size=0.10,
        random_state=42,
        stratify=y_test
    )

    print(f"Train : {len(X_train):,} événements")
    print(f"Test  : {len(X_test):,} événements")

    # ==================================================
    # OPTIMISATION MAX_DEPTH
    # ==================================================
    depths = [5, 6, 7, 8, 9, 10]

    best_depth = None
    best_score = -np.inf

    print("\n===== MAX DEPTH OPTIMIZATION =====")

    for depth in depths:

        print(f"\nTesting max_depth = {depth}")

        model = BoostedDecisionTree()
        model.model.set_params(max_depth=depth)

        model.fit(X_train, y_train, weights=w_train)

        preds = model.predict(X_test)

        score = significance_score(
            y_test,
            preds,
            w_test
        )

        print(f"AMS = {score:.4f}")

        if score > best_score:
            best_score = score
            best_depth = depth

    # ==================================================
    # OPTIMISATION SUBSAMPLE / COLSAMPLE
    # ==================================================
    subsamples = [0.7, 0.8, 0.9]
    colsamples = [0.7, 0.8, 0.9]

    best_subsample = None
    best_colsample = None
    best_sampling_score = -np.inf

    print("\n===== SUBSAMPLE / COLSAMPLE OPTIMIZATION =====")

    for subsample in subsamples:
        for colsample in colsamples:

            print(f"\nTesting subsample={subsample}, colsample={colsample}")

            model = BoostedDecisionTree()

            model.model.set_params(
                max_depth=best_depth,
                subsample=subsample,
                colsample_bytree=colsample
            )

            model.fit(X_train, y_train, weights=w_train)

            preds = model.predict(X_test)

            score = significance_score(
                y_test,
                preds,
                w_test
            )

            print(f"AMS = {score:.4f}")

            if score > best_sampling_score:
                best_sampling_score = score
                best_subsample = subsample
                best_colsample = colsample

    # ==================================================
    # RESULTATS FINAUX
    # ==================================================
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)

    print(f"max_depth = {best_depth}")
    print(f"subsample = {best_subsample}")
    print(f"colsample_bytree = {best_colsample}")
    print(f"AMS = {best_sampling_score:.4f}")


if __name__ == "__main__":
    optimize_hyperparameters()