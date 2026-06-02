
###Max depth optimization for the BDT model. We test different values of max_depth and compute the AMS score on a validation set to find the best one###
from get_data import get_clean_splits
from boosted_decision_tree import BoostedDecisionTree
from courbes import significance_score

import numpy as np


def optimize_max_depth():

    X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()

    depths = [4, 5, 6, 7, 8, 9]

    best_depth = None
    best_significance = -np.inf

    results = []

    print("\n===== MAX DEPTH OPTIMIZATION =====")

    for depth in depths:

        print(f"\nTesting max_depth = {depth}")

        model = BoostedDecisionTree()

        model.model.set_params(
            max_depth=depth
        )

        model.fit(
            X_train,
            y_train,
            weights=w_train
        )

        predictions = model.predict(X_test)

        significance = significance_score(
            y_true=y_test,
            y_score=predictions,
            sample_weight=w_test
        )

        print(
            f"Significance = {significance:.4f}"
        )

        results.append(
            (depth, significance)
        )

        if significance > best_significance:

            best_significance = significance
            best_depth = depth

    print("\n===== RESULTS =====")

    for depth, sig in results:

        print(
            f"max_depth = {depth} "
            f"-> Z = {sig:.4f}"
        )

    print("\n===== BEST MODEL =====")

    print(
        f"Best depth = {best_depth}"
    )

    print(
        f"Best significance = {best_significance:.4f}"
    )

    return best_depth, results


if __name__ == "__main__":

    optimize_max_depth()