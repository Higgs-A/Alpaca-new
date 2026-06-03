from get_data import get_clean_splits
from boosted_decision_tree import BoostedDecisionTree
from courbes import significance_score

import numpy as np
import pandas as pd


def optimize_hyperparameters():

    X_train, X_test, y_train, y_test, w_train, w_test = get_clean_splits()

    
    # PARTIE 1 : MAX_DEPTH
    

    depth_results = []

    depths = [5, 6, 7, 8, 9, 10]

    best_depth = None
    best_depth_score = -np.inf

    print("\n===== MAX DEPTH OPTIMIZATION =====")

    for depth in depths:

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

        depth_results.append(
            [depth, significance]
        )

        if significance > best_depth_score:

            best_depth_score = significance
            best_depth = depth

    depth_df = pd.DataFrame(
        depth_results,
        columns=[
            "max_depth",
            "AMS"
        ]
    )

    
    # PARTIE 2 : SUBSAMPLE / COLSAMPLE

    sampling_results = []

    subsamples = [0.7, 0.8, 0.9]
    colsamples = [0.7, 0.8, 0.9]

    best_subsample = None
    best_colsample = None
    best_sampling_score = -np.inf

    print("\n===== SUBSAMPLE / COLSAMPLE OPTIMIZATION =====")

    for subsample in subsamples:

        for colsample in colsamples:

            model = BoostedDecisionTree()

            model.model.set_params(

                max_depth=best_depth,

                subsample=subsample,

                colsample_bytree=colsample
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

            sampling_results.append(
                [
                    subsample,
                    colsample,
                    significance
                ]
            )

            if significance > best_sampling_score:

                best_sampling_score = significance

                best_subsample = subsample
                best_colsample = colsample

    sampling_df = pd.DataFrame(
        sampling_results,
        columns=[
            "subsample",
            "colsample_bytree",
            "AMS"
        ]
    )

    
    # AFFICHAGE
    

    print("\n")
    print("=" * 60)
    print("TABLEAU 1 : MAX_DEPTH")
    print("=" * 60)

    print(
        depth_df.sort_values(
            by="AMS",
            ascending=False
        ).to_string(index=False)
    )

    print("\n")
    print("=" * 60)
    print("TABLEAU 2 : SUBSAMPLE / COLSAMPLE")
    print("=" * 60)

    print(
        sampling_df.sort_values(
            by="AMS",
            ascending=False
        ).to_string(index=False)
    )

    print("\n")
    print("=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)

    print(f"max_depth = {best_depth}")
    print(f"subsample = {best_subsample}")
    print(f"colsample_bytree = {best_colsample}")
    print(f"AMS = {best_sampling_score:.4f}")

    depth_df.to_csv(
        "depth_optimization.csv",
        index=False
    )

    sampling_df.to_csv(
        "sampling_optimization.csv",
        index=False
    )


if __name__ == "__main__":

    optimize_hyperparameters()