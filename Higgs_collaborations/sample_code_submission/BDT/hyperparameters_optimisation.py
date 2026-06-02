
###Max depth optimization for the BDT model. We test different values of max_depth and compute the AMS score on a validation set to find the best one###
import numpy as np
from sklearn.model_selection import train_test_split

from boosted_decision_tree import BoostedDecisionTree


def ams(s, b, b_reg=10.0):
    """
    AMS officielle HiggsML.
    """
    if b + b_reg <= 0:
        return 0.0

    radicand = 2 * (
        (s + b + b_reg)
        * np.log(1.0 + s / (b + b_reg))
        - s
    )

    return np.sqrt(max(radicand, 0.0))


def compute_best_ams(scores, labels, weights):
    """
    Cherche le meilleur AMS en balayant le seuil
    sur les scores du BDT.
    """

    best_ams = 0.0

    for threshold in np.linspace(0, 1, 500):

        selected = scores > threshold

        s = weights[(labels == 1) & selected].sum()
        b = weights[(labels == 0) & selected].sum()

        current_ams = ams(s, b)

        if current_ams > best_ams:
            best_ams = current_ams

    return best_ams


def optimize_max_depth(train_data, labels, weights):

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        train_data,
        labels,
        weights,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    depths = [4, 5, 6, 7, 8, 9]

    best_depth = None
    best_ams = -np.inf

    for depth in depths:

        print(f"\n=== Testing max_depth = {depth} ===")

        model = BoostedDecisionTree()

        model.model.set_params(max_depth=depth)

        model.fit(
            X_tr,
            y_tr,
            w_tr
        )

        scores = model.predict(X_val)

        current_ams = compute_best_ams(
            scores,
            y_val,
            w_val
        )

        print(f"AMS = {current_ams:.4f}")

        if current_ams > best_ams:
            best_ams = current_ams
            best_depth = depth

    print("\n========================")
    print("Optimization finished")
    print("========================")
    print(f"Best depth : {best_depth}")
    print(f"Best AMS   : {best_ams:.4f}")

    return best_depth

if __name__ == "__main__":

    best_depth = optimize_max_depth(
        train_data,
        labels,
        weights
    )

    print(f"Best depth = {best_depth}")