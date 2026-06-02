import numpy as np
from HiggsML.systematics import systematics
from iminuit import Minuit

"""
Task 1a : Counting Estimator
1.write the saved_info dictionary such that it contains the following keys
    1. beta
    2. gamma
2. Estimate the mu using the formula
    mu = (sum(score * weight) - beta) / gamma
3. return the mu and its uncertainty

Task 1b : Stat-Only Likelihood Estimator
1. Modify the estimation of mu such that it uses the likelihood function
    1. Write a function for the likelihood function which profiles over mu
    2. Use Minuit to minimize the NLL

Task 2 : Systematic Uncertainty
1. substitute the beta and gamma with the tes_fit and jes_fit functions
2. Write a function to likelihood function which profiles over mu, tes and jes
3. Use Minuit to minimize the NLL
4. return the mu and its uncertainty

"""


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """

    score = score.flatten() > 0.5
    score = score.astype(int)

    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.0 * mu)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def calculate_saved_info(model, holdout_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """

    score = model.predict(holdout_set["data"])

    from systematic_analysis import tes_fitter
    from systematic_analysis import jes_fitter

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    label = holdout_set["labels"]

    print("score shape after threshold", score.shape)

    gamma = np.sum(holdout_set["weights"] * score * label)

    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
        "beta": beta,
        "gamma": gamma,
        "tes_fit": tes_fitter(model, holdout_set),
        "jes_fit": jes_fitter(model, holdout_set),
    }

    print("saved_info", saved_info)

    return saved_info


# TASK 1B : Stat-Only Likelihood Estimator

N_bins = 5
def prepare_binned(N_bins, S_scores, S_weights, B_scores, B_weights, Data_scores, Data_weights):
    '''Objective : splitting signal, background, and data into binned arrays for the NLL'''
    # bin boundaries between 0 and 1
    bin_edges = np.linspace(0.0, 1.0, N_bins + 1)
    # each array bin by bin
    N_obs, _ = np.histogram(Data_scores, bins=bin_edges, weights=Data_weights)
    S, _ = np.histogram(S_scores, bins=bin_edges, weights=S_weights)
    B, _ = np.histogram(B_scores, bins=bin_edges, weights=B_weights)
    return N_obs, S, B


def NLL(mu, N, S, B):
    """"
    Define the negative log-likelihood function for a counting experiment.
    Parameters:
    - mu: signal strength parameter (positive integer)
    - N: observed number of events (1D array of length n_bins of positive integers)
    - S: expected number of signal events for mu=1 (1D array of length n_bins of positive integers)
    - B: expected number of background events (1D array of length n_bins of positive integers)
    """    
    assert mu >= 0, "mu must be a positive integer"
    assert np.all(N >= 0) and np.all(S >= 0) and np.all(B >= 0), (
        "N, S and B must be positive integers"
    )
    expected = mu * S + B
    nll_val = np.sum(expected - N * np.log(expected))
    return nll_val


# N, S et B sont déjà définis et "fixes" dans le contexte de cette analyse 
# N : nombre d'événements observés
# S : nombre d'événements attendus du signal pour mu=1
# B : nombre d'événements attendus du background

m = Minuit(lambda mu: NLL(mu, N, S, B), mu=1.0) #mu est le paramètre à estimer, initialisé à 1.0 (les autres paramètres sont fixés)

m.errordef = Minuit.LIKELIHOOD # on minimise NLL

m.migrad()  # recherche du minimum
m.hesse()   # calcul des erreurs

print("mu_hat =", m.values["mu"])  #valeur estimée de mu qui minimise la NLL
print("sigma_mu =", m.errors["mu"]) #incertitudes sur mu
print("NLL_min =", m.fval)) # valeur minimale de NLL3
