from kiwisolver import strength
import numpy as np
from HiggsML.systematics import systematics
from iminuit import Minuit
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def compute_mu(score, weight, saved_info):
    score = score.flatten() > saved_info["threshold"]
    score = score.astype(int)

    N_obs = np.sum(score * weight)

    mu = (N_obs - saved_info["beta"]) / saved_info["gamma"]

    del_mu_stat = (np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"])
    del_mu_sys = 0.0
    del_mu_tot = del_mu_stat

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }

def calculate_saved_info(model, holdout_set):
    score = model.predict(holdout_set["data"])
    MAX = sorted(set(score.flatten()))[-1] 
    threshold_list = np.linspace(0, MAX, 100) 
    ams = [0] * 100

    for i, t in enumerate(threshold_list):
        score_bis = score.flatten() > t
        score_bis = score_bis.astype(int)

        if np.sum(score_bis) == 0: 
            continue

        label = holdout_set["labels"]
        gamma = np.sum(holdout_set["weights"] * score_bis * label)
        beta = np.sum(holdout_set["weights"] * score_bis * (1 - label))
        ams[i] = np.sqrt(2 * ((gamma + beta) * np.log(1 + gamma / beta) - gamma))

    index = np.argmax(ams)
    threshold = threshold_list[index]
    score = score.flatten() > threshold
    score = score.astype(int)

    label = holdout_set["labels"]
    gamma = np.sum(holdout_set["weights"] * score * label)
    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
        "beta": beta,
        "gamma": gamma,
        "threshold": threshold
    }

    return saved_info

def prepare_binned(N_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights):
    bin_edges = np.linspace(0.0, 1.0, N_bins + 1)
    N_obs, _ = np.histogram(N_scores, bins=bin_edges, weights=N_weights)
    S, _ = np.histogram(S_scores, bins=bin_edges, weights=S_weights)
    B, _ = np.histogram(B_scores, bins=bin_edges, weights=B_weights)
    return N_obs, S, B

def NLL(mu, N, S, B):
    assert mu >= 0, "mu must be a positive integer"
    assert np.all(N >= 0) and np.all(S >= 0) and np.all(B >= 0), ("N, S and B must be positive integers")
    expected = mu * S + B
    nll_val = np.sum(expected - N * np.log(expected))
    return nll_val

def compute_mu_binned(mu0, N_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights):
    Nb, Sb, Bb = prepare_binned(N_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights)
    m = Minuit(lambda mu: NLL(mu, Nb, Sb, Bb), mu=mu0)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()
    return {
        "mu_hat": m.values["mu"],
        "sigma_mu": m.errors["mu"],
        "NLL_min": m.fval
    }

def prepare_unbinned(S_scores, S_weights, B_scores, B_weights):
    pdf_S = gaussian_kde(S_scores, weights=S_weights)
    pdf_B = gaussian_kde(B_scores, weights=B_weights)
    N_S_expected = np.sum(S_weights)
    N_B_expected = np.sum(B_weights)
    return pdf_S, pdf_B, N_S_expected, N_B_expected

def unbinned_NLL(mu, N_scores, N_weights, pdf_S, pdf_B, N_S_exp, N_B_exp):
    if mu < 0:
        return 1e10  
    N_expected_total = mu * N_S_exp + N_B_exp
    f_S = pdf_S(N_scores)
    f_B = pdf_B(N_scores)
    event_likelihood = (mu * N_S_exp * f_S + N_B_exp * f_B) / N_expected_total
    nll_val = N_expected_total - np.sum(N_weights * np.log(event_likelihood))
    return nll_val

def compute_mu_unbinned(mu0, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights):
    pdf_S, pdf_B, N_S, N_B = prepare_unbinned(S_scores, S_weights, B_scores, B_weights)
    nll_func = lambda mu: unbinned_NLL(mu, N_scores, N_weights, pdf_S, pdf_B, N_S, N_B)
    m = Minuit(nll_func, mu=1.0)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad() 
    m.hesse()  
    return {
        "mu_hat": m.values["mu"],
        "sigma_mu": m.errors["mu"],
        "NLL_min": m.fval
    }