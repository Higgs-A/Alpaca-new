from kiwisolver import strength
import numpy as np
from HiggsML.systematics import systematics
from iminuit import Minuit
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
#import Higgs_collaborations.sample_code_submission.systematic_analysis as sys

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

    score = score.flatten() > saved_info["threshold"]
    score = score.astype(int)

    N_obs = np.sum(score * weight)

    mu = (
        N_obs - saved_info["beta"]
    ) / saved_info["gamma"]

    del_mu_stat = (np.sqrt(saved_info["beta"]+ saved_info["gamma"])/ saved_info["gamma"])

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

    print("score shape before threshold", score.shape)

    #We compute the optimized threshold by avergage median signficance method
    MAX = sorted(set(score.flatten()))[-1] 
    threshold_list = np.linspace(0, MAX, 100) #We generate 100 values of potential threshold values 
    ams = [0] * 100
 
    for i, t in enumerate(threshold_list): #We iterate over the values of threshold in order to compute AMS
 
        score_bis = score.flatten() > t
        score_bis = score_bis.astype(int)
 
        if np.sum(score_bis)==0: 
            continue

        label = holdout_set["labels"]
 
        gamma = np.sum(holdout_set["weights"] * score_bis * label)
 
        beta = np.sum(holdout_set["weights"] * score_bis* (1 - label))
 
        ams[i] = np.sqrt(2 * ((gamma + beta) * np.log(1 + gamma / beta) - gamma))
 
    index = np.argmax(ams)
    threshold = threshold_list[index]
    score = score.flatten() > threshold
    score = score.astype(int)

    label = holdout_set["labels"]

    gamma = np.sum(
    holdout_set["weights"] * score * label
    )

    beta = np.sum(
    holdout_set["weights"] * score * (1 - label)
    )

    print("score shape after threshold", score.shape)

    gamma = np.sum(holdout_set["weights"] * score * label)

    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
    "beta": beta,
    "gamma": gamma,
    "threshold": threshold
    }

    print("saved_info", saved_info)
    return(saved_info)
N_bins = 5

def prepare_binned(N_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights):
    '''Objective : splitting signal, background, and data into binned arrays for the NLL'''
    # bin boundaries between 0 and 1
    bin_edges = np.linspace(0.0, 1.0, N_bins + 1)
    # each array bin by bin
    N_obs, _ = np.histogram(N_scores, bins=bin_edges, weights=N_weights)
    S, _ = np.histogram(S_scores, bins=bin_edges, weights=S_weights)
    B, _ = np.histogram(B_scores, bins=bin_edges, weights=B_weights)
    return N_obs, S, B


def NLL(mu, N, S, B):
    '''Define the negative log-likelihood function for a counting experiment.''' 
    assert mu >= 0, "mu must be a positive integer"
    assert np.all(N >= 0) and np.all(S >= 0) and np.all(B >= 0), ("N, S and B must be positive integers")
    expected = mu * S + B
    nll_val = np.sum(expected - N * np.log(expected))
    return nll_val

def compute_mu_binned(mu0, N_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights):
    '''
    Estimate mu using the binned likelihood method.
    '''
    Nb, Sb, Bb = prepare_binned(N_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights)
    m = Minuit(lambda mu: NLL(mu, Nb, Sb, Bb), mu=mu0) #mu est le paramètre à estimer, initialisé à 1.0 (les autres paramètres sont fixés)
    m.errordef = Minuit.LIKELIHOOD # on minimise NLL
    m.migrad()  # recherche du minimum
    m.hesse()   # calcul des erreurs
    #résultats
    print("mu_hat =", m.values["mu"])  #valeur estimée de mu qui minimise la NLL
    print("sigma_mu =", m.errors["mu"]) #incertitudes sur mu
    print("NLL_min =", m.fval) # valeur minimale de NLL3



def prepare_unbinned(S_scores, S_weights, B_scores, B_weights):
    """Création des PDFs continues grâce aux scores et poids des simulations."""
    pdf_S = gaussian_kde(S_scores, weights=S_weights)
    pdf_B = gaussian_kde(B_scores, weights=B_weights)
    
    # Pour un nombre total attendu mu=1
    N_S_expected = np.sum(S_weights)
    N_B_expected = np.sum(B_weights)
    return pdf_S, pdf_B, N_S_expected, N_B_expected


def unbinned_NLL(mu, N_scores, N_weights, pdf_S, pdf_B, N_S_exp, N_B_exp):
    if mu < 0:  # pénalité si minuit teste un mu négatif
        return 1e10  
    
    N_expected_total = mu * N_S_exp + N_B_exp
    
    # PDFs absolues
    f_S = pdf_S(N_scores)
    f_B = pdf_B(N_scores)
    
    # terme de densité absolue pour chaque événement observé
    prediction_par_evenement = mu * N_S_exp * f_S + N_B_exp * f_B
    prediction_par_evenement = np.maximum(prediction_par_evenement, 1e-10) # Évite le log(0)
    
    # Extended Unbinned NLL
    nll_val = N_expected_total - np.sum(N_weights * np.log(prediction_par_evenement))
    return nll_val


def compute_mu_unbinned(mu0, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights):
    """Ajustement du paramètre mu via Minuit."""
    pdf_S, pdf_B, N_S, N_B = prepare_unbinned(S_scores, S_weights, B_scores, B_weights)
    
    nll_func = lambda mu: unbinned_NLL(mu, N_scores, N_weights, pdf_S, pdf_B, N_S, N_B)
    m = Minuit(nll_func, mu=mu0)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad() 
    m.hesse()  
    
    # Résultats en console
    print("mu_hat =", m.values["mu"])
    print("sigma_mu =", m.errors["mu"])
    print("NLL_min =", m.fval)
    return m.values["mu"]


# PLOTS 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot_profile_likelihood_scan(
    n_obs,
    S,
    B,
    mu_hat,
    plot_show=True,
):
    """
    Plot ΔNLL(μ) for a binned likelihood scan, showing the best-fit μ and 1σ confidence interval.

    Paramètres: 
    N_obs : tableau, le nb de données observées dans chaque bin.
    S : tableau, le nb de signaux attendus dans chaque bin pour μ=1.
    B : tableau, le nb de fonds attendus dans chaque bin.
    mu_hat : la valeur de μ qui minimise la NLL.

    NLL : la fonction de log-vraisemblance négative à minimiser.
    """

    def neg_ll(mu):
        lam = mu * S + B
        lam = np.clip(lam, 1e-10, None)

        return -(n_obs * np.log(lam) - lam)

    # Scan μ
    mu_values = np.linspace(
        0,
        max(5, 2 * mu_hat + 1),
        1000,
    )

    nll_values = np.array(
        [neg_ll(mu) for mu in mu_values]
    )

    nll_min = np.min(nll_values)

    delta_nll = nll_values - nll_min

    # Mask pour se concentrer sur la région d'intérêt
    mask = delta_nll < 20

    mu_values = mu_values[mask]
    delta_nll = delta_nll[mask]

    # Trouver les points où ΔNLL = 0.5 pour l'intervalle de confiance à 1σ
    left_mask = mu_values < mu_hat
    right_mask = mu_values > mu_hat

    try:

        left_interp = interp1d(
            delta_nll[left_mask],
            mu_values[left_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        right_interp = interp1d(
            delta_nll[right_mask],
            mu_values[right_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        mu_minus = float(left_interp(0.5))
        mu_plus = float(right_interp(0.5))

    except Exception as e:

        print("Interpolation error:", e)

        mu_minus = mu_hat
        mu_plus = mu_hat

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        mu_values,
        delta_nll,
        linewidth=2,
        label=r"$\Delta$NLL$(\mu)$",
    )

    # Best-fit point
    plt.axvline(
        mu_hat,
        color="red",
        linestyle="--",
        label=rf"$\hat{{\mu}}={mu_hat:.3f}$",
    )

    plt.scatter(
        [mu_hat],
        [0],
        color="red",
        zorder=5,
    )

    # ΔNLL = 0.5
    plt.axhline(
        0.5,
        color="black",
        linestyle=":",
        label=r"$\Delta$NLL = 0.5",
    )

    # ±1σ interval
    plt.axvline(
        mu_minus,
        color="green",
        linestyle=":",
    )

    plt.axvline(
        mu_plus,
        color="green",
        linestyle=":",
    )

    plt.scatter(
        [mu_minus, mu_plus],
        [0.5, 0.5],
        color="green",
        zorder=5,
    )

    # Labels
    plt.annotate(
        rf"$\hat{{\mu}}={mu_hat:.3f}$",
        xy=(mu_hat, 0),
        xytext=(10, 10),
        textcoords="offset points",
    )

    plt.annotate(
        rf"$\mu_{{-1\sigma}}={mu_minus:.3f}$",
        xy=(mu_minus, 0.5),
        xytext=(-70, 10),
        textcoords="offset points",
    )

    plt.annotate(
        rf"$\mu_{{+1\sigma}}={mu_plus:.3f}$",
        xy=(mu_plus, 0.5),
        xytext=(10, 10),
        textcoords="offset points",
    )

    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$NLL")

    plt.title(
        "Profile Likelihood Scan"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if plot_show:
        plt.show()

    return mu_minus, mu_plus


def plot_binned_profile_likelihood(
    N_obs,
    S,
    B,
    mu_hat,
    NLL,
    plot_show=True,
):
    """
    Plot ΔNLL(μ) for a binned likelihood scan, showing the best-fit μ and 1σ confidence interval.

    Paramètres: 
    N_obs : tableau, le nb de données observées dans chaque bin.
    S : tableau, le nb de signaux attendus dans chaque bin pour μ=1.
    B : tableau, le nb de fonds attendus dans chaque bin.
    mu_hat : la valeur de μ qui minimise la NLL.

    NLL : la fonction de log-vraisemblance négative à minimiser.
    """

    # Scan range around the best-fit value
    mu_values = np.linspace(0, max(5, 2 * mu_hat + 1), 1000)

    # Compute NLL scan
    nll_values = np.array(
        [NLL(mu, N_obs, S, B) for mu in mu_values]
    )

    #ΔNLL
    nll_min = np.min(nll_values)
    delta_nll = nll_values - nll_min

    # Mask pour se concentrer sur la région d'intérêt
    mask = delta_nll < 20

    mu_values = mu_values[mask]
    delta_nll = delta_nll[mask]

    # Trouver les points où ΔNLL = 0.5 pour l'intervalle de confiance à 1σ
    left_mask = mu_values < mu_hat
    right_mask = mu_values > mu_hat

    try:
        left_interp = interp1d(
            delta_nll[left_mask],
            mu_values[left_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        right_interp = interp1d(
            delta_nll[right_mask],
            mu_values[right_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        mu_minus = float(left_interp(0.5))
        mu_plus = float(right_interp(0.5))

    except Exception as e:

        print("Interpolation error:", e)

        mu_minus = mu_hat
        mu_plus = mu_hat

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        mu_values,
        delta_nll,
        linewidth=2,
        label=r"$\Delta$NLL$(\mu)$",
    )

    # Best-fit μ
    plt.axvline(
        mu_hat,
        color="red",
        linestyle="--",
        label=rf"$\hat{{\mu}}={mu_hat:.3f}$",
    )

    plt.scatter(
        [mu_hat],
        [0],
        color="red",
        zorder=5,
    )

    # 1σ horizontal line
    plt.axhline(
        0.5,
        color="black",
        linestyle=":",
        label=r"$\Delta$NLL = 0.5",
    )

    # 1σ interval
    plt.axvline(
        mu_minus,
        color="green",
        linestyle=":",
    )

    plt.axvline(
        mu_plus,
        color="green",
        linestyle=":",
    )

    plt.scatter(
        [mu_minus, mu_plus],
        [0.5, 0.5],
        color="green",
        zorder=5,
    )

    # Annotations
    plt.annotate(
        rf"$\hat{{\mu}}={mu_hat:.3f}$",
        xy=(mu_hat, 0),
        xytext=(10, 10),
        textcoords="offset points",
    )

    plt.annotate(
        rf"$\mu_{{-1\sigma}}={mu_minus:.3f}$",
        xy=(mu_minus, 0.5),
        xytext=(-70, 10),
        textcoords="offset points",
    )

    plt.annotate(
        rf"$\mu_{{+1\sigma}}={mu_plus:.3f}$",
        xy=(mu_plus, 0.5),
        xytext=(10, 10),
        textcoords="offset points",
    )

    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$NLL")

    plt.title(
        "Binned Profile Likelihood Scan"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if plot_show:
        plt.show()

    return mu_minus, mu_plus


def plot_binned_histograms(N_obs, S, B, mu_hat, N_bins=5, plot_show=True):
    '''Binned histograms for Task 1b'''
    plt.figure(figsize=(8, 5))
    bin_edges = np.linspace(0.0, 1.0, N_bins + 1)
    width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.bar(bin_centers, N_obs, width=width, alpha=0.5, label="Observed", edgecolor="black")
    plt.step(bin_centers, B, where="mid", label="Background", color="orange")
    plt.step(bin_centers, mu_hat * S + B, where="mid", label=f"Signal + Background (mu={mu_hat:.2f})", color="green")

    plt.xlabel("Score")
    plt.ylabel("Weighted Events")
    plt.legend()
    plt.grid(True)
    plt.title("Binned Histogram: Observed vs Model Prediction")
    plt.tight_layout()
    if plot_show: plt.show()






def plot_unbinned_likelihood(Data_scores, Data_weights, pdf_S, pdf_B, N_S_exp, N_B_exp, mu_hat, plot_show=True):
    def neg_ll(mu):
        return unbinned_NLL(mu, Data_scores, Data_weights, pdf_S, pdf_B, N_S_exp, N_B_exp)
 
    # scan large pour trouver l'ordre de grandeur des intersections
    mu_vals_full = np.linspace(max(0, mu_hat - 2.5), mu_hat + 2.5, 1000)
    nll_vals_full = np.array([neg_ll(mu) for mu in mu_vals_full])
    nll_min = np.min(nll_vals_full)
    delta_nll_full = nll_vals_full - nll_min
 
    # séparation gauche/droite pour l'interpolation de l'erreur à 1-sigma
    left_mask = mu_vals_full < mu_hat
    right_mask = mu_vals_full > mu_hat
 
    try:
        left_interp = interp1d(delta_nll_full[left_mask], mu_vals_full[left_mask], bounds_error=False, fill_value="extrapolate")
        right_interp = interp1d(delta_nll_full[right_mask], mu_vals_full[right_mask], bounds_error=False, fill_value="extrapolate")
        mu_lower = float(left_interp(0.5))
        mu_upper = float(right_interp(0.5))
        delta_mu = mu_upper - mu_lower
    except Exception as e:
        mu_lower, mu_upper, delta_mu = mu_hat, mu_hat, 0.2
        print("Interpolation error:", e)
 

    demi_fenetre = max(0.5, 3.5 * (delta_mu / 2.0))
    xmin = max(0, mu_hat - demi_fenetre)
    xmax = mu_hat + demi_fenetre

    mu_vals = np.linspace(xmin, xmax, 500)
    delta_nll = np.array([neg_ll(mu) for mu in mu_vals]) - nll_min
 
    plt.figure(figsize=(10, 6.5))
   
    err_moins = mu_hat - mu_lower
    err_plus = mu_upper - mu_hat
   
    label_courbe = rf"Extended $\Delta$NLL : $\mu = {mu_hat:.3f}_{{-{err_moins:.3f}}}^{{+{err_plus:.3f}}}$ | $\Delta\mu_{{1\sigma}} = {delta_mu:.3f}$"
    plt.plot(mu_vals, delta_nll, label=label_courbe, color="#8B008B", lw=2.5)
    
    plt.axvline(mu_hat, color="red", linestyle="--", lw=1.5, alpha=0.6)
    plt.plot(mu_hat, 0, 'ro', markersize=8)
    
    plt.axhline(0.5, color="gray", linestyle="-.", alpha=0.5, label=r"$\Delta$NLL = 0.5 ($\pm 1\sigma$)")
    
    plt.axvline(mu_lower, color="green", linestyle=":", lw=1.5, alpha=0.8)
    plt.axvline(mu_upper, color="green", linestyle=":", lw=1.5, alpha=0.8)
    plt.scatter([mu_lower, mu_upper], [0.5, 0.5], color="green", s=50, zorder=5)
    
    plt.hlines(0.5, mu_lower, mu_upper, colors="green", linestyles="-", lw=2)
   
    plt.text(mu_hat + (demi_fenetre * 0.03), 0.15, rf"$\hat{{\mu}} = {mu_hat:.3f}$", color="red", fontweight="bold", fontsize=10)
    plt.text(mu_lower - (demi_fenetre * 0.25), 0.65, rf"$\mu_{{-1\sigma}} = {mu_lower:.3f}$", color="green", fontsize=9, fontweight="bold")
    plt.text(mu_upper + (demi_fenetre * 0.03), 0.65, rf"$\mu_{{+1\sigma}} = {mu_upper:.3f}$", color="green", fontsize=9, fontweight="bold")
   
    plt.xlabel(r"Force du Signal $\mu$", fontsize=11)
    plt.ylabel(r"$\Delta$NLL = NLL - $\text{NLL}_{min}$", fontsize=11)
    plt.title(r"Extended Unbinned Profile Likelihood Scan (Parfaitement Centré)", fontsize=12, fontweight='bold', pad=12)
   
    plt.xlim(xmin, xmax)
    plt.ylim(-0.15, 4.5) 
   
    plt.legend(loc="upper center", fontsize=10, framealpha=0.95)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
   
    if plot_show:
        plt.show()


def plot_unbinned_distributions(Data_scores, Data_weights, pdf_S, pdf_B, N_S_exp, N_B_exp, mu_hat, plot_show=True):
    """Plot des formes lissées des distributions (KDE)."""
    plt.figure(figsize=(8, 5))
    counts, bins, _ = plt.hist(Data_scores, bins=40, weights=Data_weights, alpha=0.3, label="Observed Data", color="gray", edgecolor="black")
    
    bin_width = bins[1] - bins[0]
    x_plot = np.linspace(0, 1, 500)
    
    bkg_line = pdf_B(x_plot) * N_B_exp * bin_width
    plt.plot(x_plot, bkg_line, label="Background", color="orange", lw=2)
    
    sig_bkg_line = (mu_hat * N_S_exp * pdf_S(x_plot) + N_B_exp * pdf_B(x_plot)) * bin_width
    plt.plot(x_plot, sig_bkg_line, label=f"Signal + Background (mu={mu_hat:.2f})", color="green", lw=2)
    
    plt.xlabel("Score")
    plt.ylabel("Events")
    plt.title("Unbinned Distribution: Observed vs Model Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_show: plt.show()





# Task 2: on ne travaille qu'avec une méthode binned pour l'instant
 
# on suppose que f et g sont les fonctions d'interpolations calculées par systematic_analysis.py pour gamma i et beta i respectivement
# f et g sont sous la forme d'un tableau de fonctions évaluées dans les bins, par exemple f[i](tes) donne la valeur de la fonction f pour le bin i et une valeur de tes. De même pour g[i](jes).
# n-obs: tableau des nombres d'observations dans les bins, par exemple n_obs[i] donne le nombre d'observations dans le bin i.
# tes_min et tes_max sont les limites de tes, jes_min et jes_max sont les limites de jes. Ces limites seront utilisées pour contraindre les paramètres tes et jes lors de l'optimisation avec Minuit.
 
import numpy as np
import matplotlib.pyplot as plt

from iminuit import Minuit
from scipy.interpolate import interp1d

from systematic_analysis import (
    generer_saved_info,
    N_total_bin,
)

##############################################################################
# NLL COMPLETE
##############################################################################

def NLL_systematics(
    mu,
    tes,
    jes,
    bnorm,
    smet,
    n_obs,
    saved_info,
):

    nll = 0.0

    n_bins = len(n_obs)

    for i in range(n_bins):

        S_i = N_total_bin(
            bin_i=i,
            tes=tes,
            jes=jes,
            bnorm=bnorm,
            smet=smet,
            saved_info=saved_info,
            classe="S",
        )

        B_i = N_total_bin(
            bin_i=i,
            tes=tes,
            jes=jes,
            bnorm=bnorm,
            smet=smet,
            saved_info=saved_info,
            classe="B",
        )

        lam = mu * S_i + B_i

        lam = max(lam, 1e-10)

        nll += lam - n_obs[i] * np.log(lam)

    ##########################################################################
    # CONTRAINTES GAUSSIENNES
    ##########################################################################

    nll += 0.5 * ((tes - 1.0) / 0.03) ** 2

    nll += 0.5 * ((jes - 1.0) / 0.03) ** 2

    nll += 0.5 * ((bnorm - 1.0) / 0.05) ** 2

    nll += 0.5 * ((smet - 0.0) / 3.0) ** 2

    return nll


##############################################################################
# FIT GLOBAL
##############################################################################

def fit_global(
    n_obs,
    saved_info,
    active_nuisances,
):

    def nll(
        mu,
        tes,
        jes,
        bnorm,
        smet,
    ):

        return NLL_systematics(
            mu,
            tes,
            jes,
            bnorm,
            smet,
            n_obs,
            saved_info,
        )

    m = Minuit(
        nll,
        mu=1.0,
        tes=1.0,
        jes=1.0,
        bnorm=1.0,
        smet=0.0,
    )

    m.errordef = Minuit.LIKELIHOOD

    m.limits["mu"] = (0.0, None)

    m.limits["tes"] = (0.97, 1.03)

    m.limits["jes"] = (0.97, 1.03)

    m.limits["bnorm"] = (0.95, 1.05)

    m.limits["smet"] = (0.0, 3.0)

    all_nuisances = [
        "tes",
        "jes",
        "bnorm",
        "smet",
    ]

    for nuisance in all_nuisances:

        if nuisance not in active_nuisances:

            m.fixed[nuisance] = True

            if nuisance == "smet":
                m.values[nuisance] = 0.0
            else:
                m.values[nuisance] = 1.0

    m.migrad()
    m.hesse()

    return m


##############################################################################
# PROFILAGE DES NUISANCES
##############################################################################

def profile_nuisances(
    mu_fixed,
    n_obs,
    saved_info,
    active_nuisances,
):

    def nll_profiled(
        tes,
        jes,
        bnorm,
        smet,
    ):

        return NLL_systematics(
            mu_fixed,
            tes,
            jes,
            bnorm,
            smet,
            n_obs,
            saved_info,
        )

    m = Minuit(
        nll_profiled,
        tes=1.0,
        jes=1.0,
        bnorm=1.0,
        smet=0.0,
    )

    m.errordef = Minuit.LIKELIHOOD

    m.limits["tes"] = (0.97, 1.03)

    m.limits["jes"] = (0.97, 1.03)

    m.limits["bnorm"] = (0.95, 1.05)

    m.limits["smet"] = (0.0, 3.0)

    all_nuisances = [
        "tes",
        "jes",
        "bnorm",
        "smet",
    ]

    for nuisance in all_nuisances:

        if nuisance not in active_nuisances:

            m.fixed[nuisance] = True

            if nuisance == "smet":
                m.values[nuisance] = 0.0
            else:
                m.values[nuisance] = 1.0

    m.migrad()

    return m.fval


##############################################################################
# PROFILE LIKELIHOOD SCAN
##############################################################################

def profile_scan_mu(
    mu_values,
    n_obs,
    saved_info,
    active_nuisances,
):

    fit = fit_global(
        n_obs,
        saved_info,
        active_nuisances,
    )

    global_min = fit.fval

    profiled_nll = []

    for mu in mu_values:

        nll_mu = profile_nuisances(
            mu,
            n_obs,
            saved_info,
            active_nuisances,
        )

        profiled_nll.append(
            nll_mu
        )

    profiled_nll = np.array(
        profiled_nll
    )

    delta_nll = (
        profiled_nll
        - global_min
    )

    return delta_nll


##############################################################################
# EXTRACTION SIGMA
##############################################################################

def extract_sigma(
    mu_values,
    delta_nll,
):

    idx_min = np.argmin(
        delta_nll
    )

    mu_hat = mu_values[idx_min]

    left_mask = mu_values < mu_hat

    right_mask = mu_values > mu_hat

    try:

        left_interp = interp1d(
            delta_nll[left_mask],
            mu_values[left_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        right_interp = interp1d(
            delta_nll[right_mask],
            mu_values[right_mask],
            bounds_error=False,
            fill_value="extrapolate",
        )

        mu_minus = float(
            left_interp(0.5)
        )

        mu_plus = float(
            right_interp(0.5)
        )

        sigma_mu = (
            mu_plus - mu_minus
        ) / 2.0

    except Exception:

        mu_minus = mu_hat

        mu_plus = mu_hat

        sigma_mu = np.nan

    return (
        mu_hat,
        mu_minus,
        mu_plus,
        sigma_mu,
    )


##############################################################################
# PLOT FINAL
##############################################################################

def plot_profiled_nuisance_impact(
    n_obs,
    saved_info,
):

    mu_values = np.linspace(
        0.0,
        3.0,
        120,
    )

    scenarios = [

        {
            "label": "Stat only",
            "active": [],
        },

        {
            "label": "TES",
            "active": [
                "tes",
            ],
        },

        {
            "label": "TES + JES",
            "active": [
                "tes",
                "jes",
            ],
        },

        {
            "label": "TES + JES + BNORM + SMET",
            "active": [
                "tes",
                "jes",
                "bnorm",
                "smet",
            ],
        },
    ]

    plt.figure(
        figsize=(12, 8)
    )

    for scenario in scenarios:

        print(
            "\nFit scenario :",
            scenario["label"]
        )

        delta_nll = profile_scan_mu(
            mu_values,
            n_obs,
            saved_info,
            scenario["active"],
        )

        (
            mu_hat,
            mu_minus,
            mu_plus,
            sigma_mu,
        ) = extract_sigma(
            mu_values,
            delta_nll,
        )

        print(
            f"mu_hat = {mu_hat:.4f}"
        )

        print(
            f"sigma_mu = {sigma_mu:.4f}"
        )

        plt.plot(
            mu_values,
            delta_nll,
            linewidth=2.5,
            label=(
                f"{scenario['label']} "
                f"(σμ={sigma_mu:.3f})"
            ),
        )

        plt.scatter(
            [mu_hat],
            [0],
            s=50,
        )

        plt.scatter(
            [mu_minus, mu_plus],
            [0.5, 0.5],
            s=30,
        )

    plt.axhline(
        0.5,
        linestyle="--",
        color="black",
        label=r"$\Delta NLL = 0.5$",
    )

    plt.xlabel(
        r"$\mu$",
        fontsize=13,
    )

    plt.ylabel(
        r"$\Delta NLL$",
        fontsize=13,
    )

    plt.title(
        "Profile Likelihood Scan : Impact des Paramètres de Nuisance",
        fontsize=14,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


##############################################################################
# EXECUTION
##############################################################################

saved_info = generer_saved_info(
    model,
    training_dict,
)

##########################################################################
# n_obs doit être un tableau de longueur = nombre de bins
#
# Exemple :
#
# n_obs = np.array([...])
#
##########################################################################

plot_profiled_nuisance_impact(
    n_obs,
    saved_info,
)
