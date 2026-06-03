from kiwisolver import strength
import numpy as np
from HiggsML.systematics import systematics
from iminuit import Minuit
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

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

    score = score.flatten() > 0.5 
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
    
def compute_mu_unbinned_likelihood(mu0, S_scores, S_weights, B_scores, B_weights, Data_scores, Data_weights):
    '''
    Estimate mu by minimizing the NLL using Minuit and unbinned data
    '''
    pdf_S, pdf_B, N_S, N_B = prepare_unbinned(S_scores, S_weights, B_scores, B_weights) 
    nll_func = lambda mu: unbinned_NLL(mu, Data_scores, Data_weights, pdf_S, pdf_B, N_S, N_B)
    m = Minuit(nll_func, mu=1.0)
    m.limits["mu"] = (0, None)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad() 
    m.hesse()  
    print("mu_hat =", m.values["mu"])
    print("sigma_mu =", m.errors["mu"])
    print("NLL_min =", m.fval)

def compute_mu_binned_likelihood(mu0, N_bins, S_scores, S_weights, B_scores, B_weights, Data_scores, Data_weights):
    '''
    Estimate mu by minimizing the NLL using Minuit and bins
    '''
    Nb, Sb, Bb = prepare_binned(N_bins, S_scores, S_weights, B_scores, B_weights, Data_scores, Data_weights)
    m = Minuit(lambda mu: NLL(mu, Nb, Sb, Bb), mu=mu0) #mu est le paramètre à estimer, initialisé à mu0 
    m.errordef = Minuit.LIKELIHOOD # on minimise NLL
    m.migrad()  # recherche du minimum
    m.hesse()   # calcul des erreurs
    print("mu_hat =", m.values["mu"])  #valeur estimée de mu qui minimise la NLL
    print("sigma_mu =", m.errors["mu"]) #incertitudes sur mu
    print("NLL_min =", m.fval) # valeur minimale de NLL3
    

def calculate_saved_info(model, holdout_set):

    score = model.predict(holdout_set["data"])

    print("score shape before threshold", score.shape)

    #We compute the optimized threshold by avergage median signficance method
    MAX = sorted(set(score))[-1] 
    threshold_list = np.linspace(0, MAX, 100) #We generate 100 values of potential threshold values 
    ams = [0] * 100
 
    for i, t in enumerate(threshold_list): #We iterate over the values of threshold in order to compute AMS
 
        score_bis = score.flatten() > t
        score_bis = score_bis.astype(int)
 
        label = holdout_set["labels"]
 
        gamma = np.sum(holdout_set["weights"] * score2 * label)
 
        beta = np.sum(holdout_set["weights"] * score2 * (1 - label))
 
        ams[i] = np.sqrt(2 * ((gamma + beta) * np.log(1 + gamma / beta) - gamma))
 
    index = np.argmax(ams)
    threshold = threshold_list[best_idx]   #We keep the threshold value that maximizes the AMS

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
    '''Define the negative log-likelihood function for a counting experiment.''' 
    assert mu >= 0, "mu must be a positive integer"
    assert np.all(N >= 0) and np.all(S >= 0) and np.all(B >= 0), ("N, S and B must be positive integers")
    expected = mu * S + B
    nll_val = np.sum(expected - N * np.log(expected))
    return nll_val

#minuit
m = Minuit(lambda mu: NLL(mu, N, S, B), mu=1.0) #mu est le paramètre à estimer, initialisé à 1.0 (les autres paramètres sont fixés)
m.errordef = Minuit.LIKELIHOOD # on minimise NLL
m.migrad()  # recherche du minimum
m.hesse()   # calcul des erreurs
#résultats
print("mu_hat =", m.values["mu"])  #valeur estimée de mu qui minimise la NLL
print("sigma_mu =", m.errors["mu"]) #incertitudes sur mu
print("NLL_min =", m.fval)) # valeur minimale de NLL3



def prepare_unbinned(S_scores, S_weights, B_scores, B_weights):
    #création des PDFs continues grâce aux scores et poids des simulations
    pdf_S = gaussian_kde(S_scores, weights=S_weights)
    pdf_B = gaussian_kde(B_scores, weights=B_weights)
    #pour un nb total attendu mu=1
    N_S_expected = np.sum(S_weights)
    N_B_expected = np.sum(B_weights)
    return pdf_S, pdf_B, N_S_expected, N_B_expected

def unbinned_NLL(mu, Data_scores, Data_weights, pdf_S, pdf_B, N_S_exp, N_B_exp):
    '''Extended Unbinned Negative Log-Likelihood function'''
    if mu < 0: # pénalité si minuit teste un mu négatif
        return 1e10  
    # terme de Poisson étendu
    N_expected_total = mu * N_S_exp + N_B_exp
    # PDF évaluée pour chaque événement de Data
    f_S = pdf_S(Data_scores)
    f_B = pdf_B(Data_scores)
    # Combinaison : vraisemblance pour chaque événement individuel
    event_likelihood = (mu * N_S_exp * f_S + N_B_exp * f_B) / N_expected_total
    nll_val = N_expected_total - np.sum(Data_weights * np.log(event_likelihood))
    return nll_val



# PLOTS 

def plot_likelihood(n_obs, S, B, mu_hat, plot_show=True):
    '''Plot likelihood for Task 1a'''
    def neg_ll(mu):
        lam = mu * S + B
        lam = np.clip(lam, 1e-10, None)
        return -(n_obs * np.log(lam) - lam)

    mu_vals_full = np.linspace(0, 5, 1000)
    nll_vals_full = np.array([neg_ll(mu) for mu in mu_vals_full])
    nll_min = np.min(nll_vals_full)
    delta_nll_full = nll_vals_full - nll_min

    mask = delta_nll_full < 20
    mu_vals = mu_vals_full[mask]
    delta_nll = delta_nll_full[mask]

    left_mask = mu_vals < mu_hat
    right_mask = mu_vals > mu_hat

    try:
        from scipy.interpolate import interp1d
        left_interp = interp1d(delta_nll[left_mask], mu_vals[left_mask], bounds_error=False, fill_value="extrapolate")
        right_interp = interp1d(delta_nll[right_mask], mu_vals[right_mask], bounds_error=False, fill_value="extrapolate")
        mu_lower = float(left_interp(0.5))
        mu_upper = float(right_interp(0.5))
        delta_mu = mu_upper - mu_lower
    except Exception as e:
        mu_lower, mu_upper, delta_mu = mu_hat, mu_hat, 0.0
        print("Interpolation error:", e)

    plt.figure(figsize=(8, 5))
    plt.plot(mu_vals, delta_nll, label=r"Single Binned $\Delta$NLL", color="blue")
    plt.axvline(mu_hat, color="red", linestyle="--", label=rf"Single Binned $\hat\mu = {mu_hat:.3f}$")
    plt.axvline(mu_lower, color="green", linestyle="--", label=rf"Single Binned $\mu_{{-1\sigma}} = {mu_lower:.3f}$")
    plt.axvline(mu_upper, color="green", linestyle="--", label=rf"Single Binned $\mu_{{+1\sigma}} = {mu_upper:.3f}$")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$ Negative Log-Likelihood")
    plt.title(rf"Single Binned Profile Likelihood: $\delta\mu$ = {delta_mu:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_show: plt.show()


def plot_binned_likelihood(N_obs, S, B, mu_hat, plot_show=True):
    '''NLL pour la méthode binned Task 1b'''
    def neg_ll(mu):
        return NLL(mu, N_obs, S, B)

    mu_vals_full = np.linspace(0, 5, 1000)
    nll_vals_full = np.array([neg_ll(mu) for mu in mu_vals_full])
    nll_min = np.min(nll_vals_full)
    delta_nll_full = nll_vals_full - nll_min

    mask = delta_nll_full < 20
    mu_vals = mu_vals_full[mask]
    delta_nll = delta_nll_full[mask]

    left_mask = mu_vals < mu_hat
    right_mask = mu_vals > mu_hat

    try:
        from scipy.interpolate import interp1d
        left_interp = interp1d(delta_nll[left_mask], mu_vals[left_mask], bounds_error=False, fill_value="extrapolate")
        right_interp = interp1d(delta_nll[right_mask], mu_vals[right_mask], bounds_error=False, fill_value="extrapolate")
        mu_lower = float(left_interp(0.5))
        mu_upper = float(right_interp(0.5))
        delta_mu = mu_upper - mu_lower
    except Exception as e:
        mu_lower, mu_upper, delta_mu = mu_hat, mu_hat, 0.0
        print("Interpolation error:", e)

    plt.figure(figsize=(8, 5))
    plt.plot(mu_vals, delta_nll, label=r"Binned $\Delta$NLL", color="#4A90E2")
    plt.axvline(mu_hat, color="#D0021B", linestyle="--", label=rf"Binned $\hat\mu = {mu_hat:.3f}$")
    plt.axvline(mu_lower, color="#50E3C2", linestyle="--", label=rf"Binned $\mu_{{-1\sigma}} = {mu_lower:.3f}$")
    plt.axvline(mu_upper, color="#50E3C2", linestyle="--", label=rf"Binned $\mu_{{+1\sigma}} = {mu_upper:.3f}$")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$ Negative Log-Likelihood")
    plt.title(rf"Profile Binned Likelihood: $\delta\mu$ = {delta_mu:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_show: plt.show()


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
    '''Plot likelihood for unbinned data (Task 1b)'''
    def neg_ll(mu):
        return unbinned_NLL(mu, Data_scores, Data_weights, pdf_S, pdf_B, N_S_exp, N_B_exp)

    mu_vals_full = np.linspace(0, 5, 1000)
    nll_vals_full = np.array([neg_ll(mu) for mu in mu_vals_full])
    nll_min = np.min(nll_vals_full)
    delta_nll_full = nll_vals_full - nll_min

    mask = delta_nll_full < 20
    mu_vals = mu_vals_full[mask]
    delta_nll = delta_nll_full[mask]

    left_mask = mu_vals < mu_hat
    right_mask = mu_vals > mu_hat

    try:
        from scipy.interpolate import interp1d
        left_interp = interp1d(delta_nll[left_mask], mu_vals[left_mask], bounds_error=False, fill_value="extrapolate")
        right_interp = interp1d(delta_nll[right_mask], mu_vals[right_mask], bounds_error=False, fill_value="extrapolate")
        mu_lower = float(left_interp(0.5))
        mu_upper = float(right_interp(0.5))
        delta_mu = mu_upper - mu_lower
    except Exception as e:
        mu_lower, mu_upper, delta_mu = mu_hat, mu_hat, 0.0
        print("Interpolation error:", e)

    plt.figure(figsize=(8, 5))
    plt.plot(mu_vals, delta_nll, label=r"Unbinned $\Delta$NLL", color="#8B008B")
    plt.axvline(mu_hat, color="red", linestyle="--", label=rf"Unbinned $\hat\mu = {mu_hat:.3f}$")
    plt.axvline(mu_lower, color="green", linestyle="--", label=rf"Unbinned $\mu_{{-1\sigma}} = {mu_lower:.3f}$")
    plt.axvline(mu_upper, color="green", linestyle="--", label=rf"Unbinned $\mu_{{+1\sigma}} = {mu_upper:.3f}$")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\Delta$ Negative Log-Likelihood")
    plt.title(rf"Profile Unbinned Likelihood: $\delta\mu$ = {delta_mu:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_show: plt.show()


def plot_unbinned_distributions(Data_scores, Data_weights, pdf_S, pdf_B, N_S_exp, N_B_exp, mu_hat, plot_show=True):
    '''plot for unbinned distributions (Task 1b) (smoothed shape)'''
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