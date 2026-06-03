import matplotlib.pyplot as plt
import numpy as np
from HiggsML.systematics import systematics


def tes_fitter(
    model,
    train_set,
):
    """
    Task 1 : Analysis TES Uncertainty
    1. Loop over different values of tes and make store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of tes and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given TES

    """
    syst_set = systematics(train_set, tes=1)
    score = model.predict(syst_set["data"])

    histogram = np.histogram(score, bins=100, range=(0, 1))

    # Write a function to loop over different values of tes and histogram and make fit function which transforms the histogram for any given TES

    def fit_function(array, tes):
        # Dummy fit function, replace with actual fitting procedure
        return [array[i] * f[i](tes) for i in range(len(array))]

    return fit_function


def jes_fitter(
    model,
    train_set,
):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions

      histogram and make fit function which transforms the histogram for any given jes

    """
    syst_set = systematics(train_set, jes=1)
    score = model.predict(syst_set["data"])

    histogram = np.histogram(score, bins=100, range=(0, 1))

    # Write a function to loop over different values of jes and histogram and make fit function which transforms the histogram for any given JES

    def fit_function(array, jes):
        # Dummy fit function, replace with actual fitting procedure
        return array * jes

    return fit_function

def met_fitter(
    model,
    eval_set,
    met_variations=np.array([-3.0, 0.0, 3.0])
):
    """
    Analyse de l'incertitude Soft MET.
    On applique des variations absolues (en GeV).
    """
    histograms = []
    bins = 100
    
    # 1. On génère les histogrammes pour chaque variation de MET
    for met in met_variations:
        # /!\ Vérifie le nom de l'argument: soft_met ou met_unclustered
        syst_set = systematics(eval_set, soft_met=met) 
        score = model.predict(syst_set["data"])
        
        hist, _ = np.histogram(score, bins=bins, range=(0, 1))
        histograms.append(hist)
        
    histograms = np.array(histograms)
    
    # 2. On "fit" un polynôme de degré 2 pour chaque bin de l'histogramme
    poly_coeffs = []
    for i in range(bins):
        coeffs = np.polyfit(met_variations, histograms[:, i], deg=2)
        poly_coeffs.append(coeffs)
        
    poly_coeffs = np.array(poly_coeffs)
    
    # 3. La fonction à renvoyer à stat
    def fit_function(base_histogram, met_value):
        new_hist = np.zeros_like(base_histogram, dtype=float)
        for i in range(len(new_hist)):
            new_hist[i] = np.polyval(poly_coeffs[i], met_value)
        return np.clip(new_hist, 0, None) # Empêche d'avoir un nombre de particules négatif

    return fit_function



    
def bkg_fitter(model, eval_set, bkg_variations=np.array([0.95, 1.00, 1.05])):
    """
    Analyse de l'incertitude sur la normalisation du bruit de fond (BKGnorm).
    Variation relative (0.95 = -5%, 1.05 = +5%).
    """
    histograms = []
    bins = 100
    
    for bkg in bkg_variations:
        # /!\ Vérifie le nom de l'argument: bkg_scale, bkg_norm, etc.
        syst_set = systematics(eval_set, bkg_scale=bkg)
        score = model.predict(syst_set["data"])
        
        hist, _ = np.histogram(score, bins=bins, range=(0, 1))
        histograms.append(hist)
        
    histograms = np.array(histograms)
    
    poly_coeffs = []
    for i in range(bins):
        coeffs = np.polyfit(bkg_variations, histograms[:, i], deg=2)
        poly_coeffs.append(coeffs)
        
    poly_coeffs = np.array(poly_coeffs)
    
    def fit_function(base_histogram, bkg_value):
        new_hist = np.zeros_like(base_histogram, dtype=float)
        for i in range(len(new_hist)):
            new_hist[i] = np.polyval(poly_coeffs[i], bkg_value)
        return np.clip(new_hist, 0, None)

    return fit_function

