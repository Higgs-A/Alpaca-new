import numpy as np
from HiggsML.systematics import systematics

def tes_fitter(model, train_set):
    """
    Task 1 : Analysis TES Uncertainty
    1. Loop over different values of tes and make store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of tes and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions
    """
    # 1. Définir une plage de valeurs TES autour de la valeur nominale (1.0)
    tes_values = np.linspace(0.97, 1.03, 10)
   
    # Obtenir l'histogramme des scores nominaux
    nom_set = systematics(train_set, tes=1.0)
    nom_score = model.predict(nom_set["data"])
    nom_hist, _ = np.histogram(nom_score, bins=100, range=(0, 1))
   
    # Remplacer les 0 par 1 pour éviter toute division par zéro lors du ratio
    nom_hist_safe = np.where(nom_hist == 0, 1, nom_hist)
   
    hist_ratios = []
    # Boucle sur les différentes valeurs de TES (Task 1)
    for t in tes_values:
        syst_set = systematics(train_set, tes=t)
        score = model.predict(syst_set["data"])
        histogram, _ = np.histogram(score, bins=100, range=(0, 1))
       
        # On stocke le ratio par rapport au cas nominal
        ratio = histogram / nom_hist_safe
        hist_ratios.append(ratio)
       
    hist_ratios = np.array(hist_ratios) # format: (len(tes_values), 100)
   
    # 2. Ajuster l'histogramme ratio bin par bin (Task 2)
    fit_functions = []
    for i in range(100):
        # Ajustement linéaire  de l'évolution du ratio dans le bin 'i'
        coefs = np.polyfit(tes_values, hist_ratios[:, i], deg=5)
        fit_functions.append(np.poly1d(coefs)) 
       
    def fit_function(array, tes):
        # Cette fonction applique la transformation à un tableau nominal pour un TES donné
        transformed_array = [array[i] * fit_functions[i](tes) for i in range(len(array))]
        return np.array(transformed_array)

    return fit_function


def jes_fitter(model, train_set):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions
    """
    # 1. Définir une plage de valeurs JES
    jes_values = np.linspace(0.97, 1.03, 7)
   
    # Obtenir l'histogramme des scores nominaux
    nom_set = systematics(train_set, jes=1.0)
    nom_score = model.predict(nom_set["data"])
    nom_hist, _ = np.histogram(nom_score, bins=100, range=(0, 1))
   
    # Remplacer les 0 par 1 pour éviter la division par zéro
    nom_hist_safe = np.where(nom_hist == 0, 1, nom_hist)
   
    hist_ratios = []
    # Boucle sur les différentes valeurs de JES (Task 1)
    for j in jes_values:
        syst_set = systematics(train_set, jes=j)
        score = model.predict(syst_set["data"])
        histogram, _ = np.histogram(score, bins=100, range=(0, 1))
       
        # On stocke le ratio par rapport au cas nominal
        ratio = histogram / nom_hist_safe
        hist_ratios.append(ratio)
       
    hist_ratios = np.array(hist_ratios) # format: (len(jes_values), 100)
   
    # 2. Ajuster l'histogramme ratio bin par bin (Task 2)
    fit_functions = []
    for i in range(100):
        # Ajustement linéaire (polynôme de degré 1) de l'évolution du ratio dans le bin 'i'
        coefs = np.polyfit(jes_values, hist_ratios[:, i], deg=1)
        fit_functions.append(np.poly1d(coefs))
       
    def fit_function(array, jes):
        # Applique la transformation à un tableau nominal pour un JES donné
        transformed_array = [array[i] * fit_functions[i](jes) for i in range(len(array))]
        return np.array(transformed_array)

    return fit_function