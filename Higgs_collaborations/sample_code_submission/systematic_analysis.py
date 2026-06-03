import numpy as np
from HiggsML.systematics import systematics
from utils import plot_given_hist
import model



### transform dataset into dataset that can be usd by model.predict

def get_data(dataset:dict):
    '''
    labels : background(0) or signal (1)
    weights : 
    detailled_labels : 
    '''
    training_set = dataset
    X = training_set["data"]
    y = training_set["labels"]
    w = training_set["weights"]

    n = len(y)
    split = int(n * 0.8) # 80% training 20% validation
    train_data, val_data = X[:split], X[split:] 
    train_labels, val_labels = y[:split], y[split:]
    train_weights, val_weights = w[:split], w[split:]

    return train_data, train_labels, train_weights, val_data, val_labels, val_weights

def signal_bck(score:list[float], label:list[int], weight:list[float]):
    signal = []
    s_weight = []
    bck = []
    b_weight = []
    for i in range(len(score)):
        if label[i]:
            signal.append(score[i])
            s_weight.append(weight[i])
        else:
            bck.append(score[i])
            b_weight.append(weight[i]) 
    return signal, s_weight, bck, b_weight 


def tes_fitter(model:model.Model, train_set:dict,  n_bins:int = 50):
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
    tes_values = np.linspace(0.9, 1.1, 150)

    # split train data and validation data
    data_set = train_set

     # Obtenir l'histogramme des scores nominaux 
    syst_set = systematics(data_set, tes=1)
    alt_data = get_data(syst_set)[0]

    # le score nominal

    score = model.predict(alt_data) 

    labels_data = get_data(syst_set)[1]
    labels_array = labels_data.to_numpy()
    weights_data = get_data(syst_set)[2]
    weights_array = weights_data.to_numpy()

    s_score, s_weight, b_score, b_weight = signal_bck(score, labels_array, weights_array)

    s_histogram,_ = np.histogram(
        s_score, bins=n_bins, range=(0, 1), weights=s_weight, density=True
    )
    b_histogram,_ = np.histogram(
        b_score, bins=n_bins, range=(0, 1), weights=b_weight, density=True
    )

    s_ref_data = s_histogram
    b_ref_data = b_histogram
   
    #plot_given_hist(s_ref_data, b_ref_data)
   
    # Remplacer les 0 par 1 pour éviter toute division par zéro lors du ratio
    # nom_hist_safe = np.where(nom_hist == 0, 1, nom_hist)
   
    hist1, hist2 = [], []
    # Boucle sur les différentes valeurs de TES (Task 1)

    for t in tes_values:
        syst_set = systematics(data_set, tes=t)
        alt_data = get_data(syst_set)[0]
        score = model.predict(alt_data) 
        labels_data = get_data(syst_set)[1]
        labels_array = labels_data.to_numpy()
        weights_data = get_data(syst_set)[2]
        weights_array = weights_data.to_numpy()

        s_score, s_weight, b_score, b_weight = signal_bck(score, labels_array, weights_array)

        # générer les histogrammes du signal et du background
        s_histogram,_ = np.histogram(
        s_score, bins=n_bins, range=(0, 1), weights=s_weight, density=True
        )
        b_histogram,_ = np.histogram(
            b_score, bins=n_bins, range=(0, 1), weights=b_weight, density=True
        )
        # On stocke la diff par rapport au cas nominal, bien voir que ici on stocke déjà la différence entre les histogrammes
        hist1.append(s_histogram - s_ref_data) 
        hist2.append(b_histogram - b_ref_data)
        # plot_given_hist(s_histogram, b_histogram, t)
       
    hist_arrays = np.array(hist1), np.array(hist2) # format voulu
   
    # 2. Ajuster l'histogramme diff bin par bin (Task 2)
    fit_functions_s, fit_functions_b = [], []
    for i in range(n_bins):
        # Ajustement de l'évolution du ratio dans le bin 'i'
        coefs_s = np.polyfit(tes_values, hist_arrays[0][:, i], deg=1)
        coefs_b = np.polyfit(tes_values, hist_arrays[1][:, i], deg=1)
        fit_functions_s.append(np.poly1d(coefs_s)) 
        fit_functions_b.append(np.poly1d(coefs_b))
       
    def fit_function(array, tes:float):
        # Cette fonction applique la transformation à un tableau nominal pour un TES donné
        #etape 1, caster
        my_array_s, my_array_b = np.array(array[0]), np.array(array[1])
        diff_computed_s = np.array([poly(tes) for poly in fit_functions_s])
        diff_computed_b = np.array([poly(tes) for poly in fit_functions_b])
        # Applique la transformation à un tableau nominal pour un TES donné
        transformed_array_s = my_array_s  + diff_computed_s 
        transformed_array_b = my_array_b  + diff_computed_b
        return np.array(transformed_array_s), np.array(transformed_array_b) # je caste une deuxième fois parce qu'on st jamais trop sûr


    def test_tes_fitter(transformateur, data_set, tes_value):
        print(f"------ début du test de validation pour un valeur de tes = {tes_value}-----")
        
        print("------génération de l'histogramme réel avec systematics-------")

        syst_reel = systematics(data_set, tes=tes_value)
        data_reel, labels_reel, weights_reel = get_data(syst_reel)[3:]

        score_reel = model.predict(data_reel)
        labels_reel_arr = labels_reel.to_numpy()
        weights_reel_arr = weights_reel.to_numpy()

        s_sc_rl, s_w_rl, b_sc_rl, b_w_rl = signal_bck(score_reel, labels_reel_arr, weights_reel_arr)

        vrai_s_hist, _ = np.histogram(s_sc_rl, bins=n_bins, range=(0, 1), weights=s_w_rl, density=True)
        vrai_b_hist, _ = np.histogram(b_sc_rl, bins=n_bins, range=(0, 1), weights=b_w_rl, density=True)

        print("-----evaluation avec la fonction générée------")
        # on extrait les signal et background nominaux
        val_data, labels_val, weights_val = get_data(data_set)[3:]
        score_val = model.predict(val_data)
        labels_val_arr = labels_val.to_numpy()
        weights_val_arr = weights_val.to_numpy()
        s_sc_val, s_w_val, b_sc_val, b_w_val = signal_bck(score_val, labels_val_arr, weights_val_arr)

        s_hist_val, _ = np.histogram(s_sc_val, bins=n_bins, range=(0, 1), weights=s_w_val, density=True)
        b_hist_val, _ = np.histogram(b_sc_val, bins=n_bins, range=(0, 1), weights=b_w_val, density=True)
        computed_s_hist, computed_b_hist = transformateur((s_hist_val, b_hist_val), tes_value)

        print("-----calcul de l'erreur quadratique moyenne-----")
        # calcul du MSE sur différentes valeurs
        mse_s, mse_b = [], []
        
        mse_s = np.mean(((computed_s_hist - vrai_s_hist)**2))
        mse_b = np.mean((computed_b_hist - vrai_b_hist)**2)
        print(f"Erreur quadratique moyenne (Signal) : {mse_s:.4f}")
        print(f"Erreur quadratique moyenne (Bkg)    : {mse_b:.4f}") 

    for tes in [0.98, 1.015, 1.13]:
        test_tes_fitter(fit_function, data_set, tes)




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
    # jes_values = np.linspace(0.97, 1.03, 10)
   
    # # Obtenir l'histogramme des scores nominaux
    # nom_set = systematics(train_set, jes=1.0)
    # nom_score = model.predict(nom_set["data"])
    # nom_hist, _ = np.histogram(nom_score, bins=10, range=(0, 1))
   
    # # Remplacer les 0 par 1 pour éviter la division par zéro
    # nom_hist_safe = np.where(nom_hist == 0, 1, nom_hist)
   
    # hist_diff = []
    # # Boucle sur les différentes valeurs de TES (Task 1)
    # for j in jes_values:
    #     syst_set = systematics(train_set, jes=j)
    #     score = model.predict(syst_set["data"])
    #     histogram, _ = np.histogram(score, bins=10, range=(0, 1))
       
    #     # On stocke le ratio par rapport au cas nominal
    #     diff = histogram - nom_hist
    #     plot_given_hist(diff)
    #     hist_diff.append(diff)
       
    # hist_diff = np.array(hist_diff) # format: (len(tes_values), 100)
   
    # # 2. Ajuster l'histogramme ratio bin par bin (Task 2)
    # fit_functions = []
    # for i in range(len(jes_values)):
    #     # Ajustement linéaire  de l'évolution du ratio dans le bin 'i'
    #     coefs = np.polyfit(jes_values, hist_diff[:, i], deg=7)
    #     fit_functions.append(np.poly1d(coefs)) 
       
    def fit_function(array, jes):
    #     # Cette fonction applique la transformation à un tableau nominal pour un TES donné
    #     #etape 1, caster
    #     my_array = np.array(array)
    #     ratio_computed = np.array([poly(jes) for poly in fit_functions])
    #     # Applique la transformation à un tableau nominal pour un JES donné
    #     transformed_array = my_array * ratio_computed #array[i] * [fit_functions[i](jes) for i in range(len(array))]
        return 1 # je caste une deuxième fois parce qu'on st jamais trop sûr

    return fit_function


################################################################ by beta



