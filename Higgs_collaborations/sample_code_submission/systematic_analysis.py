import numpy as np
import matplotlib.pyplot as plt
from HiggsML.systematics import systematics


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


def tes_fitter(model, train_set:dict,  n_bins:int = 50):
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
    tes_values = np.linspace(0.97, 1.03, 200)

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
        labels_array = labels_data #.to_numpy()
        weights_data = get_data(syst_set)[2]
        weights_array = weights_data #.to_numpy()

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
       
    def fit_function(tes:float):
        # Cette fonction applique la transformation à un tableau nominal pour un TES donné
        #etape 1, caster
        diff_computed_s = np.array([poly(tes) for poly in fit_functions_s])
        diff_computed_b = np.array([poly(tes) for poly in fit_functions_b])
        # Applique la transformation à un tableau nominal pour un TES donné
        transformed_array_s = diff_computed_s 
        transformed_array_b = diff_computed_b
        return np.array(transformed_array_s), np.array(transformed_array_b) # je caste une deuxième fois parce qu'on st jamais trop sûr


    # def unique_test_tes_fitter(transformateur, data_set, tes_value):
    #     print(f"------ début du test de validation pour un valeur de tes = {tes_value}-----") 
    #     print("------génération de l'histogramme réel avec systematics-------")
 
    #     syst_reel = systematics(data_set, tes=tes_value)
    #     data_reel, labels_reel, weights_reel = get_data(syst_reel)[3:]
 
    #     score_reel = model.predict(data_reel)
    #     labels_reel_arr = labels_reel.to_numpy()
    #     weights_reel_arr = weights_reel.to_numpy()
 
    #     s_sc_rl, s_w_rl, b_sc_rl, b_w_rl = signal_bck(score_reel, labels_reel_arr, weights_reel_arr)
 
    #     vrai_s_hist, _ = np.histogram(s_sc_rl, bins=n_bins, range=(0, 1), weights=s_w_rl, density=False)
    #     vrai_b_hist, _ = np.histogram(b_sc_rl, bins=n_bins, range=(0, 1), weights=b_w_rl, density=False)

    #     print("-----evaluation avec la fonction générée------")
    #     # on extrait les signal et background nominaux
    #     val_data, labels_val, weights_val = get_data(data_set)[3:]
    #     score_val = model.predict(val_data)
    #     labels_val_arr = labels_val.to_numpy()
    #     weights_val_arr = weights_val.to_numpy()
    #     s_sc_val, s_w_val, b_sc_val, b_w_val = signal_bck(score_val, labels_val_arr, weights_val_arr)
 
    #     s_hist_val, _ = np.histogram(s_sc_val, bins=n_bins, range=(0, 1), weights=s_w_val, density=False)
    #     b_hist_val, _ = np.histogram(b_sc_val, bins=n_bins, range=(0, 1), weights=b_w_val, density=False)
    #     delta_s, delta_b = transformateur((s_hist_val, b_hist_val), tes_value)


    #     computed_s_hist = s_hist_val + delta_s
    #     computed_b_hist = b_hist_val + delta_b
    #     print("-----calcul de l'erreur quadratique moyenne-----")
    #     # calcul du MSE sur différentes valeurs
    #     mse_s, mse_b = [], []
    #     mse_s = np.mean(((computed_s_hist - vrai_s_hist)**2))
    #     mse_b = np.mean((computed_b_hist - vrai_b_hist)**2)
    #     print(f"Erreur quadratique moyenne (Signal) : {mse_s:.4f}")
    #     print(f"Erreur quadratique moyenne (Bkg)    : {mse_b:.4f}") 
    #     return mse_s, mse_b

    # def test_tes_fitter(fit_function, data_set):
    #     L_tes = np.linspace(0.9, 1.1, 50)
    #     L_mse_b = []
    #     L_mse_s = []
    #     for tes in L_tes:
    #         mse_s, mse_b = unique_test_tes_fitter(fit_function, data_set, tes)
    #         L_mse_s.append(mse_s)
    #         L_mse_b.append(mse_b)
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(L_tes, L_mse_b, color="#175a8a", label="MSE pour le bkg")
    #     plt.plot(L_tes, L_mse_s, color="#e40505d0", label="MSE pour le signal")
    #     plt.title("visualisation de l'erreur commise avec la fonction")
    #     plt.legend()
    #     plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    #     plt.show()



    return fit_function




def jes_fitter(model, train_set, n_bins:int = 100):
    """
    Task 1 : Analysis JES Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions
    """
    # 1. Définir une plage de valeurs JES autour de la valeur nominale (1.0)
    jes_values = np.linspace(0.97, 1.03, 200)
 
    # split train data and validation data
    data_set = train_set
 
    #  Obtenir l'histogramme des scores nominaux 
    syst_set = systematics(data_set, jes=1)
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
    
    # plot_given_hist(s_ref_data, b_ref_data)
    
    # Remplacer les 0 par 1 pour éviter toute division par zéro lors du ratio
    # nom_hist_safe = np.where(nom_hist == 0, 1, nom_hist)
    
    hist1, hist2 = [], []
    # Boucle sur les différentes valeurs de JES (Task 1)
 
    for t in jes_values:
        syst_set = systematics(data_set, jes=t)
        alt_data = get_data(syst_set)[0]
        score = model.predict(alt_data) 
        labels_data = get_data(syst_set)[1]
        labels_array = labels_data.to_numpy()
        weights_data = get_data(syst_set)[2]
        weights_array = weights_data.to_numpy()
# 
        s_score, s_weight, b_score, b_weight = signal_bck(score, labels_array, weights_array)
# 
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
        coefs_s = np.polyfit(jes_values, hist_arrays[0][:, i], deg=1)
        coefs_b = np.polyfit(jes_values, hist_arrays[1][:, i], deg=1)
        fit_functions_s.append(np.poly1d(coefs_s)) 
        fit_functions_b.append(np.poly1d(coefs_b))
        
    def fit_function(jes:float):
        # Cette fonction applique la transformation à un tableau nominal pour un JES donné
        # etape 1, caster
        diff_computed_s = np.array([poly(jes) for poly in fit_functions_s])
        diff_computed_b = np.array([poly(jes) for poly in fit_functions_b])
        # Applique la transformation à un tableau nominal pour un JES donné
        transformed_array_s =  diff_computed_s 
        transformed_array_b =  diff_computed_b
        return np.array(transformed_array_s), np.array(transformed_array_b) # je caste une deuxième fois parce qu'on st jamais trop sûr

    return fit_function

def met_fitter(model, train_set, n_bins:int = 100):

# 1. Définir une plage de valeurs MET autour de la valeur nominale (1.0)
    met_values = np.linspace(-10, 10, 30)
 
    # split train data and validation data
    data_set = train_set
 
    #  Obtenir l'histogramme des scores nominaux 
    syst_set = systematics(data_set, soft_met=0.0)
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
    
    # plot_given_hist(s_ref_data, b_ref_data)
    
    # Remplacer les 0 par 1 pour éviter toute division par zéro lors du ratio
    # nom_hist_safe = np.where(nom_hist == 0, 1, nom_hist)
    
    hist1, hist2 = [], []
    # Boucle sur les différentes valeurs de MET (Task 1)
 
    for t in met_values:
        syst_set = systematics(data_set, soft_met=t)
        alt_data = get_data(syst_set)[0]
        score = model.predict(alt_data) 
        labels_data = get_data(syst_set)[1]
        labels_array = labels_data.to_numpy()
        weights_data = get_data(syst_set)[2]
        weights_array = weights_data.to_numpy()
# 
        s_score, s_weight, b_score, b_weight = signal_bck(score, labels_array, weights_array)
# 
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
        coefs_s = np.polyfit(met_values, hist_arrays[0][:, i], deg=2)
        coefs_b = np.polyfit(met_values, hist_arrays[1][:, i], deg=2)
        fit_functions_s.append(np.poly1d(coefs_s)) 
        fit_functions_b.append(np.poly1d(coefs_b))
        
    def fit_function(met:float):
        # Cette fonction applique la transformation à un tableau nominal pour un MET donné
        # etape 1, caster
        diff_computed_s = np.array([poly(met) for poly in fit_functions_s])
        diff_computed_b = np.array([poly(met) for poly in fit_functions_b])
        # Applique la transformation à un tableau nominal pour un MET donné
        transformed_array_s = diff_computed_s 
        transformed_array_b = diff_computed_b
        return np.array(transformed_array_s), np.array(transformed_array_b) # je caste une deuxième fois parce qu'on st jamais trop sûr
    
    # def unique_test_met_fitter(transformateur, data_set, met_value):
    #     print(f"------ début du test de validation pour un valeur de met = {met_value}-----") 
    #     print("------génération de l'histogramme réel avec systematics-------")
 
    #     syst_reel = systematics(data_set, soft_met=met_value)
    #     data_reel, labels_reel, weights_reel = get_data(syst_reel)[3:]
 
    #     score_reel = model.predict(data_reel)
    #     labels_reel_arr = labels_reel.to_numpy()
    #     weights_reel_arr = weights_reel.to_numpy()
 
    #     s_sc_rl, s_w_rl, b_sc_rl, b_w_rl = signal_bck(score_reel, labels_reel_arr, weights_reel_arr)
 
    #     vrai_s_hist, _ = np.histogram(s_sc_rl, bins=n_bins, range=(0, 1), weights=s_w_rl, density=False)
    #     vrai_b_hist, _ = np.histogram(b_sc_rl, bins=n_bins, range=(0, 1), weights=b_w_rl, density=False)

    #     print("-----evaluation avec la fonction générée------")
    #     # on extrait les signal et background nominaux
    #     val_data, labels_val, weights_val = get_data(data_set)[3:]
    #     score_val = model.predict(val_data)
    #     labels_val_arr = labels_val.to_numpy()
    #     weights_val_arr = weights_val.to_numpy()
    #     s_sc_val, s_w_val, b_sc_val, b_w_val = signal_bck(score_val, labels_val_arr, weights_val_arr)
 
    #     s_hist_val, _ = np.histogram(s_sc_val, bins=n_bins, range=(0, 1), weights=s_w_val, density=False)
    #     b_hist_val, _ = np.histogram(b_sc_val, bins=n_bins, range=(0, 1), weights=b_w_val, density=False)
    #     computed_s_hist, computed_b_hist = transformateur((s_hist_val, b_hist_val), met_value)

    #     print("-----calcul de l'erreur quadratique moyenne-----")
    #     # calcul du MSE sur différentes valeurs
    #     mse_s, mse_b = [], []
    #     mse_s = np.mean(((computed_s_hist - vrai_s_hist)**2))
    #     mse_b = np.mean((computed_b_hist - vrai_b_hist)**2)
    #     print(f"Erreur quadratique moyenne (Signal) : {mse_s:.4f}")
    #     print(f"Erreur quadratique moyenne (Bkg)    : {mse_b:.4f}") 
    #     return mse_s, mse_b

    # def test_met_fitter(fit_function, data_set):
    #     L_met = np.linspace(-10, 10, 30)
    #     L_mse_b = []
    #     L_mse_s = []
    #     for met in L_met:
    #         mse_s, mse_b = unique_test_met_fitter(fit_function, data_set, met)
    #         L_mse_s.append(mse_s)
    #         L_mse_b.append(mse_b)
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(L_met, L_mse_b, color="#175a8a", label="MSE pour le bkg")
    #     plt.plot(L_met, L_mse_s, color="#e40505d0", label="MSE pour le signal")
    #     plt.title("visualisation de l'erreur commise avec la fonction")
    #     plt.legend()
    #     plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    #     plt.show()
    
    # # test_met_fitter(fit_function, data_set)

    return fit_function

def bkg_scale_fitter(model, train_set, n_bins:int = 100):
    """
    Task 1 : Analysis bkg_norm Uncertainty
    1. Loop over different values of jes and store the score
    2. Make a histogram of the score

    Task 2 : Fit the histogram
    1. Write a function to loop over different values of JES and histogram and make fit function for each bin in the histogram
    2. store the fit functions in an array
    3. return the fit functions
    """
    # 1. Définir une plage de valeurs JES autour de la valeur nominale (1.0)
    bkg_scale_values = np.linspace(0.9, 1.1, 30)
 
    # split train data and validation data
    data_set = train_set
 
    #  Obtenir l'histogramme des scores nominaux 
    syst_set = systematics(data_set, jes=1)
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
    
    # plot_given_hist(s_ref_data, b_ref_data)
    
    # Remplacer les 0 par 1 pour éviter toute division par zéro lors du ratio
    # nom_hist_safe = np.where(nom_hist == 0, 1, nom_hist)
    
    hist1, hist2 = [], []
    # Boucle sur les différentes valeurs de BCG_NORM (Task 1)
 
    for t in bkg_scale_values:
        syst_set = systematics(data_set, bkg_scale=t)
        alt_data = get_data(syst_set)[0]
        score = model.predict(alt_data) 
        labels_data = get_data(syst_set)[1]
        labels_array = labels_data.to_numpy()
        weights_data = get_data(syst_set)[2]
        weights_array = weights_data.to_numpy()
# 
        s_score, s_weight, b_score, b_weight = signal_bck(score, labels_array, weights_array)
# 
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
        coefs_s = np.polyfit(bkg_scale_values, hist_arrays[0][:, i], deg=1)
        coefs_b = np.polyfit(bkg_scale_values, hist_arrays[1][:, i], deg=1)
        fit_functions_s.append(np.poly1d(coefs_s)) 
        fit_functions_b.append(np.poly1d(coefs_b))
        
    def fit_function(bkg_scale:float):
        # Cette fonction applique la transformation à un tableau nominal pour un JES donné
        # etape 1, caster
        diff_computed_s = np.array([poly(bkg_scale) for poly in fit_functions_s])
        diff_computed_b = np.array([poly(bkg_scale) for poly in fit_functions_b])
        # Applique la transformation à un tableau nominal pour un bkg_scale donné
        transformed_array_s =  diff_computed_s 
        transformed_array_b =  diff_computed_b
        return np.array(transformed_array_s), np.array(transformed_array_b) # je caste une deuxième fois parce qu'on st jamais trop sûr

    return fit_function





def global_fit_function(model, train_set, n_bins=100):
    transf_1 = tes_fitter(model, train_set, n_bins)
    transf_2 = jes_fitter(model, train_set, n_bins)
    transf_3 = met_fitter(model, train_set, n_bins)
    transf_4 = bkg_scale_fitter(model, train_set, n_bins)

    data_set = train_set
    NOMINAL_PARAMS = {
        "tes":1.0,
        "jes":1.0,
        "bkg_scale":1.0,
        "soft_met":0.0
    }
 
    #  Obtenir l'histogramme des scores nominaux 
    syst_set = systematics(data_set, **NOMINAL_PARAMS)
    alt_data = get_data(syst_set)[0]
 
    # le score nominal
 
    score = model.predict(alt_data) 
 
    labels_data = get_data(syst_set)[1]
    labels_array = labels_data.to_numpy()
    weights_data = get_data(syst_set)[2]
    weights_array = weights_data.to_numpy()
 
    s_score, s_weight, b_score, b_weight = signal_bck(score, labels_array, weights_array)
 
    s_histogram,_ = np.histogram(
        s_score, bins=n_bins, range=(0, 1), weights=s_weight, density=False
    )
    b_histogram,_ = np.histogram(
        b_score, bins=n_bins, range=(0, 1), weights=b_weight, density=False
    )
 
    s_ref_data = s_histogram
    b_ref_data = b_histogram

    def fit_function(array, theta):
        return (s_ref_data + transf_1(theta[0])[0] + transf_2(theta[1])[0] + transf_3(theta[2])[0] + transf_4(theta[3])) , b_ref_data + transf_1(theta[0])[1] + transf_2(theta[1])[1] + transf_3(theta[2])[1] + transf_4(theta[3])[1]

    return fit_function


######################################################################## version avec dico

# Configuration des valeurs nominales par défaut
NOMINALS = {
    "tes": 1.0,
    "jes": 1.0,
    "bkg_scale": 1.0,
    "soft_met": 0.0  
}

SIGMA_SHIFTS = {
    "tes": {"plus": 1.03, "minus": 0.97},
    "jes": {"plus": 1.03, "minus": 0.97},
    "bnorm": {"plus": 1.05, "minus": 0.95}, # Correspond à bkg_scale (5%)
    "smet": {"plus": 3.0, "minus": 0}     # Correspond à soft_met (3 GeV)
}

# Correspondance entre le nom du prof (clé de sortie) et le nom de l'argument du code

PARAM_MAPPING = {
    "tes": "tes",
    "jes": "jes",
    "bnorm": "bkg_scale",
    "smet": "soft_met"
}

# nombre de bins 
num_bins = 6



def generer_saved_info(model, training_dict, num_bins=num_bins):
    """
    Calcule les deltas d'événements par bin à +1σ et -1σ pour chaque systématique
    et retourne le dictionnaire structuré pour le groupe STAT.
    """
    print("\n==================================================")
    print("GÉNÉRATION DU DICTIONNAIRE 'saved_info' POUR STAT")
    print("==================================================")
   
    # 1. Calcul de la référence nominale commune
    dict_nom_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
    set_nom = systematics(dict_nom_input, **NOMINALS)
   
    scores_nom = np.array(model.predict(set_nom["data"])).ravel()
    labels_nom = np.array(set_nom["labels"]).ravel()
    weights_nom = np.array(set_nom["weights"]).ravel()

    is_sig_nom, is_bkg_nom = (labels_nom == 1.0), (labels_nom == 0.0)
    bins_fixes = np.linspace(np.min(scores_nom), np.max(scores_nom), num_bins + 1)
   
    nom_s_counts, _ = np.histogram(scores_nom[is_sig_nom], bins=bins_fixes, weights=weights_nom[is_sig_nom])
    nom_b_counts, _ = np.histogram(scores_nom[is_bkg_nom], bins=bins_fixes, weights=weights_nom[is_bkg_nom])

    # Structure initiale requise par le professeur
    saved_info = {
        "S": {},
        "B": {}
    }
    saved_info["S"]["nominal"] = nom_s_counts.tolist()
    saved_info["B"]["nominal"] = nom_b_counts.tolist()

    # Helper interne pour obtenir les deltas d'une configuration spécifique
    def obtenir_deltas(syst_key, direction):
        val_shift = SIGMA_SHIFTS[syst_key][direction]
        arg_name = PARAM_MAPPING[syst_key]
       
        # Recréation d'un dictionnaire propre pour l'itération
        dict_loop_input = {k: v.copy() if hasattr(v, 'copy') else v for k, v in training_dict.items()}
       
        # Configuration des arguments
        syst_kwargs = NOMINALS.copy()
        if syst_key in ["tes", "jes"]:
            syst_kwargs[arg_name] = val_shift
           
        set_shift = systematics(dict_loop_input, **syst_kwargs)
       
        scores = np.array(model.predict(set_shift["data"])).ravel()
        labels = np.array(set_shift["labels"]).ravel()
        weights = np.array(set_shift["weights"]).ravel()
       
        is_sig, is_bkg = (labels == 1.0), (labels == 0.0)
       
        # Sécurité manuelle active pour le bnorm (bkg_scale)
        if syst_key == "bnorm":
            weights[is_bkg] = weights[is_bkg] * val_shift
           
        curr_s_counts, _ = np.histogram(scores[is_sig], bins=bins_fixes, weights=weights[is_sig])
        curr_b_counts, _ = np.histogram(scores[is_bkg], bins=bins_fixes, weights=weights[is_bkg])
       
        return (curr_s_counts - nom_s_counts), (curr_b_counts - nom_b_counts)

    # 2. Boucle sur chaque systématique pour extraire les paires [+1σ, -1σ]
    for syst in SIGMA_SHIFTS.keys():
        print(f"Calcul des dérives ±1σ pour : {syst.upper()}...")
       
        delta_plus_s, delta_plus_b = obtenir_deltas(syst, "plus")
        delta_minus_s, delta_minus_b = obtenir_deltas(syst, "minus")
       
        # Structuration sous forme de liste de paires [Delta(+1σ), Delta(-1σ)] pour chaque bin
        saved_info["S"][syst] = [[delta_plus_s[i], delta_minus_s[i]] for i in range(num_bins)]
        saved_info["B"][syst] = [[delta_plus_b[i], delta_minus_b[i]] for i in range(num_bins)]

    print("Génération terminée avec succès.")
    return saved_info

def Delta_gamma_universel(bin_i, sys_val, syst_name, saved_info, classe="S",SIGMA_SHIFTS=SIGMA_SHIFTS,NOMINALS=NOMINALS):
    """
    Fonction universelle pour l'équipe STAT.
    - bin_i : numéro du bin (ex: 0 à 5)
    - sys_val : la valeur demandée par la vraisemblance (ex: 1.015)
    - syst_name : "tes", "jes", "bnorm", ou "smet"
    - saved_info : le dictionnaire que vous avez généré
    - classe : "S" (Signal) ou "B" (Background)
    """
    # Récupération des bornes selon la systématique demandée
    if syst_name in ["tes", "jes"]:
        sys_max, sys_min, sys_nom = SIGMA_SHIFTS[syst_name]["plus"], SIGMA_SHIFTS[syst_name]["minus"], NOMINALS[syst_name]
    elif syst_name == "bnorm":
        sys_max, sys_min, sys_nom = SIGMA_SHIFTS[syst_name]["plus"], SIGMA_SHIFTS[syst_name]["minus"], NOMINALS["bkg_scale"]
    elif syst_name == "smet":
        sys_max, sys_min, sys_nom = SIGMA_SHIFTS[syst_name]["plus"], SIGMA_SHIFTS[syst_name]["minus"], NOMINALS["soft_met"]

    # Lecture du dictionnaire
    delta_plus = saved_info[classe][syst_name][bin_i][0]
    delta_minus = saved_info[classe][syst_name][bin_i][1]

    # Mathématiques
    pente = (delta_plus - delta_minus) / (sys_max - sys_min)
    cste = -pente * sys_nom
    
    return pente * sys_val + cste



def N_total_bin(bin_i, tes, jes, bnorm, smet, saved_info, classe="S"):
    """
    Calcule le nombre total d'événements (S ou B) dans un bin donné
    en additionnant la base nominale et toutes les dérives systématiques.
    
    Formule : N_i = N_nominale_i + sum(Delta_i)
    """
    # 1. La base nominale (le point de départ à 1.0 partout)
    n_nominale = saved_info[classe]["nominal"][bin_i]
    
    # 2. Calcul de chaque dérive individuelle via votre fonction universelle
    # On additionne les deltas au nominal
    delta_tes = Delta_gamma_universel(bin_i, tes, "tes", saved_info, classe)
    delta_jes = Delta_gamma_universel(bin_i, jes, "jes", saved_info, classe)
    delta_bnorm = Delta_gamma_universel(bin_i, bnorm, "bnorm", saved_info, classe)
    delta_smet = Delta_gamma_universel(bin_i, smet, "smet", saved_info, classe)
    
    # 3. Somme totale
    n_total = n_nominale + delta_tes + delta_jes + delta_bnorm + delta_smet
    
    # Sécurité physique : un nombre d'événements ne peut pas être négatif
    return max(0, n_total)

def calcul_prediction_totale(parametres, saved_info,num_bins=num_bins):
    # parametres = [tes, jes, bnorm, smet]
    pred_S = [N_total_bin(i, *parametres, saved_info, classe="S") for i in range(num_bins)]
    pred_B = [N_total_bin(i, *parametres, saved_info, classe="B") for i in range(num_bins)]
    return pred_S, pred_B


def param_fitter_S(model, training_dict, num_bins=num_bins):
    """ model : votre modèle entrainé (exemple my_model.model )
        training_dict : votre dictionnaire d'entrainement (exemple my_model.training_set)
        num_bins : le nombre de bins que vous avez choisi pour votre histogramme
    """
    saved_info = generer_saved_info(model, training_dict, num_bins=num_bins)


    def obtenir_prediction_tous_bins_S(tes=1.0, jes=1.0, bnorm=1.0, smet=0.0):
        """
        Calcule la prédiction finale (Nominal + Deltas) pour l'intégralité des bins
        pour le Signal et le Background.

        Args:
            saved_info (dict): Le dictionnaire généré par generer_saved_info
            tes, jes, bnorm, smet (float): Les valeurs courantes des paramètres systématiques (parametre)
            num_bins (int): Le nombre de bins total (par défaut configuré au début de ton script)

        Returns:
            tuple: (liste_S, liste_B) contenant les prédictions pour chaque bin.
        """
        pred_S = []


        # On parcourt tous les bins un par un
        for i in range(num_bins):
            # Calcul pour le Signal dans le bin i
            n_s = N_total_bin(bin_i=i, tes=tes, jes=jes, bnorm=bnorm, smet=smet, 
                              saved_info=saved_info, classe="S")
            pred_S.append(n_s)

        return pred_S
    
    return obtenir_prediction_tous_bins_S

def param_fitter_B(model, training_dict, num_bins=num_bins):
    """ model : votre modèle entrainé (exemple my_model.model )
        training_dict : votre dictionnaire d'entrainement (exemple my_model.training_set)
        num_bins : le nombre de bins que vous avez choisi pour votre histogramme
    """
    saved_info = generer_saved_info(model, training_dict, num_bins=num_bins)


    def obtenir_prediction_tous_bins_B(tes=1.0, jes=1.0, bnorm=1.0, smet=0.0):
        """
        Calcule la prédiction finale (Nominal + Deltas) pour l'intégralité des bins
        pour le Signal et le Background.

        Args:
            saved_info (dict): Le dictionnaire généré par generer_saved_info
            tes, jes, bnorm, smet (float): Les valeurs courantes des paramètres systématiques (parametre)
            num_bins (int): Le nombre de bins total (par défaut configuré au début de ton script)

        Returns:
            tuple: (liste_S, liste_B) contenant les prédictions pour chaque bin.
        """
        pred_B = []


        # On parcourt tous les bins un par un
        for i in range(num_bins):
            # Calcul pour le Signal dans le bin i
            n_b = N_total_bin(bin_i=i, tes=tes, jes=jes, bnorm=bnorm, smet=smet, 
                              saved_info=saved_info, classe="B")
            pred_B.append(n_b)

        return pred_B
    
    return obtenir_prediction_tous_bins_B
    
