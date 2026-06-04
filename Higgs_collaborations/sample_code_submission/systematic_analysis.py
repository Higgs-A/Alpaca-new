import numpy as np
import matplotlib.pyplot as plt
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


    def unique_test_tes_fitter(transformateur, data_set, tes_value):
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
        return mse_s, mse_b

    def test_tes_fitter(fit_function, data_set):
        L_tes = np.linspace(0.9, 1.1, 50)
        L_mse_b = []
        L_mse_s = []
        for tes in L_tes:
            mse_s, mse_b = unique_test_tes_fitter(fit_function, data_set, tes)
            L_mse_s.append(mse_s)
            L_mse_b.append(mse_b)
        plt.figure(figsize=(8, 5))
        plt.plot(L_tes, L_mse_b, color="#175a8a", label="MSE pour le bkg")
        plt.plot(L_tes, L_mse_s, color="#e40505d0", label="MSE pour le signal")
        plt.title("visualisation de l'erreur commise avec la fonction")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.show()
    
    test_tes_fitter(fit_function, data_set)




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
    jes_values = np.linspace(0.9, 1.1, 30)
 
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
        
    def fit_function(array, jes:float):
        # Cette fonction applique la transformation à un tableau nominal pour un JES donné
        # etape 1, caster
        my_array_s, my_array_b = np.array(array[0]), np.array(array[1])
        diff_computed_s = np.array([poly(jes) for poly in fit_functions_s])
        diff_computed_b = np.array([poly(jes) for poly in fit_functions_b])
        # Applique la transformation à un tableau nominal pour un JES donné
        transformed_array_s = my_array_s  + diff_computed_s 
        transformed_array_b = my_array_b  + diff_computed_b
        return np.array(transformed_array_s), np.array(transformed_array_b) # je caste une deuxième fois parce qu'on st jamais trop sûr
    
    def test_jes_fitter(transformateur, data_set, jes_value):
        print(f"------ début du test de validation pour un valeur de jes = {jes_value}-----") 
        print("------génération de l'histogramme réel avec systematics-------")
 
        syst_reel = systematics(data_set, jes=jes_value)
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
        computed_s_hist, computed_b_hist = transformateur((s_hist_val, b_hist_val), jes_value)

        print("-----calcul de l'erreur quadratique moyenne-----")
        # calcul du MSE sur différentes valeurs
        mse_s, mse_b = [], []
        mse_s = np.mean(((computed_s_hist - vrai_s_hist)**2))
        mse_b = np.mean((computed_b_hist - vrai_b_hist)**2)
        print(f"Erreur quadratique moyenne (Signal) : {mse_s:.4f}")
        print(f"Erreur quadratique moyenne (Bkg)    : {mse_b:.4f}") 

    for jes in [0.98, 1.015, 1.13]:
        test_jes_fitter(fit_function, data_set, jes)




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
        
    def fit_function(array, met:float):
        # Cette fonction applique la transformation à un tableau nominal pour un MET donné
        # etape 1, caster
        my_array_s, my_array_b = np.array(array[0]), np.array(array[1])
        diff_computed_s = np.array([poly(met) for poly in fit_functions_s])
        diff_computed_b = np.array([poly(met) for poly in fit_functions_b])
        # Applique la transformation à un tableau nominal pour un MET donné
        transformed_array_s = my_array_s  + diff_computed_s 
        transformed_array_b = my_array_b  + diff_computed_b
        return np.array(transformed_array_s), np.array(transformed_array_b) # je caste une deuxième fois parce qu'on st jamais trop sûr
    
    def unique_test_met_fitter(transformateur, data_set, met_value):
        print(f"------ début du test de validation pour un valeur de met = {met_value}-----") 
        print("------génération de l'histogramme réel avec systematics-------")
 
        syst_reel = systematics(data_set, soft_met=met_value)
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
        computed_s_hist, computed_b_hist = transformateur((s_hist_val, b_hist_val), met_value)

        print("-----calcul de l'erreur quadratique moyenne-----")
        # calcul du MSE sur différentes valeurs
        mse_s, mse_b = [], []
        mse_s = np.mean(((computed_s_hist - vrai_s_hist)**2))
        mse_b = np.mean((computed_b_hist - vrai_b_hist)**2)
        print(f"Erreur quadratique moyenne (Signal) : {mse_s:.4f}")
        print(f"Erreur quadratique moyenne (Bkg)    : {mse_b:.4f}") 
        return mse_s, mse_b

    def test_met_fitter(fit_function, data_set):
        L_met = np.linspace(-10, 10, 30)
        L_mse_b = []
        L_mse_s = []
        for met in L_met:
            mse_s, mse_b = unique_test_met_fitter(fit_function, data_set, met)
            L_mse_s.append(mse_s)
            L_mse_b.append(mse_b)
        plt.figure(figsize=(8, 5))
        plt.plot(L_met, L_mse_b, color="#175a8a", label="MSE pour le bkg")
        plt.plot(L_met, L_mse_s, color="#e40505d0", label="MSE pour le signal")
        plt.title("visualisation de l'erreur commise avec la fonction")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.show()
    
    test_met_fitter(fit_function, data_set)


    return fit_function

################################################################ by beta



