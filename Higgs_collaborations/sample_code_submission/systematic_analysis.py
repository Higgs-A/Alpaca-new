import numpy as np
import matplotlib.pyplot as plt
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
        #  Dummy fit function, replace with actual fitting procedure
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
    #Dummy fit function, replace with actual fitting procedure
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
          #/!\ Vérifie le nom de l'argument: soft_met ou met_unclustered
         syst_set = systematics(eval_set, soft_met=met) 
         score = model.predict(syst_set["data"])
        
         hist, _ = np.histogram(score, bins=bins, range=(0, 1))
         histograms.append(hist)
        
     histograms = np.array(histograms)
    
      #2. On "fit" un polynôme de degré 2 pour chaque bin de l'histogramme
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
         return np.clip(new_hist, 0, None)  #Empêche d'avoir un nombre de particules négatif

     return fit_function



    
def bkg_fitter(model, eval_set, bkg_variations=np.array([0.95, 1.00, 1.05])):
     """
     Analyse de l'incertitude sur la normalisation du bruit de fond (BKGnorm).
     Variation relative (0.95 = -5%, 1.05 = +5%).
     """
     histograms = []
     bins = 100
    
     for bkg in bkg_variations:
        #  /!\ Vérifie le nom de l'argument: bkg_scale, bkg_norm, etc.
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
    
