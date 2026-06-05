import sys
from pathlib import Path
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import chisquare
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from HiggsML.systematics import systematics

#Geoffroy--------------------------------------------------------------------------------------------------------------------------------------------------------------

from HiggsML.datasets import download_dataset, Data
# Ensure the package containing HiggsML is on sys.path so we can import HiggsML.datasets
# black_swan_pkg_path = r'c:\Users\geoff\Documents\Centrale\cours_centrale\ST4\EI\black_swan_pkg'
# if black_swan_pkg_path not in sys.path:
#     sys.path.insert(0, black_swan_pkg_path)
# from HiggsML.datasets import download_dataset

#Youri--------------------------------------------------------------------------------------------------------------------------------------------------------------

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
# from datasets import download_dataset

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

# Download dataset later (we change working dir first); keep this line commented to avoid duplicate downloads
# data = download_dataset("blackSwan_data")

# # Changer vers le dossier du script pour télécharger les données localement
# os.chdir(os.path.dirname(__file__))

# data = download_dataset(
#     "blackSwan_data"
# )
# data.load_train_set()
# data_set = data.get_train_set()
# target = data_set["labels"]
# data_cleaned = data_set.replace(-25, np.nan)

def data_jet_one(data): 
    # We keep all the rows where the value of "PRI_n_jets" is 1 but we keep all the columns, it's just take out some data
    data_jet_one = data[data["PRI_n_jets"] == 1]
    return data_jet_one

def data_jet_zero(data):
    # We keep all the rows where the value of "PRI_n_jets" is 0 
    data_jet_zero = data[data["PRI_n_jets"] == 0]
    return data_jet_zero

def data_jet_two(data):
    # We keep all the rows where the value of "PRI_n_jets" is 2 
    data_jet_two = data[data["PRI_n_jets"] == 2]
    return data_jet_two

def feature_corrilations(data, show=False):
    
    
    # Validate input
    if data is None:
        raise ValueError("No data provided to feature_corrilations: train set is empty or failed to load")

    # we Handle the -25 sentinel value
    data_cleaned = data.replace(-25, np.nan)
    
    #we create vectors with the physical values, we don't take the colums labels"
    feature_cols = [col for col in data_cleaned.columns if col.startswith('PRI_') or col.startswith('DER_')]
    
    #calcul of the matrix of correlation#
    corr_matrix = data_cleaned[feature_cols].corr()
    
    # Debug: print np.shape to ensure it's 28x28 (or show actual shape)
    print(f"Number of features: {len(feature_cols)}")
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    # Plot: wider but less tall figure to reduce vertical space
    plt.figure(figsize=(16, 10))
    annot_kws = {"size": 6}
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='vlag',
        center=0,
        square=False,
        xticklabels=True,
        yticklabels=True,
        annot_kws=annot_kws,
        cbar_kws={"shrink": 0.6},
    )
    # Reduce tick label font size (feature titles) and rotate x labels for readability
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=6, rotation=0)
    plt.title("Correlation matrix of features", fontsize=12)
    plt.tight_layout()
    if show:
        plt.show()
    return corr_matrix


def feature_signal_background_correlations(data, show=False):
    sns.set_theme(style="whitegrid")
    rows = {}
    num_cols = data.shape[1]
    feature_names = list(data.columns)

    for i in range(16):
        corr_matrix = data.iloc[:, [i, 18]].corr()
        rows[feature_names[i]] = corr_matrix.iloc[0, 1]

    for i in range(19, num_cols):
        corr_matrix = data.iloc[:, [18, i]].corr()
        rows[feature_names[i]] = corr_matrix.iloc[0, 1]

    dfplot = pd.DataFrame({"Correlation with signal/background": list(rows.values())}, index=list(rows.keys()))

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        dfplot,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8},
        cbar_kws={"label": "Correlation", "shrink": 0.9},
    )
    plt.title("Correlation of each feature with signal/background", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if show:
        plt.show()

    #print(dfplot)
    return dfplot




features_all = [
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_had_pt",
    "PRI_had_eta",
    "PRI_had_phi",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_n_jets",
    "PRI_jet_all_pt",
    "PRI_met",
    "PRI_met_phi",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_had_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_had",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
]

def systematic_dependence(data, show=False, features=features_all):
    #downloading the observed dataset without the bias
    data.load_train_set()
    dataframe_original = data.get_train_set()
    weights_original = dataframe_original["weights"]
    labels_original = dataframe_original["labels"]

    #Configuring the systematic bias
    biases = {
            "tes":{"m":1.0,"sigma":0.01},
            "jes":{"m":1.0,"sigma":0.01},
            "soft_met":{"m":1.0,"sigma":0.01},
            "ttbar_scale":{"m":1.0,"sigma":0.01},
            "diboson_scale":{"m":1.0,"sigma":0.01},
            "bkg_scale":{"m":1.0,"sigma":0.01}}
    
    #Computing the quadratic difference between biased and original datframe
    def quad_err(exp,obs):
        keep_ind = exp > 0
        return np.sum((obs[keep_ind]-exp[keep_ind])**2/exp[keep_ind])
    
    #Applying the bias and computing the quadratic error for each feature
    quad_errs = {}
    for feat in features:
        quad_errs[f"{feat} (signal)"] = {}
        quad_errs[f"{feat} (background)"] = {}
    for b_name, b_params in biases.items():
        for err in ["+sig","-sig"]:
            if b_name == "soft_MET" and err == "-sig":
                continue
            args = {}
            if err == "+sig":
                val = b_params["m"] + b_params["sigma"]
            else:
                val = b_params["m"] - b_params["sigma"]
            args[b_name] = val
            biased_result = systematics({"data": dataframe_original.copy(), "weights": weights_original.copy(), "labels": labels_original.copy()}, **args, dopostprocess=False)
            dataframe_biased = biased_result["data"]
            weights_biased = biased_result["weights"]
            labels_biased = biased_result["labels"]

            for feat in features:
                if feat not in dataframe_original.columns or feat not in dataframe_biased.columns:
                    continue

                vals = pd.concat([dataframe_original[feat], dataframe_biased[feat]]).dropna()
                x_min, x_max = np.percentile(vals, [0.5, 99.5])
                bins = np.linspace(x_min, x_max, 80)

                signal_original = np.histogram(dataframe_original[feat][labels_original == 1], bins=bins, weights=weights_original[labels_original == 1])[0]
                signal_biased = np.histogram(dataframe_biased[feat][labels_biased == 1], bins=bins, weights=weights_biased[labels_biased == 1])[0]
                background_original = np.histogram(dataframe_original[feat][labels_original == 0], bins=bins, weights=weights_original[labels_original == 0])[0]
                background_biased = np.histogram(dataframe_biased[feat][labels_biased == 0], bins=bins, weights=weights_biased[labels_biased == 0])[0]

                errs_sig = quad_err(signal_biased,signal_original)
                errs_bkg = quad_err(background_biased,background_original)
                quad_errs[f"{feat} (signal)"][f"{b_name} ({err})"] = errs_sig
                quad_errs[f"{feat} (background)"][f"{b_name} ({err})"] = errs_bkg
            
            # Plotting the distributions for the current bias and feature
                if show :
                    # Configuration du style
                    plt.figure(figsize=(7, 4.5))
                    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)

                    # Configuration des 4 histogrammes à tracer
                    plots_config = {
                        "orig_sig": {"df": dataframe_original, "labels": labels_original, "target": 1, "w": weights_original
            , "color": "red", "histtype": "bar", "alpha": 0.15, "lbl": "Signal (original)"},
                        "bias_sig": {"df": dataframe_biased, "labels": labels_biased, "target": 1, "w": weights_biased, "color": "darkred", "histtype": "step", "alpha": 1.0, "lbl": f"Signal ({b_name} {err})"},
                        "orig_bkg": {"df": dataframe_original, "labels": labels_original, "target": 0, "w": weights_original
            , "color": "blue", "histtype": "bar", "alpha": 0.15, "lbl": "Background (original)"},
                        "bias_bkg": {"df": dataframe_biased, "labels": labels_biased, "target": 0, "w": weights_biased, "color": "darkblue", "histtype": "step", "alpha": 1.0, "lbl": f"Background ({b_name} {err})"}
                    }

                    # Dictionnaire pour stocker les hauteurs récupérées
                    heights = {}

                    # Boucle pour tracer les 4 histogrammes proprement
                    for key, cfg in plots_config.items():
                        mask = cfg["labels"] == cfg["target"]
                        h, _, _ = plt.hist(
                            cfg["df"][feat][mask],
                            bins=bins,
                            weights=cfg["w"][mask],
                            label=cfg["lbl"],
                            color=cfg["color"],
                            histtype=cfg["histtype"],
                            alpha=cfg["alpha"],
                            density=True,
                            linewidth=1.5 if cfg["histtype"] == "step" else 1.0,
                            zorder=2
                        )
                        heights[key] = h

                    # Remplissage des zones d'écart
                    bin_centers = (bins[:-1] + bins[1:]) / 2

                    # Zone Signal
                    plt.fill_between(
                        bin_centers, heights["orig_sig"], heights["bias_sig"],
                        where=(heights["orig_sig"] != heights["bias_sig"]),
                        color="red", alpha=0.2, step="mid", zorder=1,
                        label="Syst. Uncertainty (Signal)"
                    )

                    # Zone Background
                    plt.fill_between(
                        bin_centers, heights["orig_bkg"], heights["bias_bkg"],
                        where=(heights["orig_bkg"] != heights["bias_bkg"]),
                        color="blue", alpha=0.2, step="mid", zorder=1,
                        label="Syst. Uncertainty (Background)"
                    )

                    # Habillage
                    plt.xlim(x_min, x_max)
                    plt.title(f"Feature: {feat} | Systematic Impact ({b_name} {err})", fontsize=11, fontweight='bold', pad=10)
                    plt.xlabel(feat, fontsize=10)
                    plt.ylabel("Probability Density", fontsize=10)
                    
                    plt.legend(fontsize="small", frameon=True, facecolor="white", edgecolor="none", loc="upper right")
                    
                    plt.tight_layout()
                    plt.show()
                    
                    print(
                        f"[{feat}] {b_name} {err}\n χ² (signal): {errs_sig:.2f}, χ² (background): {errs_bkg:.2f}"
                    )



    if show:
        for feat in features:
            if feat not in dataframe_original.columns:
                continue
            plt.figure(figsize=(12, 6))
            plt.bar(quad_errs[f"{feat} (signal)"].keys(), quad_errs[f"{feat} (signal)"].values(), label="Signal", alpha=0.7)
            plt.bar(quad_errs[f"{feat} (background)"].keys(), quad_errs[f"{feat} (background)"].values(), label="Background", alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Quadratic Error")
            plt.title(f"Systematic Dependence for {feat}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    quad_df = pd.DataFrame.from_dict(quad_errs, orient="index")
    
    return quad_df


def Score_systematics (data,features=features_all ) :
    table_chi2=systematic_dependence(data, show=True, features=features)
    score = table_chi2.sum(axis=1).to_dict()
    score_total = {}
    for feat, chi2 in score.items():
        base_feature = feat.replace(" (signal)", "").replace(" (background)", "")
        score_total[base_feature] = score_total.get(base_feature, 0) + chi2
    plt.figure(figsize=(10, 6))
    plt.bar(score_total.keys(), score_total.values(), color="red")
    
    # Add labels and title
    plt.xlabel("Features")
    plt.ylabel("Chi-2")
    plt.title("Total Impact of bias Chi-2 for each feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    
    return sorted(score_total.keys(), key=lambda k: score_total[k], reverse=True)

def minimal_dependent_features(data):
    return data.columns


# if __name__ == "__main__":
#     # When run as a script, show the correlation matrix for the training set
#     feature_corrilations(data_set)




def value_signal(data_set, lst):
    
    E=0
    for i in range(len(lst)): 
        E += (matrix_line.iloc[0,lst[i]])**2
    return E

def value_correlation(data_set, lst):
    E=0
    for i in range(len(lst)): 
        for j in range(len(lst)):
            if i>j:
                E += ((matrix_square).iloc[lst[i],lst[j]])**2
    return E

def value(data_set, lst):
    n=len(lst)
    return 3*value_signal(data_set, lst) - (2/(n-1))*value_correlation(data_set, lst)
        
    
def best_features_set(data, n):
    Score = -np.inf
    indice_max = 0
    num_cols = matrix_square.shape[1]
    les_entiers = range(num_cols)
    combinaisons = list(itertools.combinations(les_entiers, n))
    for i in range (len(combinaisons)):
        if value(data, combinaisons[i]) > Score:
            Score = value(data, combinaisons[i])
            indice_max = i

    return Score, combinaisons[indice_max]





def engineering_angles(data):
    df=data.copy()
    
    # We take the colums which ended with "phi"
    phi_cols = [col for col in df.columns if col.endswith('_phi')]
    
    for col in phi_cols:
        # We create new features by taking the sine and cosine of the angle features
        df[f'{col}_sin'] = np.sin(df[col])
        df[f'{col}_cos'] = np.cos(df[col])
    return df

def to_cartesian(data):
    particle_prefixes = ['PRI_lep', 'PRI_had', 'PRI_jet_leading', 'PRI_jet_subleading']
    df = data.copy()

    for particle_prefix in particle_prefixes:
        pt = df[f'{particle_prefix}_pt']
        phi = df[f'{particle_prefix}_phi']
        eta = df[f'{particle_prefix}_eta']

        df[f'{particle_prefix}_px'] = pt * np.cos(phi)
        df[f'{particle_prefix}_py'] = pt * np.sin(phi)
        df[f'{particle_prefix}_pz'] = pt * np.sinh(eta)

    return df


def conversion_numbers_into_names(score, indices, data_frame):
    feature_cols = [col for col in data_frame.columns if col.startswith('PRI_') or col.startswith('DER_')]
    selected_features = [feature_cols[i] for i in indices]
    return score, selected_features


systematic_dependence(Data("c:/Users/noahl/Documents/EI_PP"),True)
# data = Data("c:/Users/noahl/Documents/EI_PP")
# data.load_train_set()
# dfmod = data.get_train_set()
# df_lowbias = dfmod.drop(columns=["PRI_had_pt", "PRI_jet_subleading_pt", "DER_mass_vis", "DER_pt_ratio_lep_had"])
# df_lowcorr = dfmod.drop(columns=["PRI_lep_phi","PRI_had_phi","PRI_jet_subleading_phi","PRI_met_phi"])
# Les matrices sont calculées sans affichage graphique ici.
# Si vous voulez les visualiser, appelez les fonctions avec show=True ou décommentez les lignes ci-dessous.
# matrix_line = feature_signal_background_correlations(data_set, show=False)
# matrix_square = feature_corrilations(data_set, show=False)
# feature_signal_background_correlations(data_set, show=True)
# feature_corrilations(data_jet_one(data_set))
# feature_corrilations(data_jet_zero(data_set))
# feature_corrilations(data_jet_two(data_set))
# print(data_set.columns)
# print(data_set["detailed_labels"])
# print(data_set["weights"])
# print(data_set["labels"])