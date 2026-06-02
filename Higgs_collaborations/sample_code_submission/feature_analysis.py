from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import chisquare
from HiggsML.datasets import download_dataset, Data
from HiggsML.systematics import systematics
import numpy as np
from matplotlib.font_manager import FontProperties

def feature_corrilations(data):
    pass

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
    

def minimal_dependent_features(data):
    return data.columns

systematic_dependence(Data("c:/Users/noahl/Documents/EI_PP"), show=True)