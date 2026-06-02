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
    

# Plot

if show :
    # Configuration du style
    plt.figure(figsize=(7, 4.5))
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)

    # Configuration des 4 histogrammes à tracer
    plots_config = {
        "orig_sig": {"df": original_df, "labels": original_labels, "target": 1, "w": original_weights, "color": "red", "histtype": "bar", "alpha": 0.15, "lbl": "Signal (original)"},
        "bias_sig": {"df": biased_df, "labels": biased_labels, "target": 1, "w": biased_weights, "color": "darkred", "histtype": "step", "alpha": 1.0, "lbl": f"Signal ({bias_name} {shift})"},
        "orig_bkg": {"df": original_df, "labels": original_labels, "target": 0, "w": original_weights, "color": "blue", "histtype": "bar", "alpha": 0.15, "lbl": "Background (original)"},
        "bias_bkg": {"df": biased_df, "labels": biased_labels, "target": 0, "w": biased_weights, "color": "darkblue", "histtype": "step", "alpha": 1.0, "lbl": f"Background ({bias_name} {shift})"}
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
    plt.title(f"Feature: {feat} | Systematic Impact ({bias_name} {shift})", fontsize=11, fontweight='bold', pad=10)
    plt.xlabel(feat, fontsize=10)
    plt.ylabel("Probability Density", fontsize=10)
    
    plt.legend(fontsize="small", frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    
    plt.tight_layout()
    plt.show()
    
    print(
        f"[{feat}] {bias_name} {shift}\n χ² (signal): {chi2_sig:.2f}, χ² (background): {chi2_bkg:.2f}"
    )


def Score_systematics (data,features=features_all ) :
    table_chi2=impact_syst_bias_all(data, features=features)
    score = table_chi2.sum(axis=1).to_dict()
    score_total = {}
    for feat, chi2 in score.items():
        base_feature = feat.replace(" (signal)", "").replace(" (background)", "")
        score_total[base_feature] = score_total.get(base_feature, 0) + chi2
    plt.figure(figsize=(10, 6))
    plt.bar(score_total.keys(), score_total.values(), color="red")
    
    # Add labels and title
    plt.xlabel("Features")
    plt.ylabel("Score")
    plt.title("Total Impact of bias Score")

    # module_path = os.path.join(os.getcwd(), "sample_code_submission", "BDT")
    # if module_path not in sys.path:
    #   sys.path.append(module_path)

    # import sample_code_submission.BDT.boosted_decision_tree as BoostedDecisionTree

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Display the plot
    plt.show()
    return sorted(score_total.keys(), key=lambda k: score_total[k], reverse=True)

def minimal_dependent_features(data):
    return data.columns

systematic_dependence(Data("c:/Users/noahl/Documents/EI_PP"), show=True)