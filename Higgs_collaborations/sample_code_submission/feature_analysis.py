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
        weights_original
  

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




def minimal_dependent_features(data):
    return data.columns

data = Data("c:/Users/noahl/Documents/EI_PP")
impact_syst_bias_all(data)