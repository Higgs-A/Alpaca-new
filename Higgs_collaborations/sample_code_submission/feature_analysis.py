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
  

def minimal_dependent_features(data):
    return data.columns

data = Data("c:/Users/noahl/Documents/EI_PP")
impact_syst_bias_all(data)