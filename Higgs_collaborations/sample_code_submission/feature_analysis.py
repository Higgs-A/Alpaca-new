import sys
from pathlib import Path

# Ensure the package containing `HiggsML` is on sys.path so we can import HiggsML.datasets
black_swan_pkg_path = r'c:\Users\geoff\Documents\Centrale\cours_centrale\ST4\EI\black_swan_pkg'
if black_swan_pkg_path not in sys.path:
    sys.path.insert(0, black_swan_pkg_path)

from HiggsML.datasets import download_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = download_dataset("blackSwan_data")
# Ensure a train set is loaded: Data.get_train_set() returns None until load_train_set() is called
try:
    data_set = data.get_train_set()
    if data_set is None:
        data.load_train_set()
        data_set = data.get_train_set()
except Exception as e:
    print("Error loading train set:", e)
    data_set = None

def feature_corrilations(data):
    
    
    # Validate input
    if data is None:
        raise ValueError("No data provided to feature_corrilations: train set is empty or failed to load")

    # we Handle the -25 sentinel value
    data_cleaned = data.replace(-25, np.nan)
    
    #we create vectors with the physical values, we don't take the colums labels"
    feature_cols = [col for col in data_cleaned.columns if col.startswith('PRI_') or col.startswith('DER_')]
    
    #calcul of the matrix of correlation#
    corr_matrix = data_cleaned[feature_cols].corr()
    
    # Debug: print shape to ensure it's 28x28 (or show actual shape)
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
    plt.show()
    
   
    


def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns


if __name__ == "__main__":
    # When run as a script, show the correlation matrix for the training set
    feature_corrilations(data_set)
