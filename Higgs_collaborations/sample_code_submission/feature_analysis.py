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
    
   
    
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from datasets import download_dataset
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Changer vers le dossier du script pour télécharger les données localement
os.chdir(os.path.dirname(__file__))

data = download_dataset(
    "blackSwan_data"
)
data.load_train_set()
data_set = data.get_train_set()
target = data_set["labels"]

# print(data_set.columns)
# print(data_set["detailed_labels"])
# print(data_set["weights"])
# print(data_set["labels"])

def feature_signal_background_correlations(data):
    
    data_cleaned = data.replace(-25, np.nan)
    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")
    rows={}
    num_cols = data_cleaned.shape[1]

    for i in range(17):       
        corrMatrix = data_cleaned.iloc[:, [i, 18]].corr()
        rows[f"col{i}"]= corrMatrix.iloc[0, 1]
    for i in range(19, num_cols):       
        corrMatrix = data_cleaned.iloc[:, [18, i]].corr()
        rows[f"col{i}"]= corrMatrix.iloc[0, 1]
    dfplot = pd.DataFrame([rows])


    print("Features to Signal-Background correlation matrix")
    sns.heatmap(dfplot, annot=True)
    plt.title("Correlation matrix of features")
    plt.show()
    print(dfplot)
    return dfplot
    del dfplot
#feature_signal_background_correlations(data_set)

def best_features_set(n):
    k = 4  # Les entiers iront de 0 à 4 inclus (0, 1, 2, 3, 4)
    les_entiers = range(k + 1)
    combinaisons = list(itertools.combinations(les_entiers, n))
    return combinaisons
print(best_features_set(3))

def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns


if __name__ == "__main__":
    # When run as a script, show the correlation matrix for the training set
    feature_corrilations(data_set)


def value_signal(data_set, lst):
    E=0
    for i in range(lst): 
    E += (feature_signal_background_correlations(data_set).iloc[0,lst[i]])**2
    return E

def value_correlation(data_set, lst):
    E=0
    for i in range(lst): 
        for j in range(lst):
            if i>j:
                E += (feature_corrilations(data_set).iloc[lst[i],lst[j]])**2
    return E

def value(data_set, lst):
    n=len(lst)
    return value_signal(data_set, lst) - (2/n-1)*value_correlation(data_set, lst)
        
    
     
    