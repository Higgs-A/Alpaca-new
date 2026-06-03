import sys
from pathlib import Path
import os

# Ensure the package containing HiggsML is on sys.path so we can import HiggsML.datasets

black_swan_pkg_path = r'c:\Users\geoff\Documents\Centrale\cours_centrale\ST4\EI\black_swan_pkg'
if black_swan_pkg_path not in sys.path:
    sys.path.insert(0, black_swan_pkg_path)
from HiggsML.datasets import download_dataset

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
# from datasets import download_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Download dataset later (we change working dir first); keep this line commented to avoid duplicate downloads
# data = download_dataset("blackSwan_data")

# Changer vers le dossier du script pour télécharger les données localement
os.chdir(os.path.dirname(__file__))

data = download_dataset(
    "blackSwan_data"
)
data.load_train_set()
data_set = data.get_train_set()
target = data_set["labels"]
data_cleaned = data_set.replace(-25, np.nan)

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
    plt.show()
    return corr_matrix
    
feature_corrilations(data_jet_one(data_set))
feature_corrilations(data_jet_zero(data_set))
feature_corrilations(data_jet_two(data_set))
   
    






print(data_set.columns)
# print(data_set["detailed_labels"])
# print(data_set["weights"])
# print(data_set["labels"])

def feature_signal_background_correlations(data):
    

    sns.set_theme(rc={"figure.figsize": (10, 10)}, style="whitegrid")
    rows={}
    num_cols = data.shape[1]

    for i in range(16):       
        corrMatrix = data.iloc[:, [i, 18]].corr()
        rows[f"col{i}"]= corrMatrix.iloc[0, 1]
    for i in range(19, num_cols):       
        corrMatrix = data.iloc[:, [18, i]].corr()
        rows[f"col{i}"]= corrMatrix.iloc[0, 1]
    dfplot = pd.DataFrame([rows])


    # print("Features to Signal-Background correlation matrix")
    # sns.heatmap(dfplot, annot=True)
    # plt.title("Correlation matrix of features")
    # plt.show()
    #print(dfplot)
    return dfplot
    del dfplot
#feature_signal_background_correlations(data_set)



def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns


if __name__ == "__main__":
    # When run as a script, show the correlation matrix for the training set
    feature_corrilations(data_set)


# matrix_line = feature_signal_background_correlations(data_set)
# matrix_square = feature_corrilations(data_set)

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

def to_cartesian(data, particle_prefix):
    df = data.copy()
    
    pt = df[f'{particle_prefix}_pt']
    phi = df[f'{particle_prefix}_phi']
    eta = df[f'{particle_prefix}_eta']
    
    df[f'{particle_prefix}_px'] = pt * np.cos(phi)
    df[f'{particle_prefix}_py'] = pt * np.sin(phi)
    df[f'{particle_prefix}_pz'] = pt * np.sinh(eta)
    
    return df

def conversion_numbers_into_names(score, indices):
    feature_cols = [col for col in data_cleaned.columns if col.startswith('PRI_') or col.startswith('DER_')]
    selected_features = [feature_cols[i] for i in indices]
    return score, selected_features


