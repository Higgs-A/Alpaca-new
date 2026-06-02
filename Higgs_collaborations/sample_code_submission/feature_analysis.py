import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from datasets import download_dataset
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    del dfplot
feature_signal_background_correlations(data_set)


def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns
