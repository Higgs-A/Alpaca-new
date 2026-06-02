import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    

def feature_corrilations(data):
    
    
    #we Handle the -25 sentinel value#
    data_cleaned = data.replace(-25, np.nan)
    
    #we create vectors with the physical values, we don't take the colums labels"
    feature_cols = [col for col in data.columns if col.startswith('PRI_') or col.startswith('DER_')]
    
    #calcul of the matrix of correlation#
    corr_matrix = data_cleaned[feature_cols].corr()
    
    print(f"=== Correlation matrix ===")
    print(f"Dimensions of the matrix : {corr_matrix.shape} (28x28)")
    
    print(corr_matrix)
    return corr_matrix
    


def systematics_dependence(data):
    pass


def minimal_dependent_features(data):
    return data.columns
