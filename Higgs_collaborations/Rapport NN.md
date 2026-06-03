# Travail sur le WG Neural Network

## Lundi 01/06

Prise en main du projet. Lecture du `walkthrough.md` et de tous les fichiers du
starter kit (`model.py`, `neural_network.py`, `utils.py`, etc.) pour comprendre
comment les 5 working groups s'articulent entre eux.

Analyse du `neural_network.py` de départ : un réseau à 2 couches
de 10 neurones, entraîné 5 epochs. 
Identification des principales faiblesses :
- Capacité trop faible pour un problème comme H→ττ
- Pas de protection contre l'overfitting (ni dropout, ni batch norm, ni early
  stopping)
- Métrique `accuracy` inutile vu que le bruit représente 99.85% des événements
- Trop peu d'epochs pour que le réseau converge

Première version améliorée écrite avec :
- Architecture 4 couches de 256 neurones (largeur constante)
- Dropout 0.4 sur les couches cachées (pas sur l'entrée)
- BatchNormalization entre Dense et activation
- Early stopping avec patience=15 et validation_split=0.2


## Mardi 02/06

Setup technique le matin : installation de Python 3.12 (la 3.14 n'est pas
compatible avec TensorFlow), création d'un venv, install des dépendances
(tensorflow, xgboost, scikit-learn, HiggsML, etc.), connexion SSH au repo.

Recherche en parallèle sur les architectures NN utilisées en physique des
particules sur ce dataset. 

Mon premier choix d'architecture (entonnoir) était à revoir. Réécriture
de `neural_network.py` avec :
- Architecture rectangulaire 4×256 (au lieu de 128→64→32→16)
- Dropout passé à 0.5 sur les cachées
- Optimiseur Adam avec learning rate 0.0005 (au lieu du défaut 0.001)
- ReduceLROnPlateau qui divise le LR par 2 quand val_auc stagne 8 epochs
- Métrique AUC pour le monitoring et l'early stopping (au lieu de accuracy)
- Méthodes `save()` et `load()` pour sauvegarder le modèle entraîné
- Méthode `significance()` qui calcule l'AMS 

Modification de `model.py` pour utiliser `model_type="NN"` au lieu de "BDT" et
ajout d'une cellule dans le notebook pour afficher l'AMS après l'entraînement.

## Résultats

Exécution du pipeline complet sur 500 pseudo-expériences.

| Métrique         | BDT baseline | Mon   NN  |
|------------------|--------------|-----------|
| RMSE             | 7.97         | 2.58      |
| MAE              | 5.19         | 2.63      |
| Coverage         | 0.0          | 0.0       |
| Quantile Score   | -13.15       | -13.14    |
| AMS (max)        | n/a          | **4.36**  |

