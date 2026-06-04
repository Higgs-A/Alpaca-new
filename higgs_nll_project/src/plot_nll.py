from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from src.statistical_analysis import compute_mu_binned, prepare_binned
from src.models.bdt_model import load_bdt_model, predict_bdt

def plot_binned_nll(data, weights, model, n_bins=5):
    S_scores = data['signal_scores']
    S_weights = data['signal_weights']
    B_scores = data['background_scores']
    B_weights = data['background_weights']
    N_scores = data['data_scores']
    N_weights = data['data_weights']

    Nb, Sb, Bb = prepare_binned(n_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights)
    
    mu0 = 1.0  # Initial guess for mu
    compute_mu_binned(mu0, n_bins, S_scores, S_weights, B_scores, B_weights, N_scores, N_weights)

    # Plotting the NLL
    mu_values = np.linspace(0, 5, 1000)
    nll_values = np.array([NLL(mu, Nb, Sb, Bb) for mu in mu_values])
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, nll_values, label='Negative Log-Likelihood', color='blue')
    plt.axhline(y=np.min(nll_values) + 0.5, color='red', linestyle='--', label='ΔNLL = 0.5')
    plt.xlabel('μ')
    plt.ylabel('NLL')
    plt.title('Binned Negative Log-Likelihood')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load the BDT model
    model = load_bdt_model('path/to/your/model')  # Adjust the path as necessary

    # Load the blackSwan_data
    data = np.load('data/blackSwan_data/data_file.npy', allow_pickle=True).item()  # Adjust the file name as necessary

    # Plot the binned NLL
    plot_binned_nll(data, weights=None, model=model)  # Adjust weights as necessary