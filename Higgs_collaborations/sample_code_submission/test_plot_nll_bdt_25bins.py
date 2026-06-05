import numpy as np
import matplotlib.pyplot as plt
import statistical_analysis as stat

# Synthetic "BDT-like" saved_info for 25 bins
nbins = 25
# Nominal expectations
S_nom = np.linspace(60.0, 5.0, nbins)  # signal decreasing with bin
B_nom = np.linspace(200.0, 20.0, nbins)  # background

# Build saved_info structure expected by the analysis code
saved_info = {
    "S": {"nominal": S_nom.tolist()},
    "B": {"nominal": B_nom.tolist()},
}

# Define synthetic systematics: for each bin provide [plus, minus] deltas
# TES and JES predominantly affect signal shape; BNORM scales background; SMET small additive
for syst in ["tes", "jes", "bnorm", "smet"]:
    saved_info["S"][syst] = []
    saved_info["B"][syst] = []

for i in range(nbins):
    s = S_nom[i]
    b = B_nom[i]
    # TES / JES: +/- 3% on signal, smaller on background
    delta_tes_plus_s = 0.03 * s
    delta_tes_minus_s = -0.03 * s
    delta_tes_plus_b = 0.005 * b
    delta_tes_minus_b = -0.005 * b

    delta_jes_plus_s = 0.025 * s
    delta_jes_minus_s = -0.025 * s
    delta_jes_plus_b = 0.004 * b
    delta_jes_minus_b = -0.004 * b

    # BNORM: background normalization +/-5%
    delta_bnorm_plus_s = 0.0
    delta_bnorm_minus_s = 0.0
    delta_bnorm_plus_b = 0.05 * b
    delta_bnorm_minus_b = -0.05 * b

    # SMET: small additive shifts
    delta_smet_plus_s = 0.01 * s
    delta_smet_minus_s = -0.01 * s
    delta_smet_plus_b = 0.02 * b
    delta_smet_minus_b = -0.02 * b

    saved_info["S"]["tes"].append([float(delta_tes_plus_s), float(delta_tes_minus_s)])
    saved_info["B"]["tes"].append([float(delta_tes_plus_b), float(delta_tes_minus_b)])

    saved_info["S"]["jes"].append([float(delta_jes_plus_s), float(delta_jes_minus_s)])
    saved_info["B"]["jes"].append([float(delta_jes_plus_b), float(delta_jes_minus_b)])

    saved_info["S"]["bnorm"].append([float(delta_bnorm_plus_s), float(delta_bnorm_minus_s)])
    saved_info["B"]["bnorm"].append([float(delta_bnorm_plus_b), float(delta_bnorm_minus_b)])

    saved_info["S"]["smet"].append([float(delta_smet_plus_s), float(delta_smet_minus_s)])
    saved_info["B"]["smet"].append([float(delta_smet_plus_b), float(delta_smet_minus_b)])

# Observed counts: nominal plus Poisson fluctuations
np.random.seed(42)
n_obs = np.random.poisson(S_nom + B_nom).astype(float)

# Define scan
mu_values = np.linspace(0.0, 3.0, 80)

# Scenarios to compare (1,2,3 and all fixed)
scenarios = [
    ("1 fixee (TES+1sigma)", ["tes"], {"tes": stat.SIGMA_SHIFTS["tes"]["plus"]}),
    ("2 fixees (TES+JES, +1sigma)", ["tes", "jes"], {"tes": stat.SIGMA_SHIFTS["tes"]["plus"], "jes": stat.SIGMA_SHIFTS["jes"]["plus"]}),
    (
        "3 fixees (TES+JES+BNORM, +1sigma)",
        ["tes", "jes", "bnorm"],
        {
            "tes": stat.SIGMA_SHIFTS["tes"]["plus"],
            "jes": stat.SIGMA_SHIFTS["jes"]["plus"],
            "bnorm": stat.SIGMA_SHIFTS["bnorm"]["plus"],
        },
    ),
    (
        "Toutes fixees (+1sigma)",
        ["tes", "jes", "bnorm", "smet"],
        {
            "tes": stat.SIGMA_SHIFTS["tes"]["plus"],
            "jes": stat.SIGMA_SHIFTS["jes"]["plus"],
            "bnorm": stat.SIGMA_SHIFTS["bnorm"]["plus"],
            "smet": stat.SIGMA_SHIFTS["smet"]["plus"],
        },
    ),
]

plt.figure(figsize=(10, 7))
results = {}
for label, fixed_vars, fixed_vals in scenarios:
    res = stat.plot_minimized_nll_vs_mu_with_fixed_variables(
        mu_values=mu_values,
        n_obs=n_obs,
        saved_info=saved_info,
        fixed_variables=fixed_vars,
        fixed_values=fixed_vals,
        nbins=nbins,
        include_constraints=True,
        plot_show=False,
    )
    results[label] = res
    plt.plot(res["mu_values"], res["delta_nll"], linewidth=2.1, label=f"{label} | mu_hat={res['mu_hat']:.3f}")

plt.axhline(0.5, linestyle="--", color="black", alpha=0.8, label=r"$\Delta$NLL = 0.5")
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\Delta$NLL")
plt.title("Test: NLL superposé vs mu (BDT-like synthétique, 25 bins)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Optionally save results dict for inspection
import joblib
joblib.dump({"results": results, "n_obs": n_obs, "saved_info": saved_info}, "test_nll_bdt_25bins_results.joblib")
print("Test complete. Results dumped to test_nll_bdt_25bins_results.joblib")
