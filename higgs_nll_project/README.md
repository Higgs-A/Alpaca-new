# Higgs NLL Project

## Overview
The Higgs NLL Project is designed to perform statistical analysis on Higgs boson data using advanced machine learning techniques, specifically Boosted Decision Trees (BDT). The project focuses on calculating the negative log-likelihood (NLL) for various models and visualizing the results through informative plots.

## Project Structure
```
higgs_nll_project
├── src
│   ├── statistical_analysis.py  # Functions for statistical analysis and NLL calculations
│   ├── plot_nll.py              # Script for plotting the binned NLL graphic
│   └── models
│       └── bdt_model.py         # Definition and training of the BDT model
├── data
│   └── blackSwan_data           # Dataset used for analysis
├── requirements.txt              # List of project dependencies
└── README.md                     # Project documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Place your data files in the `data/blackSwan_data` directory. Ensure that the data is formatted correctly for processing.

2. **Model Training**: Use the `bdt_model.py` script to train the BDT model on your dataset. This model will be used for predictions in the NLL calculations.

3. **NLL Calculation and Plotting**: Run the `plot_nll.py` script to calculate the binned NLL and generate the corresponding plots. This script will utilize the trained BDT model and the statistical analysis functions.

## Dependencies
The project requires the following Python packages:
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- iminuit
- kiwisolver

Make sure to install these packages using the provided `requirements.txt`.

## Contributing
Contributions to the project are welcome. Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.