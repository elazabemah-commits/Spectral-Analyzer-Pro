# Spectral Analyzer Pro

Desktop application for chemometrics and QSAR analysis of spectral data.

## Features

- Load spectral matrices (X) and target properties (y) from CSV or Excel
- Preprocessing: SNV, MSC, Savitzky-Golay (smoothing & derivatives), baseline detrending, mean centering
- Collinearity reduction (pairwise correlation + VIF)
- Feature / wavelength selection: VIP (PLS), correlation, interval (iPLS-style)
- Models: PLSR, PCR, Ridge, Lasso, ElasticNet, SVR, Random Forest, Extra Trees, Gradient Boosting, Gaussian Process, kNN, MLP, Stepwise MLR
- Leak-free cross-validation (Q²) and external validation
- Metrics: R², Adjusted R², RMSEC/RMSEP, CCC, RPD, F-statistic, Golbraikh–Tropsha criteria
- Interactive plots (predicted vs actual, residuals, SHAP, spectral overview)
- Excel report export, methods text, session log
- Save closed model pipeline and predict on new raw spectra

## Requirements

- Python 3.8+
- See `requirements.txt`

On Linux, also install Tkinter if needed:

```bash
# Ubuntu/Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter
```

## Installation

```bash
pip install -r requirements.txt
```

Optional (for SHAP explanations):

```bash
pip install shap
```

## Run

```bash
python SpecPro.py
```

## Developer

Dr. Abdelhalim Khaled  
Plant Ecophysiology  
University of Concepción
