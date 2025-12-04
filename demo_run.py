#%%
"""
DIRAS+ Demo Script - Baseline correction comparison between DIRAS, DIRAS+ and Raw spectra
"""

from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings and logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from src.DIRAS import DIRAS
from src.DIRAS_plus import diras_plus_xgb

# Specify Model and data paths
SPECTRA_CSV = ROOT / "data" / "Synthetic_spectra.csv"
ENCODER_PATH = ROOT / "models" / "encoder.keras"
XGB_PATH = ROOT / "models" / "xgb_model.joblib"
if not XGB_PATH.exists():
    alt = ROOT / "models" / "XGBoost.joblib"
    if alt.exists(): XGB_PATH = alt

# Load spectral data
df = pd.read_csv(SPECTRA_CSV)
wn = df.iloc[:, 0].to_numpy()  # Wavenumbers
Y  = df.iloc[:, 1:].to_numpy()  # Spectral intensities

def run_diras_all(Y, lam=1e4, ar_order=50, omega=0.01, zeta=2):
    """Apply DIRAS baseline correction to all spectra in dataset."""
    B = np.zeros_like(Y)
    for k in range(Y.shape[1]):
        B[:, k] = DIRAS(Y[:, k], lam=lam, ar_order=ar_order, omega=omega, zeta=zeta)
    return B

# Run baseline correction
B_diras = run_diras_all(Y)

# Run DIRAS+ with ML-predicted parameters
lam_hat, B_plus, C_plus = diras_plus_xgb(
    wavenumber=wn, spectra=Y, diras_fn=DIRAS,
    encoder_path=str(ENCODER_PATH), xgb_path=str(XGB_PATH),
    pre_lam=5e3, ar_order=50, omega=0.01, zeta=2, pad_len=1536,)

# Plot comparison
j = 1 if Y.shape[1] > 2 else Y.shape[1]-1
plt.figure(figsize=(10,5))
plt.plot(wn, Y[:,j], label="Raw", alpha=0.5, color="gray")
plt.plot(wn, B_diras[:,j], "--", label="DIRAS (fixed λ)", color="tab:red")
plt.plot(wn, B_plus[:,j], "--", label="DIRAS+ (predicted λ)", color="tab:green")
plt.gca(); plt.legend(); plt.tight_layout(); plt.show()
