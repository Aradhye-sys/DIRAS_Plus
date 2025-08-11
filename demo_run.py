
#%%
import os, sys, warnings
import pandas as pd
import matplotlib.pyplot as plt

# Path to your working folder
BASE_DIR = r"/DIRAS+"

# Add that folder to sys.path so Python can import your code directly
sys.path.append(BASE_DIR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# Import your own functions from that path
from src.DIRAS import DIRAS
from src.DIRAS_plus import diras_plus_xgb

# File paths in your BASE_DIR
SPECTRA_CSV = os.path.join(BASE_DIR, "Synthetic_spectra.csv")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.keras")
XGB_PATH = os.path.join(BASE_DIR, "XGBoost.joblib")

# Load the spectra: first col = wavenumber, rest = spectra
df = pd.read_csv(SPECTRA_CSV)
wavenumber = df.iloc[:, 0].to_numpy()
Y = df.iloc[:, 1:].to_numpy()
n_points, n_spectra = Y.shape

def run_diras_all(Y, lam=1e5, ar_order=50, omega=0.01, zeta=2):
    B = np.zeros_like(Y)
    for k in range(Y.shape[1]):
        B[:, k] = DIRAS(Y[:, k], lam=lam, ar_order=ar_order, omega=omega, zeta=zeta)
    return B

# Run DIRAS (fixed λ)
B_diras = run_diras_all(Y)
C_diras = Y - B_diras

# Run DIRAS+ (predicted λ)
lam_hat, B_plus, C_plus = diras_plus_xgb(
    wavenumber=wavenumber,
    spectra=Y,
    diras_fn=DIRAS,
    encoder_path=ENCODER_PATH,
    xgb_path=XGB_PATH,
    pre_lam=1e4,
    ar_order=50,
    omega=0.01,
    zeta=2,
    pad_len=1536,
)

# Plot comparison DIRAS vs DIRAS+
j = 1 if n_spectra > 2 else n_spectra - 1
plt.figure(figsize=(10, 6))
plt.plot(wavenumber, Y[:, j], label="Raw", alpha=0.5, color="gray")
plt.plot(wavenumber, B_diras[:, j], "-", lw=2, label="DIRAS (fixed λ)", color="tab:red")
plt.plot(wavenumber, B_plus[:, j], "-", lw=2, label="DIRAS+ (predicted λ)", color="tab:green")
plt.title(f"Baseline comparison | Spectrum {j+1}")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Intensity (a.u.)")
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.grid(False)
plt.gca()
plt.show()

#%%
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from src.DIRAS import DIRAS
from src.DIRAS_plus import diras_plus_xgb

SPECTRA_CSV = ROOT / "data" / "Synthetic_spectra.csv"
ENCODER_PATH = ROOT / "models" / "encoder.keras"
XGB_PATH = ROOT / "models" / "xgb_model.joblib"
if not XGB_PATH.exists():
    alt = ROOT / "models" / "XGBoost.joblib"
    if alt.exists(): XGB_PATH = alt

df = pd.read_csv(SPECTRA_CSV)
wn = df.iloc[:, 0].to_numpy()
Y  = df.iloc[:, 1:].to_numpy()

def run_diras_all(Y, lam=1e5, ar_order=50, omega=0.01, zeta=2):
    B = np.zeros_like(Y)
    for k in range(Y.shape[1]):
        B[:, k] = DIRAS(Y[:, k], lam=lam, ar_order=ar_order, omega=omega, zeta=zeta)
    return B

B_diras = run_diras_all(Y)

lam_hat, B_plus, C_plus = diras_plus_xgb(
    wavenumber=wn, spectra=Y, diras_fn=DIRAS,
    encoder_path=str(ENCODER_PATH), xgb_path=str(XGB_PATH),
    pre_lam=1e4, ar_order=50, omega=0.01, zeta=2, pad_len=1536,)


j = 2 if Y.shape[1] > 2 else Y.shape[1]-1
plt.figure(figsize=(10,5))
plt.plot(wn, Y[:,j], label="Raw", alpha=0.5)
plt.plot(wn, B_diras[:,j], "--", label="DIRAS (fixed λ)")
plt.plot(wn, B_plus[:,j], "--", label="DIRAS+ (predicted λ)")
plt.gca().invert_xaxis(); plt.legend(); plt.tight_layout(); plt.show()
