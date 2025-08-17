# DIRAS+

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DIRAS+** (Dynamic Iterative Reweighted Autoregressive Spectral baseline correction algorithm) is an advanced baseline correction pipeline for Raman spectra that combines the original DIRAS algorithm with machine learning-based parameter optimization.

## Overview

DIRAS+ extends the original DIRAS algorithm by using a deep learning encoder and XGBoost regressor to automatically predict optimal regularization parameters for each spectrum, eliminating the need for manual parameter tuning.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/Aradhye-sys/DIRAS_Plus.git
cd DIRAS_Plus

# Install dependencies
pip install -r requirements.txt
```

### Alternative: Conda Installation

```bash
# Create conda environment (if environment.yml exists)
conda env create -f environment.yml
conda activate diras-plus

# Or install manually
conda install numpy scipy pandas matplotlib xgboost tensorflow joblib
```

## Dependencies

- `numpy` - Numerical computing
- `scipy` - Scientific computing and sparse matrices
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `xgboost` - Gradient boosting for parameter prediction
- `tensorflow>=2.12` - Deep learning framework
- `joblib` - Model serialization

## Quick Start

### Basic Usage

```python
import numpy as np
import pandas as pd
from src.DIRAS import DIRAS
from src.DIRAS_plus import diras_plus_xgb

# Load your spectral data
# First column: wavenumbers, remaining columns: spectral intensities
df = pd.read_csv("your_spectra.csv")
wavenumbers = df.iloc[:, 0].to_numpy()
spectra = df.iloc[:, 1:].to_numpy()

# Apply DIRAS+ with automatic parameter prediction
lam_pred, baseline, corrected = diras_plus_xgb(
    wavenumber=wavenumbers,
    spectra=spectra,
    diras_fn=DIRAS,
    encoder_path="models/encoder.keras",
    xgb_path="models/XGBoost.joblib"
)

# Access results
print(f"Predicted lambda values: {lam_pred}")
print(f"Baseline shape: {baseline.shape}")
print(f"Corrected spectra shape: {corrected.shape}")
```

### Demo Script

Run the included demo to see DIRAS+ in action:

```bash
python demo_run.py
```

This will:
1. Load synthetic spectral data
2. Apply DIRAS with fixed parameters
3. Apply DIRAS+ with ML-predicted parameters
4. Display comparison plots

## Project Structure

```
DIRAS_Plus/
├── data/
│   └── Synthetic_spectra.csv    # Example spectral data
├── models/
│   ├── encoder.keras            # Pre-trained encoder model
│   └── XGBoost.joblib          # Pre-trained XGBoost model
├── src/
│   ├── DIRAS.py                # Original DIRAS algorithm
│   ├── DIRAS_plus.py           # DIRAS+ with ML enhancement
│   └── CITATION.cff            # Citation information
├── demo_run.py                 # Demo script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## API Reference

### `DIRAS(y, lam=1e6, ar_order=50, omega=0.05, zeta=2, ratio=1e-6, eps=1e-6)`

Original DIRAS baseline correction algorithm.

**Parameters:**
- `y`: Input signal array
- `lam`: Regularization parameter
- `ar_order`: Autoregressive model order
- `omega`: Weight for positive residuals
- `zeta`: Weight for negative residuals
- `ratio`: Convergence threshold
- `eps`: Numerical stability constant

**Returns:**
- `b`: Estimated baseline array

### `diras_plus_xgb(wavenumber, spectra, diras_fn, encoder_path, xgb_path, **kwargs)`

DIRAS+ algorithm with ML-based parameter prediction.

**Parameters:**
- `wavenumber`: Wavenumber axis array
- `spectra`: Spectral data matrix
- `diras_fn`: DIRAS function reference
- `encoder_path`: Path to encoder model
- `xgb_path`: Path to XGBoost model
- `pre_lam`: Pre-processing lambda value
- `ar_order`: Autoregressive model order
- `omega`: Weight parameter for positive residuals
- `zeta`: Weight parameter for negative residuals
- `spline_s`: Spline smoothing parameter
- `pad_len`: Padding length for encoder input

**Returns:**
- `lam`: Predicted lambda values
- `baseline`: Estimated baselines
- `corrected`: Baseline-corrected spectra

## Performance

DIRAS+ typically provides:
- **Automatic parameter optimization** without manual tuning
- **Consistent performance** across different spectral types
- **Scalable processing** for large datasets

## Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation


---

**Note**: This is research software. Please ensure proper validation for your specific use case.
