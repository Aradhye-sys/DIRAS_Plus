"""
DIRAS (Dynamic Iterative Reweighted Autoregressive Spectral baseline correction algorithm)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import toeplitz
from numba import njit

def ar_refine(x, order, eps=1e-6):
    """Refine autoregressive coefficients using Yule-Walker equations."""
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    r = ac[:order+1]
    R = toeplitz(r[:-1]) + np.eye(order) * eps
    return np.linalg.solve(R, r[1:])

@njit
def ar_psd(a, f, eps):
    """Compute power spectral density of AR model."""
    m = a.shape[0]
    num = np.zeros(f.shape[0], dtype=np.complex128)
    for k in range(m):
        for i in range(f.shape[0]):
            num[i] += a[k] * np.exp(-2j * np.pi * (k + 1) * f[i])
    return 1.0 / (np.abs(1 + num) + eps)

def ar_kernel(x, order=50, eps=1e-6, alpha=0.5):
    """Compute AR-based adaptive kernel for weighting."""
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    r = ac[:order+1]
    R = toeplitz(r[:-1]) + np.eye(order) * eps
    a = np.linalg.solve(R, r[1:])
    f = np.fft.fftfreq(len(x), d=1/len(x))
    s = ar_psd(a, f, eps)
    e = np.abs(np.fft.ifft(s))
    rms = np.sqrt(np.mean(e**2))
    return 1 - alpha * np.exp(-e / (rms + eps))

def noise_var(x, order=50, eps=1e-6):
    """Estimate noise variance using AR model prediction error."""
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    r = ac[:order+1]
    R = toeplitz(r[:-1]) + np.eye(order) * eps
    a = np.linalg.solve(R, r[1:])
    return r[0] - np.dot(a, r[1:])

def DIRAS(y, lam=1e6, ar_order=50, omega=0.05, zeta=2, ratio=1e-6, eps=1e-6):
    """
    DIRAS baseline correction algorithm.
    
    Parameters:
    -----------
    y : array-like
        Input signal
    lam : float
        Regularization parameter
    ar_order : int
        Autoregressive model order
    omega : float
        Weight for positive residuals
    zeta : float
        Weight for negative residuals
    ratio : float
        Convergence threshold
    eps : float
        Small constant for numerical stability
    
    Returns:
    --------
    b : array
        Estimated baseline
    """
    # Adaptive alpha based on signal variance
    alpha = 0.8 if np.std(y) > 0.05 else 0.5
    n = len(y)
    
    # Second-order difference matrix
    D = diags([1, -2, 1], [0, -1, -2], shape=(n, n-2))
    D = lam * D.dot(D.transpose())
    
    # Initial adaptive weights
    w = 1 - ar_kernel(y, order=ar_order, eps=eps, alpha=alpha)
    b_old = np.zeros(n)
    
    # Iterative refinement
    for _ in range(10):
        W = diags(w, 0)
        b = spsolve(W + D, w * y)
        r = y - b
        
        # Update kernel
        k = ar_kernel(r, order=ar_order, eps=eps, alpha=alpha)
        
        # Handle negative residuals
        neg = r[r < 0]
        mu = np.mean(neg) if neg.size else 0
        sd = np.std(neg) if neg.size else eps
        
        # Separate positive and negative masks
        pos_mask = r >= 0
        neg_mask = ~pos_mask
        
        # Update weights for positive residuals
        epos = np.clip((r[pos_mask] - mu) / (sd + eps), -100, 100)
        omega_dyn = omega * (1 - np.exp(-np.abs(r) / (np.std(r) + eps)))
        w_pos = 1.0 / (1.0 + np.exp(epos))
        w_pos *= (1 - omega_dyn[pos_mask] * k[pos_mask])
        
        # Update weights for negative residuals
        eneg = np.clip(-2 * (r[neg_mask] - (mu - 2 * sd)) / (sd + eps), -100, 100)
        zeta_dyn = zeta * (1 - np.exp(-np.abs(r) / (np.std(r) + eps)))
        w_neg = zeta_dyn[neg_mask] * k[neg_mask] * (1.0 / (1.0 + np.exp(eneg)))
        
        # Handle weak signals
        weak = (np.abs(r) < (0.1 * (np.std(r) + eps))) & neg_mask
        
        # Combine weights
        wn = np.zeros(n)
        wn[pos_mask] = w_pos
        wn[neg_mask] = w_neg
        wn[weak] = 1
        
        # Check convergence
        db = np.linalg.norm(b - b_old) / (np.linalg.norm(b) + eps)
        dw = np.linalg.norm(w - wn) / (np.linalg.norm(w) + eps)
        if db < eps and dw < ratio and np.sum(r < 0) < n * 0.01:
            b = spsolve(diags(w, 0) + D, w * y)
            return b
        
        b_old = b
        w = wn
    
    return b
