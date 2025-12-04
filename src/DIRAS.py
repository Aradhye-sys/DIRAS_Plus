"""
DIRAS (Dynamic Iterative Reweighted Autoregressive Spectral baseline correction algorithm)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import toeplitz
from numba import njit

def ar_refine(x, order, eps=1e-6):
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    r = ac[:order+1]
    R = toeplitz(r[:-1]) + np.eye(order) * eps
    return np.linalg.solve(R, r[1:])

@njit
def ar_psd(a, f, eps):
    m = a.shape[0]
    num = np.zeros(f.shape[0], dtype=np.complex128)
    for k in range(m):
        for i in range(f.shape[0]):
            num[i] += a[k] * np.exp(-2j * np.pi * (k + 1) * f[i])
    return 1.0 / (np.abs(1 + num) + eps)

def ar_kernel(x, order=50, eps=1e-6, alpha=0.5):
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
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    r = ac[:order+1]
    R = toeplitz(r[:-1]) + np.eye(order) * eps
    a = np.linalg.solve(R, r[1:])
    return r[0] - np.dot(a, r[1:])

def DIRAS(y, lam=1e4, ar_order=50, omega=0.05, zeta=2, ratio=1e-6, eps=1e-6):
    alpha = 0.6 if np.std(y) > 0.05 else 0.5
    n = len(y)
    
    D = diags([1, -2, 1], [0, -1, -2], shape=(n, n-2))
    D = lam * D.dot(D.transpose())

    w = 1 - ar_kernel(y, order=ar_order, eps=eps, alpha=alpha)
    b_old = np.zeros(n)
    
    for _ in range(20):
        W = diags(w, 0)
        b = spsolve(W + D, w * y)
        r = y - b
        
        k = ar_kernel(r, order=ar_order, eps=eps, alpha=alpha)
        
        neg = r[r < 0]
        mu = np.mean(neg) if neg.size else 0
        sd = np.std(neg) if neg.size else eps

        pos_mask = r >= 0
        neg_mask = ~pos_mask
        
        epos = np.clip((r[pos_mask] - mu) / (sd + eps), -100, 100)
        omega_dyn = omega * (1 - np.exp(-np.abs(r) / (np.std(r) + eps)))
        w_pos = 1.0 / (1.0 + np.exp(epos))
        w_pos *= (1 - omega_dyn[pos_mask] * k[pos_mask])
        
        eneg = np.clip(-2 * (r[neg_mask] - (mu - 2 * sd)) / (sd + eps), -100, 100)
        zeta_dyn = zeta * (1 - np.exp(-np.abs(r) / (np.std(r) + eps)))
        w_neg = zeta_dyn[neg_mask] * k[neg_mask] * (1.0 / (1.0 + np.exp(eneg)))
        
        weak = (np.abs(r) < (0.1 * (np.std(r) + eps))) & neg_mask

        wn = np.zeros(n)
        wn[pos_mask] = w_pos
        wn[neg_mask] = w_neg
        wn[weak] = 1
        
        db = np.linalg.norm(b - b_old) / (np.linalg.norm(b) + eps)
        dw = np.linalg.norm(w - wn) / (np.linalg.norm(w) + eps)
        if db < eps and dw < ratio and np.sum(r < 0) < n * 0.01:
            b = spsolve(diags(w, 0) + D, w * y)
            return b
        
        b_old = b
        w = wn
    
    return b
