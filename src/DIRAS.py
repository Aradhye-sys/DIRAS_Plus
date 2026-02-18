"""
DIRAS (Dynamic Iterative Reweighted Autoregressive Spectral baseline correction algorithm)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import toeplitz
from numba import njit

def _yule_walker_ar(x, order, eps=1e-10):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode="full")[len(x) - 1 :]
    r = r[: order + 1]
    R = toeplitz(r[:-1]) + np.eye(order) * eps
    a = np.linalg.solve(R, r[1:])
    sigma2 = float(max(r[0] - a @ r[1:], eps))
    return a.astype(float), sigma2

@njit
def _ar_psd_numba(a, sigma2, freq, eps=1e-12):
    nfreq = freq.shape[0]
    p = a.shape[0]
    psd = np.empty(nfreq, dtype=np.float64)
    for i in range(nfreq):
        re = 1.0
        im = 0.0
        for k in range(p):
            ang = 2.0 * np.pi * freq[i] * (k + 1)
            re -= a[k] * np.cos(ang)
            im += a[k] * np.sin(ang)
        den = re * re + im * im
        psd[i] = sigma2 / (den + eps)
    return psd

def ar_model_kernel_psd(residual, order=50, eps=1e-10, alpha=0.6):
    x = np.asarray(residual, dtype=float)
    x = x - np.mean(x)
    order = int(min(order, max(1, len(x) - 3)))
    a, sigma2 = _yule_walker_ar(x, order, eps=eps)
    N = len(x)
    freq = np.fft.fftfreq(N, d=1.0)
    psd = _ar_psd_numba(a, sigma2, freq.astype(np.float64), eps=eps)
    energy = np.abs(np.fft.ifft(psd)).real
    rms = np.sqrt(np.mean(energy * energy) + eps)
    kernel = 1.0 - alpha * np.exp(-energy / (rms + eps))
    return np.clip(kernel, 0.0, 1.0)

def _smoothness_penalty(z):
   
    d2 = np.diff(z, n=2)
    return float(np.sum(d2 * d2))

def DIRAS_v7(
    y,
    lam=1e4,
    ar_order=50,
    omega=0.05,
    zeta=2.0,
    ratio=1e-6,
    eps=1e-6,
    max_iter=50,
    w_floor=1e-6,
    w_ceiling=1.0,
    protect_scale=2.0,
    beta=0.2,                 
    kernel_freeze_iter=10,    
    stop_baseline=1e-4,       
    stop_weight=1e-3,         
    patience=3,               
    return_debug=False,       
):

    y = np.asarray(y, dtype=float).ravel()
    L = len(y)
    if L < 5:
        out = y.copy()
        return (out, {}) if return_debug else out

    noise_level = float(np.std(y))
    alpha = 0.5 if noise_level > 0.05 else 0.5  

    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())

    kernel = ar_model_kernel_psd(y, order=ar_order, eps=eps, alpha=alpha)
    w = 1.0 - kernel
    w = np.clip(w, w_floor, w_ceiling)

    baseline_old = np.zeros(L, dtype=float)

    best_baseline = None
    best_obj = np.inf

    stable_count = 0

    debug = {
        "baseline_change": [],
        "weight_change": [],
        "obj": [],
        "mean_w": [],
        "std_res": [],
    }

    for it in range(int(max_iter)):
        W = diags(w, 0)
        Z = W + D
        baseline = spsolve(Z, w * y)
        residual = y - baseline

        if it < int(kernel_freeze_iter):
            kernel = ar_model_kernel_psd(residual, order=ar_order, eps=eps, alpha=alpha)

        neg = residual < 0
        pos = ~neg

        neg_vals = residual[neg]
        if neg_vals.size > 0:
            mean_res = float(np.mean(neg_vals))
            std_res = float(np.std(neg_vals) + eps)
        else:
            mean_res = 0.0
            std_res = float(np.std(residual) + eps)

        res_std = float(np.std(residual) + eps)

        omega_dynamic = omega * (1.0 - np.exp(-np.abs(residual) / res_std))
        zeta_dynamic = zeta * (1.0 - np.exp(-np.abs(residual) / res_std))

        w_new = np.empty(L, dtype=float)

        d_plus = residual[pos]
        exp_arg_pos = (d_plus - mean_res) / (std_res + eps)
        exp_arg_pos = np.clip(exp_arg_pos, -60, 60)
        sigmoid_pos = 1.0 / (1.0 + np.exp(exp_arg_pos))

        protect = np.exp(- (d_plus / (protect_scale * res_std + eps)) ** 2)

        w_pos = sigmoid_pos * protect * (1.0 - omega_dynamic[pos] * kernel[pos])
        w_new[pos] = w_pos

        d_minus = residual[neg]
        exp_arg_neg = -2.0 * (d_minus - (mean_res - 2.0 * std_res)) / (std_res + eps)
        exp_arg_neg = np.clip(exp_arg_neg, -60, 60)
        sigmoid_neg = 1.0 / (1.0 + np.exp(exp_arg_neg))
        w_neg = zeta_dynamic[neg] * kernel[neg] * sigmoid_neg

        weak_peak_mask = (np.abs(residual) < (0.1 * res_std)) & neg
        if np.any(weak_peak_mask):
            w_neg[weak_peak_mask[neg]] = 1.0

        w_new[neg] = w_neg

        w_new = np.clip(w_new, w_floor, w_ceiling)

        beta_eff = float(np.clip(beta, 0.0, 1.0))
        w_damped = (1.0 - beta_eff) * w + beta_eff * w_new
        w_damped = np.clip(w_damped, w_floor, w_ceiling)

        baseline_change = float(np.linalg.norm(baseline - baseline_old) / (np.linalg.norm(baseline) + eps))
        weight_change = float(np.linalg.norm(w_damped - w) / (np.linalg.norm(w) + eps))

        obj = float(np.sum(w_damped * (residual * residual)) + lam * _smoothness_penalty(baseline))

        if obj < best_obj:
            best_obj = obj
            best_baseline = baseline.copy()

        debug["baseline_change"].append(baseline_change)
        debug["weight_change"].append(weight_change)
        debug["obj"].append(obj)
        debug["mean_w"].append(float(np.mean(w_damped)))
        debug["std_res"].append(res_std)

        if (baseline_change < stop_baseline) and (weight_change < stop_weight):
            stable_count += 1
            if stable_count >= int(patience):
                break
        else:
            stable_count = 0

        baseline_old = baseline
        w = w_damped

    if best_baseline is None:
        best_baseline = baseline

    return (best_baseline, debug) if return_debug else best_baseline
