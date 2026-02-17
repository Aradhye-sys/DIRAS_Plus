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
    r = np.correlate(x, x, mode="full")[len(x)-1:]
    r = r[:order + 1]
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

def DIRAS(
    y,
    lam=1e4,
    ar_order=50,
    omega=0.05,
    zeta=2.0,
    ratio=1e-6,
    eps=1e-6,
    max_iter=10,
    w_floor=1e-2,
    w_ceiling=1.0,
):
    y = np.asarray(y, dtype=float).ravel()
    L = len(y)

    noise_level = float(np.std(y))
    alpha = 0.5 if noise_level > 0.05 else 0.8

    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())

    initial_kernel = ar_model_kernel_psd(y, order=ar_order, eps=eps, alpha=alpha)
    w = 1.0 - initial_kernel
    w = np.clip(w, w_floor, w_ceiling)

    baseline_old = np.zeros(L, dtype=float)

    for _ in range(max_iter):
        W = diags(w, 0)
        Z = W + D
        baseline = spsolve(Z, w * y)
        residual = y - baseline

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
        w_pos = sigmoid_pos * (1.0 - omega_dynamic[pos] * kernel[pos])
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

        baseline_change = np.linalg.norm(baseline - baseline_old) / (np.linalg.norm(baseline) + eps)
        weight_change = np.linalg.norm(w - w_new) / (np.linalg.norm(w) + eps)

        if (baseline_change < eps) and (weight_change < ratio) and (np.sum(neg) < L * 0.01):
            lam_final = lam * 0.01
            D_final = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
            D_final = lam_final * D_final.dot(D_final.transpose())
            Z_final = diags(w, 0) + D_final
            baseline = spsolve(Z_final, w * y)
            return baseline

        baseline_old = baseline
        w = w_new

    return baseline

