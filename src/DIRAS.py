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
    x -= np.mean(x)
    r = np.correlate(x, x, mode="full")[len(x)-1:]
    r = r[:order+1]
    R = toeplitz(r[:-1]) + np.eye(order)*eps
    a = np.linalg.solve(R, r[1:])
    sigma2 = max(r[0] - a @ r[1:], eps)
    return a.astype(float), float(sigma2)

@njit
def _ar_psd_numba(a, sigma2, freq, eps=1e-12):
    nfreq = freq.shape[0]
    p = a.shape[0]
    psd = np.empty(nfreq, dtype=np.float64)
    for i in range(nfreq):
        re = 1.0
        im = 0.0
        for k in range(p):
            ang = 2.0*np.pi*freq[i]*(k+1)
            re -= a[k]*np.cos(ang)
            im += a[k]*np.sin(ang)
        psd[i] = sigma2/(re*re + im*im + eps)
    return psd

def ar_model_kernel_psd(residual, order=50, eps=1e-10, alpha=0.6):
    x = np.asarray(residual, dtype=float)
    x -= np.mean(x)
    order = min(order, max(1, len(x)-3))
    a, sigma2 = _yule_walker_ar(x, order, eps)
    freq = np.fft.fftfreq(len(x), d=1.0)
    psd = _ar_psd_numba(a, sigma2, freq.astype(np.float64), eps)
    energy = np.abs(np.fft.ifft(psd)).real
    rms = np.sqrt(np.mean(energy*energy)+eps)
    kernel = 1.0 - alpha*np.exp(-energy/(rms+eps))
    return np.clip(kernel, 0.0, 1.0)

def _smoothness_penalty(z):
    return float(np.sum(np.diff(z, n=2)**2))

def DIRAS_v7(y, lam=1e4, ar_order=50, omega=0.05, zeta=2.0, eps=1e-6, max_iter=50, w_floor=1e-6, w_ceiling=1.0, protect_scale=2.0, beta=0.2, kernel_freeze_iter=10, stop_baseline=1e-4, stop_weight=1e-3, patience=3, return_debug=False):
    y = np.asarray(y, dtype=float).ravel()
    L = len(y)
    if L < 5:
        return (y.copy(), {}) if return_debug else y.copy()
    alpha = 0.5
    D = diags([1,-2,1],[0,-1,-2],shape=(L,L-2))
    D = lam*D.dot(D.transpose())
    kernel = ar_model_kernel_psd(y, ar_order, eps, alpha)
    w = np.clip(1.0-kernel, w_floor, w_ceiling)
    baseline_old = np.zeros(L)
    best_baseline = None
    best_obj = np.inf
    stable_count = 0
    debug = {"baseline_change":[],"weight_change":[],"objective":[],"mean_weight":[],"residual_std":[]}
    for it in range(max_iter):
        baseline = spsolve(diags(w,0)+D, w*y)
        residual = y-baseline
        if it < kernel_freeze_iter:
            kernel = ar_model_kernel_psd(residual, ar_order, eps, alpha)
        neg = residual<0
        pos = ~neg
        neg_vals = residual[neg]
        if neg_vals.size:
            mean_res = np.mean(neg_vals)
            std_res = np.std(neg_vals)+eps
        else:
            mean_res = 0.0
            std_res = np.std(residual)+eps
        res_std = np.std(residual)+eps
        omega_dyn = omega*(1.0-np.exp(-np.abs(residual)/res_std))
        zeta_dyn = zeta*(1.0-np.exp(-np.abs(residual)/res_std))
        w_new = np.empty(L)
        d_pos = residual[pos]
        sig_pos = 1.0/(1.0+np.exp(np.clip((d_pos-mean_res)/std_res,-60,60)))
        protect = np.exp(-(d_pos/(protect_scale*res_std))**2)
        w_new[pos] = sig_pos*protect*(1.0-omega_dyn[pos]*kernel[pos])
        d_neg = residual[neg]
        sig_neg = 1.0/(1.0+np.exp(np.clip(-2.0*(d_neg-(mean_res-2.0*std_res))/std_res,-60,60)))
        w_new[neg] = zeta_dyn[neg]*kernel[neg]*sig_neg
        weak = (np.abs(residual)<0.1*res_std)&neg
        if np.any(weak):
            w_new[weak] = 1.0
        w_new = np.clip(w_new, w_floor, w_ceiling)
        w_damped = np.clip((1.0-beta)*w + beta*w_new, w_floor, w_ceiling)
        baseline_change = np.linalg.norm(baseline-baseline_old)/(np.linalg.norm(baseline)+eps)
        weight_change = np.linalg.norm(w_damped-w)/(np.linalg.norm(w)+eps)
        obj = np.sum(w_damped*residual**2) + lam*_smoothness_penalty(baseline)
        if obj < best_obj:
            best_obj = obj
            best_baseline = baseline.copy()
        debug["baseline_change"].append(baseline_change)
        debug["weight_change"].append(weight_change)
        debug["objective"].append(obj)
        debug["mean_weight"].append(np.mean(w_damped))
        debug["residual_std"].append(res_std)
        if baseline_change < stop_baseline and weight_change < stop_weight:
            stable_count += 1
            if stable_count >= patience:
                break
        else:
            stable_count = 0
        baseline_old = baseline
        w = w_damped
    if best_baseline is None:
        best_baseline = baseline
    return (best_baseline, debug) if return_debug else best_baseline
