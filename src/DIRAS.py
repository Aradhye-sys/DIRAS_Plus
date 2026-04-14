"""
DIRAS (Dynamic Iterative Reweighted Autoregressive Spectral baseline correction algorithm)
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d

_EPS = 1e-12

def _mad(x):
    x = np.asarray(x, float).ravel()
    m = np.median(x)
    return 1.4826 * (np.median(np.abs(x - m)) + _EPS)

def global_noise_gate(residual, hf_start_frac=0.25, target=0.15, p=2.0, eps=1e-12):
    r = np.asarray(residual, float).ravel()
    r = r - r.mean()
    if r.size < 16:
        return 1.0
    spec = np.fft.rfft(r)
    pow_ = spec.real * spec.real + spec.imag * spec.imag
    tot = float(pow_.sum() + eps)
    k0 = int(np.floor((1.0 - hf_start_frac) * pow_.size))
    k0 = max(1, min(k0, pow_.size - 1))
    hf = float(pow_[k0:].sum())
    ratio = hf / tot
    gate = 1.0 / (1.0 + (ratio / (target + eps)) ** p)
    return float(np.clip(gate, 0.0, 1.0))

def _yule_walker_ar(x, order=30, eps=1e-10):
    x = np.asarray(x, float).ravel()
    if x.size < order + 5:
        order = max(5, x.size // 4)
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[len(x) - 1 :]
    r = r[: order + 1]
    R = toeplitz(r[:-1]) + np.eye(order) * eps
    a = np.linalg.solve(R, r[1:])
    sigma2 = float(max(r[0] - a @ r[1:], eps))
    return a.astype(float), sigma2


def _ar_hf_ratio(a, sigma2, nfreq=512, eps=1e-12, hf_start_frac=0.25):
    f = np.linspace(0, 0.5, nfreq, endpoint=True)
    H = np.ones_like(f, dtype=np.complex128)
    for k in range(a.size):
        H += a[k] * np.exp(-2j * np.pi * f * (k + 1))
    psd = sigma2 / (np.abs(H) ** 2 + eps)
    tot = float(psd.sum() + eps)
    k0 = int(np.floor((1.0 - hf_start_frac) * psd.size))
    k0 = max(1, min(k0, psd.size - 1))
    hf = float(psd[k0:].sum())
    return hf / tot


def ar_model_kernel_psd(x, order=50, eps=1e-6, alpha=0.7, sigma=6.0, gamma_clip=(1.0, 5.0)):
    x = np.asarray(x, float).ravel()
    L = x.size
    if L < 8:
        return np.ones(L, float)

    d2 = np.diff(x, n=2)
    d2 = np.pad(d2, (1, 1), mode="edge")
    peakness = gaussian_filter1d(d2 * d2, sigma=float(max(0.5, sigma)), mode="reflect")

    pmin, pmax = float(peakness.min()), float(peakness.max())
    if (pmax - pmin) < 1e-15:
        peak01 = np.zeros_like(peakness)
    else:
        peak01 = (peakness - pmin) / (pmax - pmin + _EPS)

    try:
        a, sigma2 = _yule_walker_ar(x, order=order, eps=eps * 1e-2)
        hf_ratio = _ar_hf_ratio(a, sigma2, nfreq=512, eps=eps)
    except Exception:
        hf_ratio = 0.15

    g0, g1 = gamma_clip
    gamma = float(np.clip(g0 + (g1 - g0) * (hf_ratio / (0.35 + _EPS)), g0, g1))

    kernel = 1.0 - (peak01 ** gamma)
    kernel = alpha * kernel + (1.0 - alpha)

    return np.clip(kernel, 0.0, 1.0)


def DIRAS(
    y,
    lam=5e5,
    max_iter=30,
    ar_order=50,
    alpha_ar=0.7,
    kernel_ema=0.12,
    sigma_struct=6.0,
    lam_boost=10.0,
    gate_target=0.15,
    gate_hf_frac=0.25,
    omega0=0.08,
    zeta0=2.0,
    omega_min=0.01,
    zeta_min=0.3,
    omega_pow=1.0,
    zeta_pow=0.6,
    beta=0.25,
    w_floor=1e-6,
    w_ceiling=1.0,
    stop_baseline=1e-4,
    stop_weight=1e-3,
    patience=3,
    eps=1e-6,
):
    y = np.asarray(y, float).ravel()
    L = y.size
    if L < 5:
        return y.copy()

    D2 = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D_base = D2 @ D2.T

    baseline_old = np.zeros(L, float)

    y0 = gaussian_filter1d(y, sigma=float(max(0.5, sigma_struct * 0.6)), mode="reflect")
    kernel = ar_model_kernel_psd(y0, order=ar_order, eps=eps, alpha=alpha_ar)
    w = np.clip(kernel, w_floor, w_ceiling)

    stable = 0

    for _ in range(int(max_iter)):

        gate = global_noise_gate(y - baseline_old, hf_start_frac=gate_hf_frac, target=gate_target)

        lam_eff = lam * (1.0 + lam_boost * (1.0 - gate))

        omega_eff = omega_min + (omega0 - omega_min) * (gate ** omega_pow)
        zeta_eff = zeta_min + (zeta0 - zeta_min) * (gate ** zeta_pow)

        W = diags(w, 0)
        baseline = spsolve(W + lam_eff * D_base, w * y)

        residual = y - baseline

        res_struct = gaussian_filter1d(residual, sigma=float(max(0.5, sigma_struct)), mode="reflect")

        k_new = ar_model_kernel_psd(res_struct, order=ar_order, eps=eps, alpha=alpha_ar)

        kernel = np.clip((1.0 - kernel_ema) * kernel + kernel_ema * k_new, 0.0, 1.0)

        neg = residual < 0
        pos = ~neg

        if np.any(neg):
            mu_neg = np.mean(residual[neg])
            sig_neg = np.std(residual[neg]) + eps
        else:
            mu_neg = 0.0
            sig_neg = _mad(residual) + eps

        w_new = np.empty(L, float)

        if np.any(pos):
            d = residual[pos]
            t = np.clip((d - (mu_neg + 2.0 * sig_neg)) / sig_neg, -60, 60)
            logistic = 1.0 / (1.0 + np.exp(t))
            w_new[pos] = logistic * (1.0 - omega_eff * (1.0 - kernel[pos]))
        else:
            w_new[pos] = 1.0

        if np.any(neg):
            d = residual[neg]
            t = np.clip((-d) / sig_neg, -60, 60)
            logistic = 1.0 / (1.0 + np.exp(-t))
            w_new[neg] = zeta_eff * kernel[neg] * logistic
        else:
            w_new[neg] = 1.0

        w_new = np.clip(w_new, w_floor, w_ceiling)

        w_damped = np.clip((1.0 - beta) * w + beta * w_new, w_floor, w_ceiling)

        bchg = np.linalg.norm(baseline - baseline_old) / (np.linalg.norm(baseline) + eps)
        wchg = np.linalg.norm(w_damped - w) / (np.linalg.norm(w) + eps)

        if bchg < stop_baseline and wchg < stop_weight:
            stable += 1
            if stable >= patience:
                break
        else:
            stable = 0

        baseline_old = baseline
        w = w_damped

    return baseline
