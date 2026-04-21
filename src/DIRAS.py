"""
DIRAS (Dynamic Iterative Reweighted Autoregressive Spectral baseline correction algorithm)
"""
import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d

_EPS = 1e-12
_DBASE_CACHE = {}

def _mad(x):
    x = np.asarray(x, float).ravel()
    m = np.median(x)
    return 1.4826 * (np.median(np.abs(x - m)) + _EPS)

def _smoothness_penalty(z):
    d2 = np.diff(np.asarray(z, float), n=2)
    return float(np.dot(d2, d2))

def _hf_start_index(n, frac):
    return max(1, min(int(np.floor((1.0 - frac) * n)), n - 1))

def _gsmooth(x, sigma):
    return gaussian_filter1d(x, sigma=float(max(0.5, sigma)), mode="reflect")

def _get_D_base(L):
    if L not in _DBASE_CACHE:
        D2 = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2), format="csc")
        _DBASE_CACHE[L] = csc_matrix(D2 @ D2.T)
    return _DBASE_CACHE[L]

# Frequency-guided conditioning
def _fft_power_ratio(x, hf_start_frac=0.25, eps=1e-12):
    x = np.asarray(x, float).ravel()
    x = x - x.mean()

    if x.size < 16:
        return 0.0

    spec = np.fft.rfft(x)
    power = spec.real**2 + spec.imag**2
    total = float(power.sum() + eps)
    k0 = _hf_start_index(len(power), hf_start_frac)
    return float(power[k0:].sum() / total)


def _adaptive_sigma(hf, sigma_range=(2.0, 40.0), hf_range=(0.05, 0.50), power=1.0):
    lo, hi = hf_range
    s_lo, s_hi = sigma_range
    t = np.clip((hf - lo) / (hi - lo + 1e-15), 0.0, 1.0) ** power
    return float(s_lo + t * (s_hi - s_lo))

# AR
def _yule_walker_ar(x, order=30, eps=1e-10):
    x = np.asarray(x, float).ravel()

    if x.size < order + 5:
        order = max(5, x.size // 4)

    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    r = r[:order + 1]

    R = toeplitz(r[:-1]) + np.eye(order) * eps
    a = np.linalg.solve(R, r[1:])
    sigma2 = float(max(r[0] - a @ r[1:], eps))
    return a.astype(float), sigma2

def _ar_hf_ratio(a, sigma2, nfreq=128, eps=1e-12, hf_start_frac=0.25):
    f = np.linspace(0, 0.5, nfreq, endpoint=True)
    H = np.ones_like(f, dtype=np.complex128)

    for k, ak in enumerate(a, start=1):
        H += ak * np.exp(-2j * np.pi * f * k)

    psd = sigma2 / (np.abs(H) ** 2 + eps)
    total = float(psd.sum() + eps)
    k0 = _hf_start_index(len(psd), hf_start_frac)
    return float(psd[k0:].sum() / total)

# AR-informed structural kernel
def ar_model_kernel_psd(
    x,
    order=50,
    eps=1e-6,
    alpha=0.7,
    sigma=6.0,
    gamma_clip=(1.0, 5.0),
):
    x = np.asarray(x, float).ravel()

    if x.size < 8:
        return np.ones_like(x, float)

    d2 = np.diff(x, n=2)
    d2 = np.pad(d2, (1, 1), mode="edge")
    peakness = _gsmooth(d2 * d2, sigma)

    pmin, pmax = float(peakness.min()), float(peakness.max())
    if (pmax - pmin) < 1e-15:
        peak01 = np.zeros_like(peakness)
    else:
        peak01 = (peakness - pmin) / (pmax - pmin + _EPS)

    try:
        a, sigma2 = _yule_walker_ar(x, order=order, eps=eps * 1e-2)
        hf_ratio = _ar_hf_ratio(a, sigma2, nfreq=128, eps=eps, hf_start_frac=0.25)
    except Exception:
        hf_ratio = 0.15

    g0, g1 = gamma_clip
    gamma = float(np.clip(g0 + (g1 - g0) * (hf_ratio / (0.35 + _EPS)), g0, g1))

    kernel = 1.0 - (peak01**gamma)
    kernel = alpha * kernel + (1.0 - alpha)
    return np.clip(kernel, 0.0, 1.0)

# Weight update
def _update_weights(
    residual,
    kernel,
    omega,
    zeta,
    mu_neg,
    sig_neg,
    w_floor,
    w_ceiling,
    eps,
):
    w_new = np.empty_like(residual, dtype=float)

    neg = residual < 0
    pos = ~neg

    if np.any(pos):
        d = residual[pos]
        t = np.clip((d - (mu_neg + 2.0 * sig_neg)) / (sig_neg + eps), -60, 60)
        logistic = 1.0 / (1.0 + np.exp(t))
        w_new[pos] = logistic * (1.0 - omega * (1.0 - kernel[pos]))
    else:
        w_new[pos] = 1.0

    if np.any(neg):
        d = residual[neg]
        t = np.clip((-d) / (sig_neg + eps), -60, 60)
        logistic = 1.0 / (1.0 + np.exp(-t))
        w_new[neg] = (zeta * kernel[neg]) * logistic
    else:
        w_new[neg] = 1.0

    return np.clip(w_new, w_floor, w_ceiling)

# Frequency-Conditioned DIRAS
def FC_DIRAS(
    y,
    lam=1e5,
    sigma_range=(2.0, 40.0),
    hf_frac=0.25,
    hf_range=(0.05, 0.50),
    smooth_power=1.0,
    max_iter=60,
    ar_order=30,
    alpha_ar=0.7,
    kernel_ema=0.12,
    sigma_struct=6.0,
    omega=0.08,
    zeta=2.0,
    beta=0.25,
    w_floor=1e-6,
    w_ceiling=1.0,
    stop_baseline=1e-4,
    stop_weight=1e-3,
    patience=3,
    eps=1e-6,
    kernel_update_every=2,
):
    y = np.asarray(y, float).ravel()
    L = y.size

    if L < 8:
        return np.zeros(L, float)

    hf_measured = _fft_power_ratio(y, hf_start_frac=hf_frac)
    conditioning_strength = _adaptive_sigma(
        hf_measured,
        sigma_range=sigma_range,
        hf_range=hf_range,
        power=smooth_power,
    )
    y_work = _gsmooth(y, conditioning_strength)

    D_base = _get_D_base(L)

    kernel = ar_model_kernel_psd(
        y_work,
        order=ar_order,
        eps=eps,
        alpha=alpha_ar,
        sigma=sigma_struct,
    )
    w = np.clip(kernel, w_floor, w_ceiling)

    baseline_old = np.zeros(L, float)
    best_baseline = None
    best_obj = np.inf
    stable = 0

    for it in range(int(max_iter)):
        W = diags(w, 0, format="csc")
        baseline = spsolve(W + lam * D_base, w * y_work)
        residual = y_work - baseline

        if (it % kernel_update_every) == 0:
            k_new = ar_model_kernel_psd(
                residual,
                order=ar_order,
                eps=eps,
                alpha=alpha_ar,
                sigma=sigma_struct,
            )
            kernel = np.clip(
                (1.0 - kernel_ema) * kernel + kernel_ema * k_new,
                0.0,
                1.0,
            )

        neg = residual < 0
        if np.any(neg):
            mu_neg = float(np.mean(residual[neg]))
            sig_neg = float(np.std(residual[neg]) + eps)
        else:
            mu_neg = 0.0
            sig_neg = float(_mad(residual) + eps)

        w_new = _update_weights(
            residual=residual,
            kernel=kernel,
            omega=omega,
            zeta=zeta,
            mu_neg=mu_neg,
            sig_neg=sig_neg,
            w_floor=w_floor,
            w_ceiling=w_ceiling,
            eps=eps,
        )
        w_damped = np.clip((1.0 - beta) * w + beta * w_new, w_floor, w_ceiling)

        bchg = float(
            np.linalg.norm(baseline - baseline_old) / (np.linalg.norm(baseline) + eps)
        )
        wchg = float(np.linalg.norm(w_damped - w) / (np.linalg.norm(w) + eps))

        obj = float(np.sum(w_damped * residual**2) + lam * _smoothness_penalty(baseline))

        if obj < best_obj:
            best_obj = obj
            best_baseline = baseline.copy()

        if (bchg < stop_baseline) and (wchg < stop_weight):
            stable += 1
            if stable >= int(patience):
                break
        else:
            stable = 0

        baseline_old = baseline
        w = w_damped

    if best_baseline is None:
        best_baseline = baseline_old.copy()

    return best_baseline
