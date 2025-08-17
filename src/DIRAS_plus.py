# diras_plus.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import Layer
from scipy.interpolate import UnivariateSpline
import joblib

def _smooth(y, x, s=1.0):
    """Smooth data using spline interpolation."""
    return UnivariateSpline(x, y, s=s)(x)

def _torsion_all(x, Y):
    """Compute torsion (curvature) for all spectra."""
    dx = float(np.mean(np.diff(x)))
    out = []
    for y in Y.T:
        d2 = (y[2:] - 2*y[1:-1] + y[:-2]) / (dx**2)
        out.append(np.sum(np.abs(d2)) * dx)
    return np.asarray(out)

def ssim1d_loss(max_val=1.0, filter_size=11, k1=0.01, k2=0.03):
    """1D Structural Similarity Index loss function."""
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    avg = tf.ones((filter_size, 1, 1), tf.float32) / filter_size
    def _loss(y_true, y_pred):
        s = tf.shape(y_true)
        y_true = tf.reshape(y_true, (s[0], s[1], 1))
        y_pred = tf.reshape(y_pred, (s[0], s[1], 1))
        mu1 = tf.nn.conv1d(y_true, avg, 1, 'SAME')
        mu2 = tf.nn.conv1d(y_pred, avg, 1, 'SAME')
        v1 = tf.nn.conv1d(y_true*y_true, avg, 1, 'SAME') - mu1**2
        v2 = tf.nn.conv1d(y_pred*y_pred, avg, 1, 'SAME') - mu2**2
        v12 = tf.nn.conv1d(y_true*y_pred, avg, 1, 'SAME') - mu1*mu2
        num = (2*mu1*mu2 + c1) * (2*v12 + c2)
        den = (mu1**2 + mu2**2 + c1) * (v1 + v2 + c2)
        return 1.0 - tf.reduce_mean(num / den)
    return _loss

class ChannelAverage(Layer):
    """Custom Keras layer for channel averaging."""
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1)

def diras_plus_xgb(
    wavenumber,
    spectra,
    diras_fn,
    encoder_path,
    xgb_path,
    *,
    pre_lam=1e4,
    ar_order=50,
    omega=0.01,
    zeta=2,
    spline_s=1.0,
    pad_len=1536,
    torsion_override=None
):
    """
    DIRAS+ algorithm with XGBoost-based lambda prediction.
    
    Parameters:
    -----------
    wavenumber : array
        Wavenumber axis
    spectra : array
        Spectral data matrix
    diras_fn : function
        DIRAS baseline correction function
    encoder_path : str
        Path to pre-trained encoder model
    xgb_path : str
        Path to pre-trained XGBoost model
    pre_lam : float
        Pre-processing lambda value
    ar_order : int
        Autoregressive model order
    omega : float
        Weight parameter for positive residuals
    zeta : float
        Weight parameter for negative residuals
    spline_s : float
        Spline smoothing parameter
    pad_len : int
        Padding length for encoder input
    torsion_override : array, optional
        Override torsion values
    
    Returns:
    --------
    lam : array
        Predicted lambda values
    baseline : array
        Estimated baselines
    corrected : array
        Baseline-corrected spectra
    """
    n_pts, n_spec = spectra.shape

    # Pre-process with fixed lambda
    pre = np.zeros_like(spectra)
    for i in range(n_spec):
        pre[:, i] = diras_fn(spectra[:, i], lam=pre_lam, ar_order=ar_order, omega=omega, zeta=zeta)

    # Normalize and smooth pre-processed data
    pre_scaled = pre / (np.max(pre, axis=0) + 1e-12)
    sm = np.column_stack([_smooth(pre_scaled[:, i], wavenumber, s=spline_s) for i in range(n_spec)])
    sm /= (np.max(sm, axis=0) + 1e-12)

    # Compute torsion features
    if torsion_override is None:
        torsion = _torsion_all(wavenumber, sm).reshape(-1, 1) * 1e3
    else:
        torsion = np.asarray(torsion_override, dtype=float).reshape(-1, 1)

    # Prepare encoder input by padding to pad_len
    X = spectra.T
    L = pad_len
    Xp = np.vstack([np.pad(x, (0, L - len(x))) if len(x) < L else x[:L] for x in X]).astype(np.float32)
    max_val = float(Xp.max() - Xp.min())

    # Extract features using encoder
    enc = load_model(
        encoder_path,
        custom_objects={'ssim1d_loss': ssim1d_loss(max_val=max_val), 'ChannelAverage': ChannelAverage}
    )
    _, enc_feats, _ = enc.predict(Xp, verbose=0)
    feats = np.concatenate([enc_feats, torsion], axis=1)

    # Predict lambda values using XGBoost
    xgb = joblib.load(xgb_path)
    lam_log = xgb.predict(feats)
    lam = (10.0 ** lam_log)

    # Apply DIRAS with predicted lambda values
    baseline = np.zeros_like(spectra)
    for i in range(n_spec):
        baseline[:, i] = diras_fn(spectra[:, i], lam=lam[i], ar_order=ar_order, omega=omega, zeta=zeta)

    corrected = spectra - baseline
    return lam, baseline, corrected
