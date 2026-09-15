import numpy as np
from scipy.stats import kurtosis, skew

def extract_features(X_raw):
    """
    Expects X_raw of shape (N, 2, 128) representing I and Q.
    Returns feature matrix of shape (N, num_features).
    """
    N = X_raw.shape[0]
    features = []
    
    for i in range(N):
        # Reconstruct complex signal
        r = X_raw[i, 0, :] + 1j * X_raw[i, 1, :]
        
        # Normalize signal to unit average power[cite: 2]
        r_norm = r / np.sqrt(np.mean(np.abs(r)**2))
        
        # Instantaneous amplitude statistics[cite: 2]
        a = np.abs(r_norm)
        a_std = np.std(a)
        a_mean_sq = np.mean(a**2)
        a_kurtosis = kurtosis(a)
        a_skewness = skew(a)
        
        # Instantaneous phase statistics[cite: 2]
        phase = np.angle(r_norm)
        phase_std = np.std(phase)
        phase_mean_abs = np.mean(np.abs(phase))
        
        # Higher-order cumulants[cite: 2]
        C20 = np.mean(r_norm**2)
        C21 = np.mean(np.abs(r_norm)**2)
        C40 = np.mean(r_norm**4) - 3 * (C20**2)
        C42 = np.mean(np.abs(r_norm)**4) - (np.abs(C20)**2) - 2 * (C21**2)
        
        # Cumulant magnitudes and normalized ratios[cite: 2]
        c20_mag = np.abs(C20)
        c40_mag = np.abs(C40)
        c42_mag = np.abs(C42)
        ratio_40_21 = c40_mag / (C21**2 + 1e-12)
        ratio_42_21 = c42_mag / (C21**2 + 1e-12)
        
        # Compile feature vector
        feat_vec = [
            a_std, a_mean_sq, a_kurtosis, a_skewness,
            phase_std, phase_mean_abs,
            c20_mag, c40_mag, c42_mag, ratio_40_21, ratio_42_21
        ]
        features.append(feat_vec)
        
    return np.array(features)