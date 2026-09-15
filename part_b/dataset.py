import numpy as np

MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]

def make_symbols(mod, n, rng):
    if mod == "BPSK":
        return (2 * rng.integers(0, 2, n) - 1).astype(complex)
    if mod == "QPSK":
        return np.exp(1j * (np.pi/4 + rng.integers(0, 4, n) * np.pi/2))
    if mod == "8PSK":
        return np.exp(1j * (rng.integers(0, 8, n) * np.pi/4))
    if mod == "16QAM":
        lv = np.array([-3, -1, 1, 3])
        return (rng.choice(lv, n) + 1j * rng.choice(lv, n)) / np.sqrt(10)
    if mod == "64QAM":
        lv = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        return (rng.choice(lv, n) + 1j * rng.choice(lv, n)) / np.sqrt(42)

def generate_amc_dataset(n_per_class_per_snr=100, n_symbols=128, snrs=range(-10, 21, 2), seed=0):
    rng = np.random.default_rng(seed)
    X, y, snr_list = [], [], []
    for snr_db in snrs:
        for k, mod in enumerate(MODULATIONS):
            for _ in range(n_per_class_per_snr):
                s = make_symbols(mod, n_symbols, rng)
                phase_offset = np.exp(1j * rng.uniform(0, 2*np.pi)) # unknown phase
                s = s * phase_offset
                
                sig_pow = np.mean(np.abs(s)**2)
                noise_pow = sig_pow / (10**(snr_db/10))
                
                w = np.sqrt(noise_pow/2) * (rng.normal(size=n_symbols) + 1j * rng.normal(size=n_symbols))
                r = s + w
                
                X.append(np.stack([r.real, r.imag])) # shape (2, n_symbols): I and Q
                y.append(k)
                snr_list.append(snr_db)
                
    return np.array(X), np.array(y), np.array(snr_list)