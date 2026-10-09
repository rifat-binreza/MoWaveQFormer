"""Optional ACC rule route; published training uses activity annotations.
Absolute energy thresholds depend on sensor units and record duration.
"""
import numpy as np
from .config import FS_ACC, MOTION_TO_GROUP

def detect_motion_group(acc, fs=FS_ACC):
    """
    Rule-based conditioner. Returns group index 0/1/2.
    Works even for subtle motions by using annotation label directly
    when available (training). At inference uses ACC signal rules.

    Rules derived from physiology:
      Walking : dominant freq in [1.0, 2.5] Hz with high energy
      Burst   : energy in [4, 15] Hz > 2× energy in [0.5, 4] Hz
      Subtle  : everything else
    """
    if acc is None:
        return 0  # no ACC → treat as subtle

    mag = np.sqrt(np.sum(acc**2, axis=1))
    mag = mag - mag.mean()   # remove gravity component

    fft   = np.abs(np.fft.rfft(mag))
    freqs = np.fft.rfftfreq(len(mag), d=1.0/fs)

    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.sum(fft[mask]**2))

    e_walk  = band_energy(1.0, 2.5)   # step frequency
    e_low   = band_energy(0.5, 4.0)   # general low motion
    e_burst = band_energy(4.0, 15.0)  # coughing / laughing

    # Walking: strong periodic step energy
    total = e_low + e_burst + 1e-8
    if e_walk / total > 0.35 and e_walk > 500:
        return 1  # walking

    # Burst: coughing/laughing have high-freq transient energy
    if e_burst > 2.0 * e_low and e_burst > 200:
        return 2  # burst

    return 0

def acc_rms_energy(acc):
    """Scalar RMS of 3-axis ACC magnitude."""
    mag=np.sqrt(np.sum(acc**2, axis=1))
    return float(np.sqrt(np.mean(mag**2)))

def acc_spectral_entropy(acc, fs=FS_ACC):
    """Normalised spectral entropy of ACC magnitude (0=periodic, 1=random)."""
    mag=np.sqrt(np.sum(acc**2, axis=1))
    fft_mag=np.abs(np.fft.rfft(mag))
    psd=fft_mag**2; psd_norm=psd/(psd.sum()+1e-12)
    ent=-np.sum(psd_norm*np.log(psd_norm+1e-12))
    return float(ent/np.log(len(psd_norm)+1e-12))

def acc_dominant_freq(acc, fs=FS_ACC):
    """Dominant frequency of ACC magnitude (Hz) — step freq for walking."""
    mag=np.sqrt(np.sum(acc**2, axis=1))
    fft_mag=np.abs(np.fft.rfft(mag))
    freq=np.fft.rfftfreq(len(mag), d=1.0/fs)
    valid=(freq>=0.5)&(freq<=5.0)
    if valid.any():
        return float(freq[valid][np.argmax(fft_mag[valid])])
    return np.nan
