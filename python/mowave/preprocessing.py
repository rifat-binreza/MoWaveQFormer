"""Notebook filtering and ECG-to-PPG peak-delay extraction.

The SWT operation is retained verbatim: the last detail coefficients are zeroed.
It is not a replacement or newly validated detrending algorithm.
"""
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks
from .config import FS_PPG, FS_ECG, FS_ACC, PPG_LEN

def bandpass_ppg(sig, fs=FS_PPG):
    nyq=fs/2; b,a=butter(4,[0.5/nyq,4.0/nyq],btype="band")
    return filtfilt(b,a,sig)

def wavelet_detrend(sig, wavelet="db4", level=3):
    n=len(sig); pad=int(2**np.ceil(np.log2(max(n,8))))
    p=np.pad(sig,(0,pad-n),mode="reflect")
    c=pywt.swt(p,wavelet,level=level)
    c[-1]=(c[-1][0],np.zeros_like(c[-1][1]))
    return pywt.iswt(c,wavelet)[:n]

def detect_ppg_peaks(ppg, fs=FS_PPG):
    peaks,_=find_peaks(ppg, distance=int(0.4*fs),
                        prominence=0.01*(ppg.max()-ppg.min()+1e-8))
    return peaks

def compute_ptt(r_peaks, ppg_peaks, fs_ecg=FS_ECG, fs_ppg=FS_PPG):
    r_t=r_peaks/fs_ecg; ppg_t=ppg_peaks/fs_ppg; ptts=[]
    for rt in r_t:
        c=ppg_t[ppg_t>rt]
        if len(c)==0: continue
        v=c[0]-rt
        if 0.05<=v<=0.60: ptts.append(v)
    return np.array(ptts, dtype=np.float32)

def prepare_ppg(raw):
    """Return fixed-length normalized input and unnormalized filtered signal."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 1 or len(raw) < 30 or not np.isfinite(raw).all():
        raise ValueError("PPG must contain at least 30 finite samples")
    clean = bandpass_ppg(wavelet_detrend(raw))
    fixed = np.zeros(PPG_LEN, dtype=np.float32)
    fixed[:min(len(clean), PPG_LEN)] = clean[:PPG_LEN]
    fixed = 2 * (fixed - fixed.min()) / (np.ptp(fixed) + 1e-8) - 1
    return fixed.astype(np.float32), clean
