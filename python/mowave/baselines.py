"""Neural and spectral comparators from the supplied notebook.

`troika_hr` is the notebook's simple spectral comparator, not the full original
TROIKA algorithm. `tapir_hr` is the legacy name for the harmonic-SP comparator.
These implementations alone do not establish reproduction of paper results.
"""
import numpy as np
import torch
from torch import nn
from scipy.signal import butter, filtfilt
from .config import FS_PPG

def troika_hr(ppg,fs=FS_PPG):
    nyq=fs/2; b,a=butter(4,[0.67/nyq,3.0/nyq],btype="band")
    ppg_f=filtfilt(b,a,ppg)
    N=len(ppg_f)
    fft=np.abs(np.fft.rfft(ppg_f,n=N*4))
    freq=np.fft.rfftfreq(N*4,d=1.0/fs)
    mask=(freq>=0.67)&(freq<=3.0)
    fhr=freq[mask]; ffhr=fft[mask]
    if ffhr.max()==0: return np.nan
    return float(fhr[np.argmax(ffhr)]*60.0)

def tapir_hr(ppg, fs=FS_PPG, prev_hr=None):
    """
    Adaptive spectral HR estimation with harmonic reinforcement.
    1. Bandpass filter PPG in HR band [0.67–3.0 Hz]
    2. Compute FFT and reinforce harmonic components (2f, 3f)
    3. If prev_hr known, bias toward continuity (±10 bpm window)
    4. Select dominant spectral peak as HR estimate
    """
    nyq=fs/2; b,a=butter(4,[0.67/nyq,3.0/nyq],btype="band")
    ppg_f=filtfilt(b,a,ppg)
    N=len(ppg_f); NFFT=N*4
    fft=np.abs(np.fft.rfft(ppg_f,n=NFFT))**2   # power spectrum
    freq=np.fft.rfftfreq(NFFT,d=1.0/fs)

    # HR band mask
    mask=(freq>=0.67)&(freq<=3.0)
    fhr=freq[mask]; pfhr=fft[mask].copy()

    # Harmonic reinforcement: add scaled power at 2f and 3f
    for i,f in enumerate(fhr):
        for harm in [2,3]:
            hf=f*harm
            if 0.67<=hf<=3.0:
                hm=np.argmin(np.abs(fhr-hf))
                pfhr[i]+=0.3*pfhr[hm]

    # Continuity bias: if previous HR known, boost nearby frequencies
    if prev_hr is not None:
        prev_f=prev_hr/60.0
        window=10/60.0   # ±10 bpm
        cont_mask=np.abs(fhr-prev_f)<=window
        pfhr[cont_mask]*=1.5

    if pfhr.max()==0: return np.nan
    return float(fhr[np.argmax(pfhr)]*60.0)

class DeepPPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.c=nn.Sequential(
            nn.Conv1d(1,16,7,padding=3),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(16,32,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.h=nn.Sequential(
            nn.Flatten(),nn.Linear(512,128),nn.ReLU(),
            nn.Dropout(0.3),nn.Linear(128,1))
    def forward(self,ppg,**kw):
        return self.h(self.c(ppg)).squeeze(-1)

class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.c=nn.Sequential(
            nn.Conv1d(1,32,7,padding=3),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.MaxPool1d(2))
        self.l=nn.LSTM(64,64,2,bidirectional=True,
                        batch_first=True,dropout=0.3)
        self.h=nn.Sequential(
            nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,ppg,**kw):
        x=self.c(ppg).transpose(1,2)
        x,_=self.l(x)
        return self.h(x.mean(1)).squeeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self,channels,kernel_size=7):
        super().__init__()
        pad=kernel_size//2
        self.conv=nn.Sequential(
            nn.Conv1d(channels,channels,kernel_size,padding=pad),
            nn.BatchNorm1d(channels),nn.ReLU(),
            nn.Conv1d(channels,channels,kernel_size,padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.act=nn.ReLU()
    def forward(self,x):
        return self.act(x+self.conv(x))

class ResNet1D(nn.Module):
    """
    1-D ResNet for HR regression from PPG.
    Architecture inspired by Q-PPG (Burrello et al. 2021).
    4 residual blocks with increasing channel count.
    """
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv1d(1,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU())
        self.blocks=nn.Sequential(
            ResBlock1D(32,7),
            nn.Conv1d(32,64,3,stride=2,padding=1),nn.ReLU(),
            ResBlock1D(64,5),
            nn.Conv1d(64,128,3,stride=2,padding=1),nn.ReLU(),
            ResBlock1D(128,3),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4,64),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(64,1)
        )
    def forward(self,ppg,**kw):
        return self.head(self.blocks(self.stem(ppg))).squeeze(-1)

harmonic_sp_hr = tapir_hr
