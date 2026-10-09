"""Learned FIR bank and patch-quality-gated Transformer.

Parameter names and forward operations retain the supplied notebook structure.
Inputs: PPG [B,1,300], annotation group [B], binary quality [B].
Output: heart rate [B], in bpm. `hard_discard` is a legacy unused flag.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .config import N_GROUPS, PPG_LEN

class MotionWaveletBank(nn.Module):
    def __init__(self,n_groups=N_GROUPS,K=8,kernel_size=31,fixed=False):
        super().__init__()
        self.K=K; self.pad=kernel_size//2; self.fixed=fixed
        n_sets = 1 if fixed else n_groups   # ablation: one shared filter set instead of per-group
        self.filters=nn.Parameter(torch.randn(n_sets,K,1,kernel_size)*0.02)
        self.gain=nn.Parameter(torch.ones(n_sets,K))

    def forward(self,ppg,motion_group):
        B,_,T=ppg.shape; outs=[]
        for b in range(B):
            g = 0 if self.fixed else motion_group[b].item()
            flt=self.filters[g]
            flt=flt/(flt.norm(dim=-1,keepdim=True)+1e-8)
            out=F.conv1d(ppg[b:b+1],flt,padding=self.pad)
            gain=torch.sigmoid(self.gain[g])
            out=out*gain.view(1,self.K,1)
            outs.append(out)
        return torch.cat(outs,dim=0)

class QualityGatedTransformer(nn.Module):
    """Quality-gated Transformer encoder (Stage 3)."""
    def __init__(self,K=8,T=PPG_LEN,d_model=128,
                 n_heads=4,n_layers=4,dropout=0.1,
                 hard_discard=False,no_quality_gate=False):
        super().__init__()
        self.patch_size=10; n_patches=T//self.patch_size
        self.hard_discard=hard_discard
        self.no_quality_gate=no_quality_gate
        self.input_proj=nn.Linear(K*self.patch_size,d_model)
        self.pos_enc=nn.Parameter(
            torch.randn(1,n_patches,d_model)*0.02)
        el=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,
            dim_feedforward=d_model*4,dropout=dropout,
            batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(el,num_layers=n_layers)
        self.quality_gate=nn.Linear(1,n_patches)
        self.hr_head=nn.Sequential(
            nn.Linear(d_model,64),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(64,1))

    def forward(self,sub_bands,quality):
        B,K,T=sub_bands.shape; n_p=T//self.patch_size
        x=sub_bands[:,:,:n_p*self.patch_size]
        x=x.reshape(B,K,n_p,self.patch_size)
        x=x.permute(0,2,1,3).reshape(B,n_p,K*self.patch_size)
        x=self.input_proj(x)+self.pos_enc
        if not self.no_quality_gate:
            gate=torch.sigmoid(self.quality_gate(quality.unsqueeze(1)))
            x=x*gate.unsqueeze(-1)
        x=self.transformer(x).mean(dim=1)
        return self.hr_head(x).squeeze(-1)

class MoWaveNet(nn.Module):
    def __init__(self,K=8,d_model=128,n_heads=4,n_layers=4,
                 fixed_wavelet=False,no_quality_gate=False,no_ptt=False):
        super().__init__()
        self.no_ptt=no_ptt
        self.wavelet=MotionWaveletBank(N_GROUPS,K,fixed=fixed_wavelet)
        self.transformer=QualityGatedTransformer(
            K,PPG_LEN,d_model,n_heads,n_layers,
            no_quality_gate=no_quality_gate)

    def forward(self,ppg,motion_group,quality):
        sub_bands=self.wavelet(ppg,motion_group)
        hr_pred=self.transformer(sub_bands,quality)
        return hr_pred

MoWaveQFormer = MoWaveNet
