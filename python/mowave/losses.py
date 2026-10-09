"""Notebook PTT auxiliary loss: 0.35 * RR, valid-target masking."""
import torch

def ptt_loss(hr_pred,ptt_target,ptt_valid):
    if ptt_valid.sum()==0:
        return torch.tensor(0.,device=hr_pred.device)
    rr=60.0/(hr_pred.clamp(30,220)+1e-6)
    ptt_pred=rr*0.35
    return (ptt_pred[ptt_valid]-ptt_target[ptt_valid]).abs().mean()
