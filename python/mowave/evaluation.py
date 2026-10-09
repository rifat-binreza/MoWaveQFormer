"""Record-level predictions and ECG-reference regression metrics."""
import numpy as np
import pandas as pd
import torch


def metrics(frame):
    if frame.empty:
        raise ValueError('Cannot evaluate an empty partition')
    pred, truth = frame.prediction.to_numpy(), frame.reference.to_numpy()
    error = pred - truth
    r = float(np.corrcoef(pred,truth)[0,1]) if len(pred)>1 and np.std(pred)>0 and np.std(truth)>0 else None
    sd = float(np.std(error,ddof=1)) if len(error)>1 else 0.
    return dict(n=len(error), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error**2))), pearson_r=r,
                bias=float(error.mean()), loa_low=float(error.mean()-1.96*sd),
                loa_high=float(error.mean()+1.96*sd))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows=[]
    for batch in loader:
        pred=model(batch['ppg'].to(device),batch['motion_group'].to(device),batch['quality'].to(device)).cpu().numpy()
        for i, value in enumerate(pred):
            rows.append(dict(record_id=batch['record_id'][i],
                             subject_id=int(batch['subject_id'][i]),
                             reference=float(batch['ecg_hr'][i]), prediction=float(value),
                             quality=int(batch['quality'][i]), motion_group=int(batch['motion_group'][i])))
    return pd.DataFrame(rows)
