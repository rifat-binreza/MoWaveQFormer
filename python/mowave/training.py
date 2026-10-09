"""Portable notebook training loop with best-validation checkpoint selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .losses import ptt_loss
from .evaluation import predict, metrics


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


def train_model(model, train_loader, val_loader, output, device, epochs=80, lr=3e-4, lambda_ptt=.1):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    warmup=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:min(1.,(epoch+1)/5))
    plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=8,min_lr=1e-6)
    best=float('inf'); history=[]
    for epoch in range(epochs):
        model.train(); errors=[]
        for batch in train_loader:
            ppg, groups, quality, target, ptt, valid = [batch[k].to(device) for k in
                ('ppg','motion_group','quality','ecg_hr','ptt','ptt_valid')]
            optimizer.zero_grad()
            prediction=model(ppg,groups,quality)
            loss=(prediction-target).abs().mean()
            if not model.no_ptt:
                loss=loss+lambda_ptt*ptt_loss(prediction,ptt,valid)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); errors.extend((prediction.detach()-target).abs().cpu().tolist())
        val=metrics(predict(model,val_loader,device))['mae']
        if not np.isfinite(val):
            raise ValueError('Nonfinite validation error')
        if val < best:
            best=val; torch.save(model.state_dict(),output/'best.pt')
        if epoch < 5:
            warmup.step()
        plateau.step(val)
        history.append(dict(epoch=epoch+1,train_mae=float(np.mean(errors)),val_mae=val))
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        print(f'Epoch {epoch+1}: validation MAE={val:.4f} bpm',flush=True)
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True))
    return history
