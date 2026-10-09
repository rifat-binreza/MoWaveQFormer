"""Train MoWaveQFormer on a local BUT PPG dataset; no dataset download or weights bundled."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mowave.data import BUTPPGDataset, prepare_metadata, subject_split
from mowave.model import MoWaveQFormer
from mowave.training import train_model, seed_worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--output',type=Path,default=Path('runs/mowave'))
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--fixed-wavelet',action='store_true')
    parser.add_argument('--no-quality-gate',action='store_true')
    parser.add_argument('--no-ptt',action='store_true')
    args=parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch size must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df=pd.read_csv(args.metadata,dtype={'base_id':str}) if args.metadata else prepare_metadata(args.data_root)
    args.output.mkdir(parents=True,exist_ok=True)
    splits=subject_split(df,args.output/'split_ids.json',args.seed)
    df.to_csv(args.output/'master_ann.csv',index=False)
    loaders=[DataLoader(BUTPPGDataset(args.data_root,splits[k],df,augment=k=='tr_ids'),
              batch_size=args.batch_size,shuffle=k=='tr_ids',worker_init_fn=seed_worker)
              for k in ('tr_ids','va_ids')]
    configuration=dict(fixed_wavelet=args.fixed_wavelet,no_quality_gate=args.no_quality_gate,no_ptt=args.no_ptt)
    (args.output/'model_config.json').write_text(json.dumps(configuration,indent=2))
    model=MoWaveQFormer(**configuration).to(args.device)
    train_model(model,*loaders,args.output,args.device,epochs=args.epochs)


if __name__=='__main__':
    main()
