"""Evaluate a saved checkpoint on its subject-independent test partition."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mowave.data import BUTPPGDataset, subject_split
from mowave.model import MoWaveQFormer
from mowave.evaluation import predict, metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    df=pd.read_csv(args.run/'master_ann.csv',dtype={'base_id':str})
    splits=subject_split(df,args.run/'split_ids.json')
    model=MoWaveQFormer(**json.loads((args.run/'model_config.json').read_text())).to(args.device)
    model.load_state_dict(torch.load(args.run/'best.pt',map_location=args.device,weights_only=True))
    frame=predict(model,DataLoader(BUTPPGDataset(args.data_root,splits['te_ids'],df),batch_size=32),args.device)
    frame.to_csv(args.run/'test_predictions.csv',index=False)
    report={'overall':metrics(frame)}
    for key in ('quality','motion_group'):
        report[key]={str(group):metrics(part) for group,part in frame.groupby(key)}
    (args.run/'test_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
