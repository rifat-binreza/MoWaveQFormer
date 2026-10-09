"""Predict HR from a one-column PPG CSV using a locally trained checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mowave.model import MoWaveQFormer
from mowave.preprocessing import prepare_ppg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ppg',required=True,type=Path,help='Headerless single-column CSV at 30 Hz')
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--motion-group',required=True,type=int,choices=(0,1,2))
    parser.add_argument('--quality',required=True,type=int,choices=(0,1))
    args=parser.parse_args()
    raw=np.loadtxt(args.ppg,delimiter=',')
    if raw.shape != (300,):
        parser.error('Provide exactly 300 samples (10 seconds at 30 Hz)')
    ppg,_=prepare_ppg(raw)
    model=MoWaveQFormer(**json.loads((args.run/'model_config.json').read_text()))
    model.load_state_dict(torch.load(args.run/'best.pt',map_location='cpu',weights_only=True)); model.eval()
    with torch.no_grad():
        result=model(torch.tensor(ppg).view(1,1,300),torch.tensor([args.motion_group]),torch.tensor([float(args.quality)]))
    print(json.dumps({'heart_rate_bpm':float(result[0])}))


if __name__=='__main__': main()
