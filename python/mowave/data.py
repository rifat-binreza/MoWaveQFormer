"""WFDB record loading and validated subject-level splitting.

Unlike the notebook, unreadable PPG and missing labels raise errors rather than
silently substituting a zero signal or an invented HR target.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from .config import FS_PPG, FS_ECG, MOTION_TO_GROUP
from .preprocessing import prepare_ppg, detect_ppg_peaks, compute_ptt

def clean_cols(df):
    df.columns = [c.strip().lower().replace(" ","_").replace("/","_")
                  .replace("[","").replace("]","").replace("%","pct")
                  for c in df.columns]
    return df

def clean_motion(val):
    if pd.isna(val): return np.nan
    return int(str(val).split(";")[0].strip())

def prepare_metadata(root):
    root = Path(root)
    ann = clean_cols(pd.read_csv(root / 'quality-hr-ann.csv'))
    subjects = clean_cols(pd.read_csv(root / 'subject-info.csv'))
    subjects['motion'] = subjects['motion'].apply(clean_motion)
    for frame in (ann, subjects):
        frame['id'] = frame['id'].astype(int)
    df = ann.merge(subjects[['id', 'motion']], on='id', how='left', validate='one_to_one')
    df['base_id'] = df['id'].astype(str).str.zfill(6)
    blocks = df['id'] // 1000
    mapping = {b:i+1 for i,b in enumerate(sorted(blocks.unique()))}
    df['subject_id'] = blocks.map(mapping)
    if df[['quality','hr','motion']].isna().any().any():
        raise ValueError('Missing reference labels in merged metadata')
    return df


def subject_split(df, path, seed=42):
    """Persist 70/15/remainder split; reject overlapping subjects or records."""
    path = Path(path)
    if path.exists():
        splits = json.loads(path.read_text())
    else:
        subjects = np.array(sorted(df.subject_id.unique()))
        np.random.RandomState(seed).shuffle(subjects)
        a, b = int(.7*len(subjects)), int(.15*len(subjects))
        groups = (subjects[:a], subjects[a:a+b], subjects[a+b:])
        splits = {key:df.loc[df.subject_id.isin(group),'base_id'].tolist()
                  for key,group in zip(('tr_ids','va_ids','te_ids'),groups)}
    lookup = df.set_index('base_id')['subject_id'].to_dict()
    seen_records, seen_subjects = set(), set()
    for key in ('tr_ids','va_ids','te_ids'):
        ids = splits[key]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= lookup.keys():
            raise ValueError(f'Invalid or empty partition: {key}')
        subjects = {lookup[r] for r in ids}
        if seen_records.intersection(ids) or seen_subjects.intersection(subjects):
            raise ValueError('Subject or record leakage between partitions')
        seen_records.update(ids); seen_subjects.update(subjects)
    if seen_records != set(lookup):
        raise ValueError('Split does not cover the complete metadata')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))
    return splits


class BUTPPGDataset(Dataset):
    """One record per 10-second window, with optional reference PTT supervision."""
    def __init__(self, root, record_ids, df, augment=False):
        self.root, self.ids = Path(root), list(record_ids)
        self.df, self.augment = df.set_index('base_id'), augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        rid = self.ids[index]
        raw = wfdb.rdrecord(str(self.root/rid/f'{rid}_PPG')).p_signal
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        ppg, clean = prepare_ppg(raw[:,0])
        row = self.df.loc[rid]
        hr, quality, motion = float(row.hr), float(row.quality), int(row.motion)
        if not np.isfinite(hr) or hr <= 10 or quality not in (0,1) or motion not in MOTION_TO_GROUP:
            raise ValueError(f'Invalid reference labels: {rid}')
        ptt, valid = .3147, False
        if (self.root/rid/f'{rid}.qrs').exists():
            peaks = wfdb.rdann(str(self.root/rid/rid), 'qrs').sample
            delays = compute_ptt(peaks, detect_ppg_peaks(clean))
            if len(delays):
                ptt, valid = float(delays.mean()), True
        if self.augment:
            if np.random.rand() > .5:
                ppg += np.random.normal(0,.02,ppg.shape).astype(np.float32)
            if np.random.rand() > .5:
                ppg = np.roll(ppg,np.random.randint(-10,10))
        return dict(ppg=torch.tensor(ppg.clip(-1,1)).unsqueeze(0),
                    ecg_hr=torch.tensor(hr), quality=torch.tensor(quality),
                    motion_group=torch.tensor(MOTION_TO_GROUP[motion]),
                    ptt=torch.tensor(ptt), ptt_valid=torch.tensor(valid),
                    record_id=rid, subject_id=int(row.subject_id))
