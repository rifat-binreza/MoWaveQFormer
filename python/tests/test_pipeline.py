"""CPU checks for model gradients, checkpoint recovery, preprocessing and split leakage."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mowave.model import MoWaveQFormer
from mowave.preprocessing import prepare_ppg
from mowave.data import subject_split
from mowave.losses import ptt_loss
from mowave.training import train_model


class PipelineTests(unittest.TestCase):
    def test_model_checkpoint_and_gradients(self):
        torch.set_num_threads(1)
        model=MoWaveQFormer()
        self.assertEqual(sum(p.numel() for p in model.parameters()),816445)
        self.assertEqual(sum(p.numel() for p in model.wavelet.parameters()),768)
        x=torch.randn(3,1,300); groups=torch.arange(3); q=torch.tensor([0.,1.,0.])
        model.eval(); expected=model(x,groups,q).detach()
        (model(x,groups,q).abs().mean()+ptt_loss(model(x,groups,q),torch.ones(3)*.3,torch.ones(3,dtype=torch.bool))).backward()
        self.assertTrue(torch.isfinite(model.wavelet.filters.grad).all())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.pt'; torch.save(model.state_dict(),path)
            restored=MoWaveQFormer(); restored.load_state_dict(torch.load(path,weights_only=True)); restored.eval()
            torch.testing.assert_close(expected,restored(x,groups,q))

    def test_signal_and_subject_isolation(self):
        signal=np.sin(2*np.pi*1.2*np.arange(300)/30)
        normalized,clean=prepare_ppg(signal)
        self.assertEqual(normalized.shape,(300,)); self.assertTrue(np.isfinite(clean).all())
        self.assertLessEqual(float(normalized.max()),1.)
        df=pd.DataFrame({'base_id':[f'{i:06d}' for i in range(20)],'subject_id':np.repeat(np.arange(10),2)})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'split.json'; first=subject_split(df,path)
            self.assertEqual(first,subject_split(df,path))
            bad=df.copy(); bad['subject_id']=1
            with self.assertRaises(ValueError): subject_split(bad,path)

    def test_one_epoch_training(self):
        torch.set_num_threads(1)
        records=[dict(ppg=torch.randn(1,300),motion_group=torch.tensor(i%3),quality=torch.tensor(float(i%2)),
                      ecg_hr=torch.tensor(75.),ptt=torch.tensor(.3),ptt_valid=torch.tensor(True),
                      record_id=str(i),subject_id=i) for i in range(4)]
        loader=DataLoader(records,batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            history=train_model(MoWaveQFormer(d_model=16,n_heads=4,n_layers=1),loader,loader,directory,'cpu',epochs=1)
            self.assertEqual(len(history),1)
            self.assertTrue((Path(directory)/'best.pt').exists())


if __name__=='__main__': unittest.main()
