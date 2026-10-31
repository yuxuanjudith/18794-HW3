"""Reload a checkpoint and evaluate all VOC validation images, retaining error counts."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import network
from main import atomic_json, get_train_val_dataset, validate
from metrics import StreamSegMetrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt',required=True)
    parser.add_argument('--exp_name',required=True)
    parser.add_argument('--data_root',default='./datasets/data')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.ckpt,map_location='cpu',weights_only=True)
    config=checkpoint['config']
    opts=SimpleNamespace(data_root=args.data_root,crop_val=False,crop_size=config['crop_size'],
                         channels_last=True,val_amp=False,limit_val_batches=0,
                         save_val_results=False,exp_name=args.exp_name)
    _,dataset=get_train_val_dataset(opts)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.workers,
                      pin_memory=device.type=='cuda')
    model=network.modeling.__dict__[config['model']](num_classes=21,
        output_stride=config['output_stride'],pretrained_backbone=False)
    model.load_state_dict(checkpoint['model_state'])
    del checkpoint
    model=model.to(device=device,memory_format=torch.channels_last)
    metrics=StreamSegMetrics(21)
    scores,_=validate(opts,model,loader,device,metrics)
    matrix=metrics.confusion_matrix.astype(np.int64)
    assert scores['validation_images']==1449 and matrix.shape==(21,21)
    # Rows are true labels, columns are predicted labels; all void pixels are excluded.
    scores['Confusion Matrix']=matrix.tolist()
    scores['background_as_foreground_rate']=float(matrix[0,1:].sum()/matrix[0,:].sum())
    scores['foreground_as_background_rate']=float(matrix[1:,0].sum()/matrix[1:,:].sum())
    scores['confusion_matrix_axes']='rows=true class, columns=predicted class; labels 0..20'
    atomic_json(Path('logs')/f'{args.exp_name}_evaluation.json',scores)
    print(metrics.to_str(metrics.get_results()),flush=True)
    print('Background predicted as foreground:',scores['background_as_foreground_rate'],flush=True)
    print('Foreground predicted as background:',scores['foreground_as_background_rate'],flush=True)


if __name__=='__main__':
    main()
