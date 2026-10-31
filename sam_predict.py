"""Problem 4: run Segment-Anything (automatic mask generation) on the 5 comparison images.

Setup (GPU machine or Colab):
  pip install -r requirements_sam.txt
  Download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  into checkpoints/.

Usage:
  python sam_predict.py --image_list report/compare_images.txt
Then:
  python visualize.py compare ... --sam_dir sam_results
"""
import argparse
import os
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def masks_to_color(masks, shape, seed=0):
    """Paint every SAM mask with a random color, largest first so small masks stay visible.
    Pixels not covered by any mask stay black."""
    rng = np.random.RandomState(seed)
    out = np.zeros((*shape, 3), dtype=np.uint8)
    for m in sorted(masks, key=lambda d: d['area'], reverse=True):
        out[m['segmentation']] = rng.randint(40, 256, size=3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+')
    ap.add_argument('--image_list', default='report/compare_images.txt')
    ap.add_argument('--ckpt', default='checkpoints/sam_vit_b_01ec64.pth')
    ap.add_argument('--model_type', default='vit_b', choices=['vit_b','vit_l','vit_h'])
    ap.add_argument('--points_per_batch', type=int, default=16)
    ap.add_argument('--out_dir', default='sam_results')
    args = ap.parse_args()
    torch.set_num_threads(4)
    paths=args.images or Path(args.image_list).read_text().splitlines()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt).to(device)
    sam.eval()
    generator = SamAutomaticMaskGenerator(sam,points_per_side=32,
        points_per_batch=args.points_per_batch,pred_iou_thresh=.88,
        stability_score_thresh=.95,crop_n_layers=0)
    metadata=dict(model=args.model_type,device=device,precision='FP32',points_per_side=32,
        points_per_batch=args.points_per_batch,pred_iou_thresh=.88,stability_score_thresh=.95,
        crop_n_layers=0,checkpoint_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        checkpoint_url='https://dl.fbaipublicfiles.com/segment_anything/'+dict(vit_b='sam_vit_b_01ec64.pth',vit_l='sam_vit_l_0b3195.pth',vit_h='sam_vit_h_4b8939.pth')[args.model_type],
        sam_commit='dca509fe793f601edb92606367a655c15ac00fdf',images=[])

    os.makedirs(args.out_dir, exist_ok=True)
    for path in paths:
        img = np.array(Image.open(path).convert('RGB'))
        started=time.perf_counter()
        with torch.inference_mode():
            masks = generator.generate(img)
        color = masks_to_color(masks, img.shape[:2])
        stem = os.path.splitext(os.path.basename(path))[0]
        np.savez_compressed(os.path.join(args.out_dir,stem+'_masks.npz'),
            masks=np.stack([m['segmentation'] for m in masks]) if masks else np.zeros((0,*img.shape[:2]),dtype=bool))
        details=[{k:v for k,v in m.items() if k!='segmentation'} for m in masks]
        Path(args.out_dir,stem+'_masks.json').write_text(json.dumps(details,indent=2))
        metadata['images'].append(dict(image=path,stem=stem,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            masks=len(masks),seconds=time.perf_counter()-started,height=img.shape[0],width=img.shape[1]))
        Image.fromarray(color).save(os.path.join(args.out_dir, stem + '.png'))
        # overlay version, nicer for the report if preferred
        Image.fromarray((0.45 * img + 0.55 * color).astype(np.uint8)).save(
            os.path.join(args.out_dir, stem + '_overlay.png'))
        print('%s: %d masks' % (stem, len(masks)),flush=True)
    metadata['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0
    Path('report/sam_results.json').write_text(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
