"""Problem 1.2: class statistics of the PascalVOC 2012 *train* segmentation split.

Outputs
  report/class_stats.tex      LaTeX table (pixel share + #images per class) for the report
  report/class_stats.json     raw numbers
  class_weights.npy           per-class alpha for the focal loss (Problem 3.4)

Usage:  python dataset_stats.py --data_root ./datasets/data
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import VOCSegmentation
from vis_utils import VOC_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='./datasets/data')
    ap.add_argument('--out_dir', default='report')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dst = VOCSegmentation(args.data_root, image_set='train', transform=None)
    n_cls = 21
    pix = np.zeros(n_cls, dtype=np.int64)
    imgs = np.zeros(n_cls, dtype=np.int64)
    for mpath in tqdm(dst.masks):
        m = np.array(Image.open(mpath))
        counts = np.bincount(m[m != 255].ravel(), minlength=n_cls)[:n_cls]
        pix += counts
        imgs += counts > 0

    share = pix / pix.sum()
    # alpha_c ∝ 1/sqrt(freq_c), normalised so that the pixel-weighted mean of alpha is 1
    # (keeps the overall loss scale comparable to CE). sqrt tempers the weights — pure
    # inverse frequency would give background ~50x less weight than rare classes.
    alpha = 1.0 / np.sqrt(np.maximum(share, 1e-8))
    alpha = alpha / (alpha * share).sum()
    np.save('class_weights.npy', alpha.astype(np.float32))

    order = np.argsort(-share)
    with open(os.path.join(args.out_dir, 'class_stats.json'), 'w') as f:
        json.dump({VOC_CLASSES[c]: {'pixel_share': float(share[c]), 'num_images': int(imgs[c]),
                                    'alpha': float(alpha[c])} for c in range(n_cls)}, f, indent=2)

    lines = [r'\begin{tabular}{lrr|lrr}', r'\hline',
             r'Class & Pixel \% & \#Img & Class & Pixel \% & \#Img \\', r'\hline']
    half = (n_cls + 1) // 2
    for i in range(half):
        row = []
        for c in ([order[i], order[i + half]] if i + half < n_cls else [order[i]]):
            row += [VOC_CLASSES[c], '%.2f' % (100 * share[c]), '%d' % imgs[c]]
        if len(row) == 3:
            row += ['', '', '']
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    with open(os.path.join(args.out_dir, 'class_stats.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('train images: %d' % len(dst))
    for c in order:
        print('%-12s pixel %6.2f%%  images %4d  alpha %.3f' % (VOC_CLASSES[c], 100 * share[c], imgs[c], alpha[c]))
    print('max/min pixel ratio (excl. background): %.1f' %
          (share[1:].max() / share[1:].min()))
    print('background / mean object class pixel ratio: %.1f' % (share[0] / share[1:].mean()))


if __name__ == '__main__':
    main()
