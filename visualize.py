"""Figures for the report.

  # Problem 1.4: one training image per category (image | mask pairs, 10 images per row)
  python visualize.py samples --data_root ./datasets/data

  # Problem 3.3 / 3.4: mIoU-per-epoch curves (reads logs/<exp_name>_miou.json written by main.py)
  python visualize.py curves --logs logs/deeplabv3plus_resnet50_cross_entropy_miou.json \
                                    logs/deeplabv3plus_resnet50_focal_loss_miou.json \
                             --names "Cross-Entropy" "Focal Loss"

  # Problem 3.5 (4 columns) and Problem 4 (5 columns, add --sam_dir)
  python visualize.py compare --data_root ./datasets/data \
        --ckpt1 checkpoints/best_deeplabv3plus_resnet50_cross_entropy_VOC_os16.pth \
        --ckpt2 checkpoints/best_deeplabv3plus_resnet50_focal_loss_VOC_os16.pth \
        [--sam_dir sam_results]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets import VOCSegmentation
from vis_utils import VOC_CLASSES

FIG_DIR = os.path.join('report', 'figs')
IDS_FILE = os.path.join('report', 'sample_ids.json')


def pick_samples(dst):
    """For every object class pick the training image where that class covers the largest
    fraction of the image (clear, easy-to-read example); avoid re-using an image."""
    n_cls = 21
    frac = np.zeros((len(dst), n_cls))
    for i, mpath in enumerate(dst.masks):
        m = np.array(Image.open(mpath))
        frac[i] = np.bincount(m[m != 255].ravel(), minlength=n_cls)[:n_cls] / m.size
    chosen, used = {}, set()
    for c in list(range(1, n_cls)) + [0]:
        for i in np.argsort(-frac[:, c]):
            if i not in used:
                chosen[VOC_CLASSES[c]] = int(i)
                used.add(i)
                break
    return chosen


def cmd_samples(args):
    dst = VOCSegmentation(args.data_root, image_set='train', transform=None)
    chosen = pick_samples(dst)
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(IDS_FILE, 'w') as f:
        json.dump(chosen, f, indent=2)

    pairs_per_row = 5  # 5 (image, mask) pairs = 10 images per row
    rows = int(np.ceil(len(chosen) / pairs_per_row))
    fig, axes = plt.subplots(rows, 2 * pairs_per_row, figsize=(2 * pairs_per_row * 2.0, rows * 1.9))
    for ax in axes.ravel():
        ax.axis('off')
    for k, (name, idx) in enumerate(chosen.items()):
        img, mask = dst[idx]
        r, c = divmod(k, pairs_per_row)
        axes[r, 2 * c].imshow(img)
        axes[r, 2 * c].set_title(name, fontsize=20)
        axes[r, 2 * c + 1].imshow(dst.decode_target(np.array(mask)).astype(np.uint8), interpolation='nearest')
        axes[r, 2 * c + 1].set_title('GT', fontsize=20)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'samples_per_class.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print('saved', out)
    print(json.dumps(chosen, indent=2))


def cmd_curves(args):
    os.makedirs(FIG_DIR, exist_ok=True)
    names = args.names or [os.path.basename(p) for p in args.logs]
    logs = [json.load(open(p)) for p in args.logs]
    for name, log, fn in zip(names, logs, args.out_names):
        plt.figure(figsize=(5, 3.4))
        ep = [d['epoch'] for d in log]
        miou = [100 * d['mIoU'] for d in log]
        plt.plot(ep, miou, marker='o', ms=3)
        b = int(np.argmax(miou))
        plt.scatter([ep[b]], [miou[b]], color='red', zorder=3, label='best %.2f (epoch %d)' % (miou[b], ep[b]))
        plt.xlabel('epoch'); plt.ylabel('val mIoU (%)'); plt.title('DeepLabV3+ R50, ' + name)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, fn), dpi=150)
        plt.close()
        print('%s: best mIoU %.2f at epoch %d, acc at best %.2f' % (name, miou[b], ep[b], 100 * log[b]['acc']))
    plt.figure(figsize=(5, 3.4))
    for name, log in zip(names, logs):
        plt.plot([d['epoch'] for d in log], [100 * d['mIoU'] for d in log], marker='o', ms=3, label=name)
    plt.xlabel('epoch'); plt.ylabel('val mIoU (%)'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'miou_both.png'), dpi=150)
    print('saved curves to', FIG_DIR)


def load_model(path, device):
    import torch
    import network
    torch.set_num_threads(4)
    model = network.modeling.deeplabv3plus_resnet50(num_classes=21, output_stride=16,
                                                    pretrained_backbone=False)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval()


def predict(model, img, device):
    import torch
    from torchvision.transforms.functional import to_tensor, normalize
    x = normalize(to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).argmax(1)[0].cpu().numpy()


def cmd_compare(args):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dst = VOCSegmentation(args.data_root, image_set='train', transform=None)
    chosen = json.load(open(IDS_FILE))
    names = args.classes or list(chosen.keys())[:5]
    m1, m2 = load_model(args.ckpt1, device), load_model(args.ckpt2, device)

    cols = ['RGB Image', 'Ground Truth', 'Model1 (CE)', 'Model2 (Focal)']
    if args.sam_dir:
        cols.append('SAM')
    fig, axes = plt.subplots(len(names), len(cols), figsize=(3 * len(cols), 2.4 * len(names)))
    for r, name in enumerate(names):
        idx = chosen[name]
        img, mask = dst[idx]
        panels = [np.array(img),
                  dst.decode_target(np.array(mask)),
                  dst.decode_target(predict(m1, img, device)),
                  dst.decode_target(predict(m2, img, device))]
        if args.sam_dir:
            stem = os.path.splitext(os.path.basename(dst.images[idx]))[0]
            panels.append(np.array(Image.open(os.path.join(args.sam_dir, stem + '.png')).convert('RGB')))
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p.astype(np.uint8), interpolation='nearest'); ax.axis('off')
            if r == 0:
                ax.set_title(cols[c], fontsize=14)
        axes[r, 0].text(-8, p.shape[0] / 2, name, rotation=90, va='center', ha='right', fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'compare_sam.png' if args.sam_dir else 'compare.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    # also list the image files so SAM can be run on exactly the same images
    with open(os.path.join('report', 'compare_images.txt'), 'w') as f:
        f.write('\n'.join(dst.images[chosen[n]] for n in names) + '\n')
    print('saved', out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('samples'); s.add_argument('--data_root', default='./datasets/data')
    s = sub.add_parser('curves')
    s.add_argument('--logs', nargs='+', required=True)
    s.add_argument('--names', nargs='+')
    s.add_argument('--out_names', nargs='+', default=['miou_ce.png', 'miou_focal.png'])
    s = sub.add_parser('compare')
    s.add_argument('--data_root', default='./datasets/data')
    s.add_argument('--ckpt1', required=True); s.add_argument('--ckpt2', required=True)
    s.add_argument('--classes', nargs='+', help='5 class names from report/sample_ids.json')
    s.add_argument('--sam_dir', default=None)
    args = ap.parse_args()
    {'samples': cmd_samples, 'curves': cmd_curves, 'compare': cmd_compare}[args.cmd](args)


if __name__ == '__main__':
    main()
