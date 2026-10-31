"""VOC2012 training for HW3 Problems 3.1--3.5.

A full run uses all 1,464 training images and all 1,449 validation images.
AMP affects training only; validation defaults to FP32 at original image size.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from PIL import Image

import network
import utils
from datasets import VOCSegmentation
from metrics import StreamSegMetrics
from utils import ext_transforms as et


def get_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', default='./datasets/data')
    p.add_argument('--num_classes', type=int, default=21, choices=[21])
    p.add_argument('--model', default='deeplabv3plus_resnet50',
                   choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50'])
    p.add_argument('--output_stride', type=int, default=16, choices=[8, 16])
    p.add_argument('--backbone_weights', default='v2', choices=['v1', 'v2'],
                   help='ImageNet-1K classification initialization for ResNet-50')
    p.add_argument('--total_itrs', type=int, default=5000)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--lr_policy', default='step', choices=['step', 'poly'])
    p.add_argument('--step_size', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--crop_size', type=int, default=513)
    p.add_argument('--crop_val', action='store_true')
    p.add_argument('--val_batch_size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='FP16 training with gradient scaling')
    p.add_argument('--val_amp', action='store_true', help='optional AMP validation; default FP32')
    p.add_argument('--channels_last', action='store_true')
    p.add_argument('--ckpt')
    p.add_argument('--continue_training', action='store_true')
    p.add_argument('--test_only', action='store_true')
    p.add_argument('--save_val_results', action='store_true')
    p.add_argument('--loss_type', default='cross_entropy', choices=['cross_entropy', 'focal_loss'])
    p.add_argument('--focal_gamma', type=float, default=2.)
    p.add_argument('--class_weights')
    p.add_argument('--exp_name')
    p.add_argument('--gpu_id', default='0')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--random_seed', type=int, default=1)
    p.add_argument('--print_interval', type=int, default=25)
    p.add_argument('--limit_val_batches', type=int, default=0,
                   help='smoke tests only; 0 evaluates the entire validation set')
    return p


def get_train_val_dataset(opts):
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop((opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(), et.ExtToTensor(),
        et.ExtNormalize([.485, .456, .406], [.229, .224, .225])])
    val_steps = [et.ExtResize(opts.crop_size), et.ExtCenterCrop(opts.crop_size)] if opts.crop_val else []
    val_transform = et.ExtCompose(val_steps + [et.ExtToTensor(),
        et.ExtNormalize([.485, .456, .406], [.229, .224, .225])])
    return (VOCSegmentation(opts.data_root, 'train', train_transform),
            VOCSegmentation(opts.data_root, 'val', val_transform))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def atomic_json(path, value):
    def finite(item):
        if isinstance(item, dict):
            return {str(k): finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if math.isfinite(item) else None
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(finite(value), indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


@torch.inference_mode()
def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    model.eval()
    # Original-sized validation images vary; avoid benchmarking each new shape.
    torch.backends.cudnn.benchmark = False
    metrics.reset()
    seen = 0
    started = time.perf_counter()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if opts.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=opts.val_amp and device.type == 'cuda'):
            logits = model(images)
        preds = logits.argmax(1).cpu().numpy()
        metrics.update(labels.numpy(), preds)
        if opts.save_val_results:
            out_dir = Path('results') / opts.exp_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for j, pred in enumerate(preds):
                Image.fromarray(loader.dataset.decode_target(pred)).save(out_dir / f'{seen+j:04d}.png')
        seen += len(images)
        if (i + 1) % 400 == 0:
            print(f'Validation {seen}/{len(loader.dataset)} images', flush=True)
        if opts.limit_val_batches and i + 1 >= opts.limit_val_batches:
            break
    score = metrics.get_results()
    score['validation_images'] = seen
    score['seconds'] = time.perf_counter() - started
    return score, []


def main():
    opts = get_argparser().parse_args()
    if opts.total_itrs < 1 or opts.batch_size < 2:
        raise ValueError('Use positive iterations and batch_size >= 2 for ASPP BatchNorm.')
    if opts.continue_training and not opts.ckpt:
        raise ValueError('--continue_training requires --ckpt')
    if opts.test_only and not opts.ckpt:
        raise ValueError('--test_only requires a trained --ckpt')
    opts.exp_name = opts.exp_name or f'{opts.model}_{opts.loss_type}'
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    torch.set_num_threads(opts.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(opts.random_seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    train_dst, val_dst = get_train_val_dataset(opts)
    if not opts.crop_val:
        opts.val_batch_size = 1  # original image sizes differ
    loader_opts = dict(num_workers=opts.workers, pin_memory=device.type == 'cuda',
                       persistent_workers=opts.workers > 0, worker_init_fn=seed_worker)
    # Distinct generators prevent validation from changing training shuffles.
    train_gen = torch.Generator().manual_seed(opts.random_seed)
    val_gen = torch.Generator().manual_seed(opts.random_seed + 1)
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True,
                              drop_last=True, generator=train_gen,
                              **{**loader_opts, 'persistent_workers': False})
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False,
                            generator=val_gen, **loader_opts)
    if not len(train_loader):
        raise ValueError('Training split has fewer images than batch_size.')
    print(f'Device: {device}; train={len(train_dst)}, val={len(val_dst)}; '
          f'iterations/epoch={len(train_loader)}', flush=True)
    model = network.modeling.__dict__[opts.model](num_classes=21,
        output_stride=opts.output_stride, pretrained_backbone=opts.ckpt is None,
        backbone_weights=opts.backbone_weights).to(device)
    utils.set_bn_momentum(model.backbone, momentum=.01)
    if opts.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD([
        {'params': model.backbone.parameters(), 'lr': .1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr}],
        momentum=.9, weight_decay=opts.weight_decay)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, opts.step_size, gamma=.1)
                 if opts.lr_policy == 'step' else utils.PolyLR(optimizer, opts.total_itrs, power=.9))
    alpha = None
    if opts.class_weights:
        alpha = torch.from_numpy(np.load(opts.class_weights)).float()
        if alpha.shape != (21,) or not torch.isfinite(alpha).all() or (alpha <= 0).any():
            raise ValueError('class_weights must contain 21 finite positive weights')
    criterion = (nn.CrossEntropyLoss(ignore_index=255) if opts.loss_type == 'cross_entropy'
                  else utils.FocalLoss(alpha=alpha, gamma=opts.focal_gamma, ignore_index=255)).to(device)
    use_amp = opts.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    metrics = StreamSegMetrics(21)
    best_score, cur_itrs, cur_epochs, history = 0., 0, 0, []
    if opts.ckpt:
        checkpoint = torch.load(opts.ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        if opts.continue_training:
            for name in ['model','output_stride','batch_size','crop_size','lr','lr_policy','step_size',
                         'loss_type','focal_gamma','amp','channels_last','random_seed','backbone_weights',
                         'weight_decay','class_weights']:
                if name in checkpoint.get('config', {}) and getattr(opts, name) != checkpoint['config'][name]:
                    raise ValueError(f'Resume setting changed: {name}')
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            scaler.load_state_dict(checkpoint.get('scaler_state', {}))
            best_score = float(checkpoint['best_score'])
            cur_itrs, cur_epochs = checkpoint['cur_itrs'], checkpoint['cur_epochs']
            history = checkpoint['history']
            # Epoch-boundary recovery; loader seeds are selected by epoch below.
            torch.set_rng_state(checkpoint['torch_rng'])
            random.setstate(checkpoint['python_rng'])
            if device.type == 'cuda' and checkpoint.get('cuda_rng'):
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        print(f'Loaded checkpoint: {opts.ckpt}', flush=True)
        del checkpoint
    if opts.test_only:
        score, _ = validate(opts, model, val_loader, device, metrics)
        atomic_json(f'logs/{opts.exp_name}_evaluation.json', score)
        print(metrics.to_str(score), flush=True)
        return
    config = vars(opts).copy()
    weight_url = network.backbone.resnet.RESNET50_WEIGHT_URLS[opts.backbone_weights]
    weight_file = Path(torch.hub.get_dir())/'checkpoints'/weight_url.rsplit('/', 1)[1]
    config.update(pytorch=str(torch.__version__), torchvision=str(torchvision.__version__),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
        train_images=len(train_dst), val_images=len(val_dst),
        training_precision='AMP FP16' if use_amp else 'FP32',
        evaluation_precision='AMP FP16' if opts.val_amp else 'FP32',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path('main.py'), *sorted(Path('network').rglob('*.py')),
                       Path('utils/loss.py'), Path('utils/ext_transforms.py'), Path('datasets/voc.py')]},
        backbone_weight_url=weight_url,
        backbone_weight_sha256=hashlib.sha256(weight_file.read_bytes()).hexdigest() if weight_file.exists() else None)
    atomic_json(f'logs/{opts.exp_name}_config.json', config)
    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    prefix = f'{opts.exp_name}_VOC_os{opts.output_stride}'
    def save_ckpt(kind):
        state = dict(cur_itrs=cur_itrs, cur_epochs=cur_epochs, model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict(), best_score=best_score, history=history, config=config,
            torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        path = ckpt_dir / f'{kind}_{prefix}.pth'
        tmp = path.with_suffix('.pth.tmp')
        torch.save(state, tmp)
        tmp.replace(path)
    start = time.perf_counter()
    while cur_itrs < opts.total_itrs:
        cur_epochs += 1
        # Use explicit epoch seeds for comparable independent CE/focal runs.
        train_gen.manual_seed(opts.random_seed + cur_epochs)
        model.train()
        torch.backends.cudnn.benchmark = device.type == 'cuda'
        loss_sum, batches = 0., 0
        epoch_start = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if opts.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at iteration {cur_itrs + 1}')
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # A scaler overflow skips an update: do not count it as an optimizer iteration.
            if scaler.get_scale() < old_scale:
                print('Skipped overflow update; scaler reduced', flush=True)
                continue
            scheduler.step()
            cur_itrs += 1
            loss_sum += loss.item()
            batches += 1
            if cur_itrs % opts.print_interval == 0:
                speed = (time.perf_counter() - epoch_start) / batches
                peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0
                print(f'Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, '
                      f'Loss={loss_sum/batches:.5f}, {speed:.3f}s/iter, peak={peak:.2f} GiB', flush=True)
            if cur_itrs >= opts.total_itrs:
                break
        del outputs, loss, images, labels
        train_seconds = time.perf_counter() - epoch_start
        score, _ = validate(opts, model, val_loader, device, metrics)
        row = dict(epoch=cur_epochs, itrs=cur_itrs, mIoU=float(score['Mean IoU']),
            acc=float(score['Overall Acc']),
            class_iou={int(k): float(v) for k, v in score['Class IoU'].items()},
            train_loss=loss_sum/max(batches, 1), lr=[g['lr'] for g in optimizer.param_groups],
            train_seconds=train_seconds, val_seconds=score['seconds'],
            validation_images=score['validation_images'],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0)
        history.append(row)
        improved = row['mIoU'] > best_score
        best_score = max(best_score, row['mIoU'])
        atomic_json(f'logs/{opts.exp_name}_miou.json', history)
        save_ckpt('latest')
        if improved:
            save_ckpt('best')
        print(f'Epoch {cur_epochs}: val mIoU={row["mIoU"]:.6f}, acc={row["acc"]:.6f}, '
              f'best={best_score:.6f}, validation={score["seconds"]:.1f}s', flush=True)
    print(f'Training finished: {cur_itrs} optimizer steps; best mIoU={best_score:.6f}; '
          f'wall time={(time.perf_counter()-start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
