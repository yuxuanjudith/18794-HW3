"""Small numerical checks for the assignment's data, network, loss and metric contracts."""
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import network
from datasets import VOCSegmentation
from metrics import StreamSegMetrics
from utils.loss import FocalLoss
from utils.ext_transforms import ExtRandomCrop


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    passed = []
    targets = torch.tensor([[[0, 1], [2, 255]], [[2, 1], [0, 1]]])
    x = torch.randn(2, 3, 2, 2, requires_grad=True)
    focal = FocalLoss(gamma=0)(x, targets)
    ce = F.cross_entropy(x, targets, ignore_index=255)
    torch.testing.assert_close(focal, ce)
    g1 = torch.autograd.grad(focal, x, retain_graph=True)[0]
    g2 = torch.autograd.grad(ce, x)[0]
    torch.testing.assert_close(g1, g2)
    passed.append('Focal gamma=0 matches CE values and gradients, excluding void pixels')
    empty_x = torch.randn(2, 3, 2, 2, requires_grad=True)
    empty_loss = FocalLoss()(empty_x, torch.full((2, 2, 2), 255))
    empty_loss.backward()
    assert empty_loss.item() == 0 and torch.count_nonzero(empty_x.grad) == 0
    passed.append('All-void focal target has zero loss and gradient')
    m = StreamSegMetrics(2)
    m.update([np.array([0,0,1,1,255])], [np.array([0,0,0,1,1])])
    result = m.get_results()
    assert np.isclose(result['Mean IoU'], 7/12)
    assert np.isclose(result['Overall Acc'], .75)
    passed.append('mIoU and accuracy match hand calculation; label 255 excluded')
    image = Image.new('RGB', (3, 3))
    label = Image.fromarray(np.ones((3, 3), dtype=np.uint8))
    _, padded = ExtRandomCrop((7, 7), pad_if_needed=True)(image, label)
    assert set(np.unique(padded)) == {1,255}
    passed.append('Padding creates ignored pixels, not invented background labels')
    dst = VOCSegmentation('./datasets/data', 'train')
    rgb, mask = dst[0]
    assert rgb.mode == 'RGB' and np.asarray(mask).ndim == 2
    assert set(np.unique(mask)) <= set(range(21)) | {255}
    assert dst.decode_target(mask).shape == (*np.asarray(mask).shape, 3)
    passed.append('VOC palette masks retain class indices; RGB decoding has correct shape')
    for name in ['deeplabv3_resnet50', 'deeplabv3plus_resnet50']:
        model = getattr(network, name)(num_classes=21, output_stride=16, pretrained_backbone=False)
        model.train()
        y = model(torch.randn(2,3,65,73))
        assert y.shape == (2,21,65,73)
        F.cross_entropy(y, torch.zeros((2,65,73), dtype=torch.long)).backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.classifier.parameters())
        passed.append(name + ': odd-sized input is restored and gradients flow through head')
        del model, y
    from pathlib import Path
    Path('report').mkdir(exist_ok=True)
    Path('report/implementation_checks.json').write_text(json.dumps({'passed':passed},indent=2),encoding='utf-8')
    print('\n'.join('PASS: '+s for s in passed), flush=True)


if __name__ == '__main__':
    main()
