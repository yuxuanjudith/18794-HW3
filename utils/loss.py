import torch.nn as nn
import torch.nn.functional as F
import torch

# =========================== Define your custom loss function ===========================================


class FocalLoss(nn.Module):
    """Class-balanced focal loss for dense prediction (Lin et al., "Focal Loss for Dense
    Object Detection", ICCV 2017), extended to the multi-class, per-pixel setting.

        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

    * (1 - p_t)^gamma  down-weights pixels that are already classified well (mostly the
      huge, easy background region), so the gradient is dominated by hard pixels:
      object boundaries, small objects and rare classes.
    * alpha_t          optional per-class weight (e.g. inverse-sqrt class frequency computed
      from the training set) that re-balances rare vs. frequent classes.

    Args:
        alpha (Tensor | list | None): per-class weights of shape [C]; None = no class weighting.
        gamma (float): focusing parameter; gamma = 0 and alpha = None reduces to cross-entropy.
        ignore_index (int): label to ignore (255 = VOC object boundaries).
        reduction (str): 'mean' (over non-ignored pixels) | 'sum' | 'none'.
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        # register as buffer so it moves to the right device together with .to(device)
        self.register_buffer('alpha', alpha if alpha is not None else None)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C, H, W] logits, targets: [N, H, W] long
        valid = targets != self.ignore_index
        safe_targets = targets.clone()
        safe_targets[~valid] = 0  # any valid index; these pixels are masked out below

        log_p = F.log_softmax(inputs.float(), dim=1)                           # FP32 reduction under AMP
        log_pt = log_p.gather(1, safe_targets.unsqueeze(1)).squeeze(1)         # [N, H, W]
        pt = log_pt.exp()

        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device, inputs.dtype)
            loss = loss * alpha[safe_targets]
        loss = loss * valid.to(loss.dtype)

        if self.reduction == 'mean':
            return loss.sum() / valid.sum().clamp(min=1)
        if self.reduction == 'sum':
            return loss.sum()
        return loss
