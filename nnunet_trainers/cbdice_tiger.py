"""
Self-contained cbDice (Centerline Boundary Dice, MICCAI'24) loss for Tiger SQ-AI.

Vendored & adapted from PengchengShi1220/cbDice (Apache-2.0):
  loss/cbdice_loss.py + loss/soft_skeleton.py  (https://github.com/PengchengShi1220/cbDice)

CHANGES vs upstream (so it runs in seg_env WITHOUT monai/cucim/cupy):
  * `monai.transforms.distance_transform_edt` -> a per-image `scipy.ndimage.
    distance_transform_edt` helper (`_edt_batch`). The EDT in upstream `get_weights`
    only ever runs on a DETACHED, thresholded binary mask (GT: `y_true`; pred:
    `(mask_prob>0.5).int()`), so it carries NO gradient — replacing the EDT backend
    is numerically equivalent and keeps differentiability intact (grads flow only
    through the soft-skeleton probabilities & the elementwise weight multiplies,
    exactly as in upstream). Same pattern the banked BoundaryFT trainer uses.
  * Default skeletonizer = MORPHOLOGICAL `SoftSkeletonize` (num_iter=10) only; the
    topology-preserving `Skeletonize` (heavy autograd machinery) is NOT used
    (t_skeletonize_flage stays False), matching the cbDice repo default trainer.

cbDice targets THIN / TUBULAR structures via a centerline (soft-skeleton) +
boundary distance-transform weighting -> the deciding w3 tier for Tiger
(nerves / vessels / lymphatics). Binary foreground formulation (all fg channels
max-pooled to one channel), as in the upstream SoftcbDiceLoss.
"""
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Morphological soft-skeleton (upstream loss/soft_skeleton.py, unchanged)      #
# --------------------------------------------------------------------------- #
class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter=40):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)


# --------------------------------------------------------------------------- #
# scipy-EDT backend (replaces monai.distance_transform_edt; no-grad binary in) #
# --------------------------------------------------------------------------- #
def _edt_batch(mask: torch.Tensor) -> torch.Tensor:
    """Per-image Euclidean distance transform of a (B, H, W) binary tensor.
    Distance from each fg voxel to the nearest bg voxel (== monai/scipy default).
    Input is detached/binary -> no gradient -> CPU scipy is exact & cheap here."""
    from scipy.ndimage import distance_transform_edt as _edt
    arr = mask.detach().cpu().numpy()
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = _edt(arr[i].astype(np.uint8)).astype(np.float32)
    return torch.from_numpy(out).to(mask.device)


def combine_tensors(A, B, C):
    A_C = A * C
    B_C = B * C
    D = B_C.clone()
    mask_AC = (A != 0) & (B == 0)
    D[mask_AC] = A_C[mask_AC]
    return D


def get_weights(mask_input, skel_input, dim, prob_flag=True):
    """Upstream cbDice get_weights with the EDT backend swapped to scipy.
    Returns (dist_map_norm*mask, skel_R_norm*mask, I_norm*skel)."""
    if prob_flag:
        mask_prob = mask_input
        skel_prob = skel_input
        mask = (mask_prob > 0.5).int()
        skel = (skel_prob > 0.5).int()
    else:
        mask = mask_input
        skel = skel_input

    distances = _edt_batch(mask).float()
    distances[mask == 0] = 0

    skel_radius = torch.zeros_like(distances, dtype=torch.float32)
    skel_radius[skel == 1] = distances[skel == 1]

    dist_map_norm = torch.zeros_like(distances, dtype=torch.float32)
    skel_R_norm = torch.zeros_like(skel_radius, dtype=torch.float32)
    I_norm = torch.zeros_like(mask, dtype=torch.float32)
    for i in range(skel_radius.shape[0]):
        distances_i = distances[i]
        skel_i = skel_radius[i]
        skel_radius_max = max(skel_i.max(), torch.tensor(1.0, device=skel_i.device))
        skel_radius_min = max(skel_i.min(), torch.tensor(1.0, device=skel_i.device))

        distances_i[distances_i > skel_radius_max] = skel_radius_max
        dist_map_norm[i] = distances_i / skel_radius_max
        skel_R_norm[i] = skel_i / skel_radius_max

        # subtraction-based inverse (linear) — the upstream "Important Update".
        if dim == 2:
            I_norm[i] = (skel_radius_max - skel_i + skel_radius_min) / skel_radius_max
        else:
            I_norm[i] = ((skel_radius_max - skel_i + skel_radius_min) / skel_radius_max) ** 2

    I_norm[skel == 0] = 0  # 0 for non-skeleton pixels

    if prob_flag:
        return dist_map_norm * mask_prob, skel_R_norm * mask_prob, I_norm * skel_prob
    else:
        return dist_map_norm * mask, skel_R_norm * mask, I_norm * skel


# --------------------------------------------------------------------------- #
# SoftcbDiceLoss (upstream loss/cbdice_loss.py, morphological skeleton only)   #
# --------------------------------------------------------------------------- #
class SoftcbDiceLoss(torch.nn.Module):
    """Centerline Boundary Dice loss (binary fg). Returns a NEGATIVE Dice-style
    value in [-1, 0] (perfect overlap -> ~ -1). The compound wrapper below shifts
    it to a proper [0, ~2] penalty (1 + cbdice) so 'perfect -> ~0'."""

    def __init__(self, iter_=10, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C fg -> 1
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]  # fg probability map

        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()
            y_pred_hard = (y_pred_prob > 0.5).float()
            # morphological soft-skeleton (default; topology variant not vendored)
            skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
            skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)

        skel_pred_prob = skel_pred_hard * y_pred_prob

        q_vl, q_slvl, q_sl = get_weights(y_true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, q_sp = get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl)) + self.smooth) / \
                  (torch.sum(combine_tensors(q_spvp, q_slvl, q_sp)) + self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp)) + self.smooth) / \
                  (torch.sum(combine_tensors(q_slvl, q_spvp, q_sl)) + self.smooth)

        cb_dice_loss = -2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        return cb_dice_loss
