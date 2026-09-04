"""
Metric-aware nnU-Net trainers for TIGER SQ-AI (custom, dropped into the nnU-Net
trainer package so recursive_find_trainer_class_by_name picks them up).

Lever P2.1 — target BOTH halves of the leaderboard metric (w=3 Dice + nHD):
  * CE + Focal-Tversky   (alpha=0.3, beta=0.7 -> recall-weighted, lifts rare/thin
    w=3 classes; gamma=4/3~=1.33 focal modulation focuses on hard low-Tversky
    classes) -> directly raises the w=3 sub-average Dice.
  * Boundary-aware: Focal-Tversky already penalises FP/FN boundary leakage; the
    recall weighting reduces the false-negative "absent class -> one-empty Dice=0"
    catastrophe and tightens masks (helps nHD).
All trainers keep h-flip/mirroring OFF (flipped thoracic view) and run 250 epochs,
matching the baseline nnUNetTrainer_250epochs_NoMirroring for an apples-to-apples
ablation. Only the LOSS changes vs the baseline.
"""
import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalTverskyLoss(nn.Module):
    """Soft Focal-Tversky over the softmax probabilities.

    Tversky_c = TP / (TP + alpha*FP + beta*FN);  loss = mean_c (1 - Tversky_c)**gamma.
    alpha<beta upweights false negatives (recall) -> better for tiny/rare structures.
    Mirrors SoftDiceLoss conventions (apply_nonlin, batch_dice, do_bg, smooth, ddp).
    """

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False,
                 smooth=1e-5, ddp=True, alpha=0.3, beta=0.7, gamma=4.0 / 3.0):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = [0] + list(range(2, len(shp_x))) if self.batch_dice else list(range(2, len(shp_x)))

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            from nnunetv2.training.loss.dice import AllGatherGrad
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        ).clamp_min(1e-8)

        ft = torch.pow((1.0 - tversky).clamp_min(0.0), self.gamma)

        if not self.do_bg:
            if self.batch_dice:
                ft = ft[1:]
            else:
                ft = ft[:, 1:]

        return ft.mean()


class CE_and_FocalTversky_loss(nn.Module):
    def __init__(self, ft_kwargs, ce_kwargs, weight_ce=1.0, weight_ft=1.0,
                 ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_ce = weight_ce
        self.weight_ft = weight_ft
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ft = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **ft_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = (target != self.ignore_label).bool()
            target_ft = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_ft = target
            mask = None

        ft_loss = self.ft(net_output, target_ft, loss_mask=mask) if self.weight_ft != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_ce * ce_loss + self.weight_ft * ft_loss


class _TverskyBuildLossMixin:
    """Replaces nnU-Net's DC_and_CE with CE + Focal-Tversky, keeping deep supervision."""

    # subclasses set these
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0

    def _build_loss(self):
        if self.label_manager.has_regions:
            # region-based (one-hot) not used for our T1/T2 -> fall back to parent
            return super()._build_loss()

        loss = CE_and_FocalTversky_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'do_bg': False,
             'smooth': 1e-5, 'ddp': self.is_ddp,
             'alpha': self._ft_alpha, 'beta': self._ft_beta, 'gamma': self._ft_gamma},
            {}, weight_ce=self._weight_ce, weight_ft=self._weight_ft,
            ignore_label=self.label_manager.ignore_label)

        if self._do_i_compile():
            loss.ft = torch.compile(loss.ft)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class _NoMirror250:
    """250 epochs + mirroring OFF (flipped thoracic view)."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rot, dummy, ips, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rot, dummy, ips, None


class nnUNetTrainer_250epochs_NoMirror_FocalTversky(
        _TverskyBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Main P2.1 lever: CE + Focal-Tversky (a0.3/b0.7/g1.33), flip-safe, 250 ep."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0


class _NoMirror1000:
    """1000 epochs + mirroring OFF (flipped thoracic view).

    Convergence lever (2026-06-16): the 250-ep T1/T2 runs were STILL improving at
    ep 250 (best EMA pseudo Dice set late, +0.013/+0.016 in the last 20% of best
    updates) -> the short schedule under-trained. 1000 ep == nnU-Net default length.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rot, dummy, ips, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rot, dummy, ips, None


class nnUNetTrainer_1000epochs_NoMirror_FocalTversky(
        _TverskyBuildLossMixin, _NoMirror1000, nnUNetTrainer):
    """Longer-schedule T2 lever: CE + Focal-Tversky, flip-safe, 1000 ep (vs 250)."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0


class nnUNetTrainer_1000epochs_NoMirroring(_NoMirror1000, nnUNetTrainer):
    """Longer-schedule T1 lever: stock DC+CE loss, flip-safe, 1000 ep (vs 250)."""
    pass


# ===========================================================================
# AUTONOMOUS LADDER (2026-06-16) — multi-seed ensemble + TopK boundary loss
# ===========================================================================

class _SeededInitMixin:
    """Set a per-trainer torch/numpy seed at construction so distinct trainer
    subclasses (each -> its own nnUNet_results dir) give genuinely different
    weight inits + augmentation streams -> a real multi-seed ensemble when their
    softmax probabilities are averaged at inference. _seed set by the subclass."""
    _seed = 0

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        import random
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)
        super().__init__(plans, configuration, fold, dataset_json, device)


# --- multi-seed baseline (DC+CE) for the MERGED task (official task2) --------
class nnUNetTrainer_250epochs_NoMirroring_seed1(
        _SeededInitMixin, _NoMirror250, nnUNetTrainer):
    """Merged-task ensemble member, seed 1 (250 ep, DC+CE, flip-safe)."""
    _seed = 1


class nnUNetTrainer_250epochs_NoMirroring_seed2(
        _SeededInitMixin, _NoMirror250, nnUNetTrainer):
    """Merged-task ensemble member, seed 2 (250 ep, DC+CE, flip-safe)."""
    _seed = 2


# --- multi-seed Focal-Tversky for the FINE task (official task1) -------------
class nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1(
        _SeededInitMixin, _TverskyBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Fine-task ensemble member, seed 1 (250 ep, CE+Focal-Tversky, flip-safe)."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0
    _seed = 1


class nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2(
        _SeededInitMixin, _TverskyBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Fine-task ensemble member, seed 2 (250 ep, CE+Focal-Tversky, flip-safe)."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0
    _seed = 2


# ===========================================================================
# TopK + Tversky loss for the MERGED task (official task2) hard w3 classes.
# FT was REFUTED on the coarse/merged task (it hurt: wDice 0.358 vs 0.396).
# TopK CE focuses gradient on the hardest k% of pixels (the thin w3 boundaries
# Lymphatic_Tissue/Nerves/Vessels) while Tversky (alpha<beta) keeps recall on the
# rare classes -> a DIFFERENT precision/recall trade than plain FT. Pre-registered
# kill criterion: REFUTE if w3 sub-avg Dice Delta <= +0.01 vs the merged baseline.
# ===========================================================================

class TopKLoss(RobustCrossEntropyLoss):
    """Per-pixel CE keeping only the hardest k% of pixels (OHEM-style).
    nnU-Net ships a TopKLoss but we re-derive it locally to avoid version drift."""

    def __init__(self, weight=None, ignore_index=-100, k=10, label_smoothing=0.0):
        self.k = k
        super().__init__(weight=weight, ignore_index=ignore_index,
                         reduction='none', label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target.long()
        ce = super().forward(inp, target)
        num_pixels = ce.numel()
        n_keep = max(1, int(num_pixels * self.k / 100.0))
        ce = ce.view(-1)
        topk, _ = torch.topk(ce, n_keep, sorted=False)
        return topk.mean()


class CE_TopK_and_Tversky_loss(nn.Module):
    """TopK-CE (hard-pixel mining) + Focal-Tversky (recall on rare w3)."""

    def __init__(self, ft_kwargs, k=10, weight_ce=1.0, weight_ft=1.0,
                 ignore_label=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_ft = weight_ft
        self.ignore_label = ignore_label
        ce_kwargs = {}
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ce = TopKLoss(k=k, **ce_kwargs)
        self.ft = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **ft_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = (target != self.ignore_label).bool()
            target_ft = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_ft = target
            mask = None
        ft_loss = self.ft(net_output, target_ft, loss_mask=mask) if self.weight_ft != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_ce * ce_loss + self.weight_ft * ft_loss


class _TopKTverskyBuildLossMixin:
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _topk_k = 10
    _weight_ce = 1.0
    _weight_ft = 1.0

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()
        loss = CE_TopK_and_Tversky_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'do_bg': False,
             'smooth': 1e-5, 'ddp': self.is_ddp,
             'alpha': self._ft_alpha, 'beta': self._ft_beta, 'gamma': self._ft_gamma},
            k=self._topk_k, weight_ce=self._weight_ce, weight_ft=self._weight_ft,
            ignore_label=self.label_manager.ignore_label)
        if self._do_i_compile():
            loss.ft = torch.compile(loss.ft)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_250epochs_NoMirror_TopKTversky(
        _TopKTverskyBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Merged-task hard-class lever: TopK-CE(k=10) + Focal-Tversky, flip-safe, 250 ep."""
    pass


# ===========================================================================
# AUTONOMOUS DEEP BATCH (2026-06-16) — extra ensemble seeds (3..7) + boundary loss
# ===========================================================================

# --- bigger multi-seed ensemble: extra MERGED (DC+CE) members seed3..7 --------
class nnUNetTrainer_250epochs_NoMirroring_seed3(
        _SeededInitMixin, _NoMirror250, nnUNetTrainer):
    """Merged-task ensemble member, seed 3 (250 ep, DC+CE, flip-safe)."""
    _seed = 3


class nnUNetTrainer_250epochs_NoMirroring_seed4(
        _SeededInitMixin, _NoMirror250, nnUNetTrainer):
    """Merged-task ensemble member, seed 4 (250 ep, DC+CE, flip-safe)."""
    _seed = 4


class nnUNetTrainer_250epochs_NoMirroring_seed5(
        _SeededInitMixin, _NoMirror250, nnUNetTrainer):
    """Merged-task ensemble member, seed 5 (250 ep, DC+CE, flip-safe)."""
    _seed = 5


# --- bigger multi-seed ensemble: extra FINE (Focal-Tversky) members seed3..7 --
class nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed3(
        _SeededInitMixin, _TverskyBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Fine-task ensemble member, seed 3 (250 ep, CE+Focal-Tversky, flip-safe)."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0
    _seed = 3


class nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed4(
        _SeededInitMixin, _TverskyBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Fine-task ensemble member, seed 4 (250 ep, CE+Focal-Tversky, flip-safe)."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0
    _seed = 4


class nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed5(
        _SeededInitMixin, _TverskyBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Fine-task ensemble member, seed 5 (250 ep, CE+Focal-Tversky, flip-safe)."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0
    _seed = 5


# ===========================================================================
# BOUNDARY-AWARE loss (Hausdorff-DT surrogate) for the hard w3 tier + nHD.
# The leaderboard ranks mean(rank_Dice, rank_nHD); nHD is boundary-dominated and
# tracks the thin w3 classes (PA/SCV/RMB nHD 0.6+). A distance-transform-weighted
# boundary term penalises errors PROPORTIONAL to their distance from the GT
# boundary (the Kervadec/Karimi Hausdorff-DT surrogate) -> pulls predicted
# boundaries onto the true ones -> lowers nHD on thin structures without the
# capacity cost. Combined with CE + Focal-Tversky (keep the recall that won on
# FINE) so it targets BOTH metric halves. Pre-registered kill criterion: REFUTE
# if nHD does not improve (Delta_nHD >= 0) AND w3 Dice does not gain > +0.01.
# ===========================================================================

class HDDTBoundaryLoss(nn.Module):
    """Boundary loss via the GT distance transform (Karimi & Salcudean 2019,
    'Reducing the Hausdorff Distance ...', the one-sided DT surrogate).

    For each foreground class c with softmax prob p_c and GT mask g_c, weight the
    squared error (p_c - g_c)^2 by the squared distance-to-boundary of BOTH the GT
    and the prediction region, so misplaced boundary pixels far from the true
    boundary are penalised more. Distance maps are computed on the (detached) GT
    and the thresholded prediction with scipy EDT, recomputed per forward (cheap
    at our batch sizes). Background channel excluded (do_bg=False)."""

    def __init__(self, apply_nonlin=softmax_helper_dim1, do_bg=False, alpha=2.0):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg
        self.alpha = alpha  # DT exponent influence (kept linear-in-DT^2 by default)

    @staticmethod
    def _dist_maps(seg_onehot):
        # seg_onehot: (B, C, H, W) float {0,1} on CPU numpy -> per-class signed-ish
        # DT field dtm = dist_outside^2 + dist_inside^2 (Karimi eq.); returns same shape.
        from scipy.ndimage import distance_transform_edt as edt
        import numpy as _np
        B, C, H, W = seg_onehot.shape
        out = _np.zeros((B, C, H, W), dtype=_np.float32)
        for b in range(B):
            for c in range(C):
                posmask = seg_onehot[b, c].astype(bool)
                if posmask.any() and (~posmask).any():
                    negmask = ~posmask
                    out[b, c] = edt(negmask) ** 2 + edt(posmask) ** 2
        return out

    def forward(self, x, y, loss_mask=None):
        x = self.apply_nonlin(x) if self.apply_nonlin is not None else x
        # build GT one-hot
        with torch.no_grad():
            if y.shape[1] == 1:
                y_oh = torch.zeros_like(x)
                y_oh.scatter_(1, y.long(), 1)
            else:
                y_oh = y.float()
            gt_dtm = torch.from_numpy(
                self._dist_maps(y_oh.detach().cpu().numpy())
            ).to(x.device)
            # normalise per-image so the term is scale-free across resolutions
            denom = gt_dtm.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
            gt_dtm = gt_dtm / denom
        delta = (x - y_oh) ** 2
        bl = delta * gt_dtm
        if not self.do_bg:
            bl = bl[:, 1:]
        return bl.mean()


class CE_FT_and_Boundary_loss(nn.Module):
    """CE + Focal-Tversky (Dice/recall on rare w3) + HD-DT boundary (nHD)."""

    def __init__(self, ft_kwargs, weight_ce=1.0, weight_ft=1.0, weight_bd=0.1,
                 ignore_label=None):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_ft = weight_ft
        self.weight_bd = weight_bd
        self.ignore_label = ignore_label
        ce_kwargs = {}
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ft = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **ft_kwargs)
        self.bd = HDDTBoundaryLoss(apply_nonlin=softmax_helper_dim1, do_bg=False)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = (target != self.ignore_label).bool()
            target_ft = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_ft = target
            mask = None
            num_fg = 1
        ft_loss = self.ft(net_output, target_ft, loss_mask=mask) if self.weight_ft != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        bd_loss = self.bd(net_output, target_ft) if self.weight_bd != 0 else 0
        return self.weight_ce * ce_loss + self.weight_ft * ft_loss + self.weight_bd * bd_loss


class _BoundaryBuildLossMixin:
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _weight_ce = 1.0
    _weight_ft = 1.0
    _weight_bd = 0.1

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()
        loss = CE_FT_and_Boundary_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'do_bg': False,
             'smooth': 1e-5, 'ddp': self.is_ddp,
             'alpha': self._ft_alpha, 'beta': self._ft_beta, 'gamma': self._ft_gamma},
            weight_ce=self._weight_ce, weight_ft=self._weight_ft,
            weight_bd=self._weight_bd,
            ignore_label=self.label_manager.ignore_label)
        # boundary term uses a numpy/scipy EDT -> NOT torch.compile-able; do not compile.
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _do_i_compile(self):
        # the boundary loss recomputes scipy EDT each step -> disable compile to
        # avoid graph breaks / recompiles on the dynamic DT field.
        return False


class nnUNetTrainer_250epochs_NoMirror_BoundaryFT(
        _BoundaryBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Boundary-aware (HD-DT) + CE + Focal-Tversky, flip-safe, 250 ep.
    Targets the nHD half of the metric on the thin w3 tier."""
    pass


# ===========================================================================
# cbDice (Centerline Boundary Dice, MICCAI'24 — Shi et al., Apache-2.0) loss.
# Vendored self-contained (no monai/cucim) in nnunetv2.training.loss.cbdice_tiger.
# cbDice weights the Dice match by a soft-skeleton (centerline) + boundary
# distance-transform, so it specifically rewards getting THIN / TUBULAR
# structures right — exactly the Tiger deciding w3 tier (nerves/vessels/
# lymphatics). Two trainers, both 250 ep + mirroring OFF == the banked
# Focal-Tversky schedule so ONLY the loss differs (clean ablation):
#   (1) CE + cbDice
#   (2) CE + cbDice + Focal-Tversky
# Weighting follows the cbDice repo recipe (DC_and_CE_and_CBDC_loss):
#   weight_cbdice = 1, weight_ce = 1  (repo uses w_ce = w_dice + w_cbdice;
#   here Dice is replaced/augmented by FT or dropped, so a clean CE:cbDice 1:1
#   is used, and 1:1:1 for the three-term variant — noted in the dossier).
# The cbDice term uses a scipy-EDT (no-grad binary) + python loop -> NOT
# torch.compile-able, so _do_i_compile() is forced False (same as BoundaryFT).
# SoftcbDiceLoss returns a NEGATIVE Dice (-1..0); we shift to (1 + cbdice) so a
# perfect prediction -> ~0 and the term is a proper non-negative penalty.
# ===========================================================================

# Fail-soft: cbDice is only needed by the cbDice training trainers, NOT by the
# submission trainers (TopK / FocalTversky). The Docker container does not stage
# cbdice_tiger.py, and trainer-discovery (recursive_find_python_class) imports
# this whole module at inference — so a hard import here crashes ALL trainer
# discovery in the container (observed: job 6209008 FAILED). Guard it so the
# module still imports when cbdice_tiger is absent; the cbDice trainer classes
# below will raise only if actually instantiated.
try:
    from nnunetv2.training.loss.cbdice_tiger import SoftcbDiceLoss
except ModuleNotFoundError:
    SoftcbDiceLoss = None


class CE_and_cbDice_loss(nn.Module):
    """RobustCE + cbDice (shifted to 1 + cbdice so perfect -> ~0)."""

    def __init__(self, cbdc_kwargs, ce_kwargs, weight_ce=1.0, weight_cb=1.0,
                 ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_ce = weight_ce
        self.weight_cb = weight_cb
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.cb = SoftcbDiceLoss(**cbdc_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            num_fg = (target != self.ignore_label).sum()
            target_cb = torch.clone(target)
            target_cb[target == self.ignore_label] = 0
        else:
            target_cb = target
            num_fg = 1
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        cb_loss = (1.0 + self.cb(net_output, target_cb)) if self.weight_cb != 0 else 0
        return self.weight_ce * ce_loss + self.weight_cb * cb_loss


class CE_cbDice_and_FocalTversky_loss(nn.Module):
    """RobustCE + cbDice (1+cbdice) + Focal-Tversky — targets w3 Dice (FT recall)
    AND thin-structure centerline/boundary (cbDice) simultaneously."""

    def __init__(self, ft_kwargs, cbdc_kwargs, ce_kwargs, weight_ce=1.0,
                 weight_cb=1.0, weight_ft=1.0, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_ce = weight_ce
        self.weight_cb = weight_cb
        self.weight_ft = weight_ft
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.cb = SoftcbDiceLoss(**cbdc_kwargs)
        self.ft = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **ft_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = (target != self.ignore_label).bool()
            target_x = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_x = target
            mask = None
            num_fg = 1
        ft_loss = self.ft(net_output, target_x, loss_mask=mask) if self.weight_ft != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        cb_loss = (1.0 + self.cb(net_output, target_x)) if self.weight_cb != 0 else 0
        return self.weight_ce * ce_loss + self.weight_cb * cb_loss + self.weight_ft * ft_loss


class _cbDiceBuildLossMixin:
    """CE + cbDice (no Focal-Tversky)."""
    _cb_iter = 10
    _cb_smooth = 1e-3
    _weight_ce = 1.0
    _weight_cb = 1.0

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()
        loss = CE_and_cbDice_loss(
            {'iter_': self._cb_iter, 'smooth': self._cb_smooth}, {},
            weight_ce=self._weight_ce, weight_cb=self._weight_cb,
            ignore_label=self.label_manager.ignore_label)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _do_i_compile(self):
        # cbDice uses scipy-EDT + a python loop -> not torch.compile-able.
        return False


class _cbDiceFTBuildLossMixin:
    """CE + cbDice + Focal-Tversky."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _cb_iter = 10
    _cb_smooth = 1e-3
    _weight_ce = 1.0
    _weight_cb = 1.0
    _weight_ft = 1.0

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()
        loss = CE_cbDice_and_FocalTversky_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'do_bg': False,
             'smooth': 1e-5, 'ddp': self.is_ddp,
             'alpha': self._ft_alpha, 'beta': self._ft_beta, 'gamma': self._ft_gamma},
            {'iter_': self._cb_iter, 'smooth': self._cb_smooth}, {},
            weight_ce=self._weight_ce, weight_cb=self._weight_cb,
            weight_ft=self._weight_ft,
            ignore_label=self.label_manager.ignore_label)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _do_i_compile(self):
        return False


class nnUNetTrainer_250epochs_NoMirror_cbDice(
        _cbDiceBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """CE + cbDice (Centerline Boundary Dice), flip-safe, 250 ep.
    Targets thin/tubular w3 structures via centerline+boundary weighting."""
    pass


class nnUNetTrainer_250epochs_NoMirror_cbDiceFocalTversky(
        _cbDiceFTBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """CE + cbDice + Focal-Tversky, flip-safe, 250 ep.
    cbDice (centerline/boundary on thin w3) + FT (recall on rare w3)."""
    pass


# --- seeded cbDice+FT members for the RARE-CLASS HIGH-RES CASCADE (stage-2) ---
# Stage-2 (w3-crop) 3-seed ensemble: cbDiceFT seed0 (above) + seed1/seed2 here.
# Same CE + cbDice(clDice-centerline + boundary) + Focal-Tversky recipe; only the
# init/aug seed differs -> a genuine multi-seed ensemble when softmax-prob-avg'd.
class nnUNetTrainer_250epochs_NoMirror_cbDiceFocalTversky_seed1(
        _SeededInitMixin, _cbDiceFTBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Cascade stage-2 ensemble member, seed 1 (CE+cbDice+Focal-Tversky)."""
    _seed = 1


class nnUNetTrainer_250epochs_NoMirror_cbDiceFocalTversky_seed2(
        _SeededInitMixin, _cbDiceFTBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """Cascade stage-2 ensemble member, seed 2 (CE+cbDice+Focal-Tversky)."""
    _seed = 2


# ===========================================================================
# Skeleton-Recall loss (Kirchhoff et al., ECCV'24; MIC-DKFZ/Skeleton-Recall,
# Apache-2.0) — thin/tubular w3 lever (esp. Nerves). Added ON TOP of the FINE-
# task's CONFIRMED winning recipe (CE + Focal-Tversky) so the only thing that
# differs vs the FT champion is the extra skeleton-recall term -> a clean
# loss-only ablation (same _NoMirror250 schedule, mirroring OFF for the flipped
# thoracic view). The skeleton-recall term penalises NOT covering the GT
# centerline -> targets tubular under-segmentation (Nerves), complementing FT
# (which targets recall on rare w3) and FP-suppress (which targets the one-empty
# FP trap).
#
# LAMBDA CHOICE: weight_srec = 1.0 (== upstream nnUNetTrainerSkeletonRecall
# default weight_srec=1; CE:FT:SkelRecall = 1:1:1, matching the cbDice+FT 1:1:1
# convention so it is an apples-to-apples sibling-loss ablation). Skeleton-recall
# magnitude is naturally O(1) (a mean recall in [0,1] shifted to [0,~1]), so 1.0
# keeps it comparable to the CE + FT terms without swamping them.
#
# The Skeleton-Recall term builds the tubed GT skeleton INTERNALLY on the detached
# label (skimage.skeletonize + double dilation, no grad — like cbDice's EDT), so
# this is a drop-in _build_loss swap needing NO custom dataloader/train_step.
# scipy/skimage + a python loop -> NOT torch.compile-able -> _do_i_compile()=False.
#
# Fail-soft import (same discipline as the cbDice import above): a HARD top-level
# import of skelrecall_tiger would crash nnU-Net trainer-DISCOVERY
# (recursive_find_python_class imports the whole module) for ALL trainers,
# including the champion FT-ensemble + TopK at inference -> would break the Docker
# (cf. the cbDice build-failure, job 6209008). Guard it -> None; the SkelRecall
# trainer below raises only if actually instantiated.
# ===========================================================================
try:
    from nnunetv2.training.loss.skelrecall_tiger import (
        SoftSkeletonRecallLoss as _TigerSoftSkeletonRecallLoss,
    )
except (ModuleNotFoundError, ImportError):
    _TigerSoftSkeletonRecallLoss = None


class CE_FocalTversky_and_SkeletonRecall_loss(nn.Module):
    """RobustCE + Focal-Tversky + lambda * SkeletonRecall (shifted to (1 + srec)
    so a perfect prediction -> ~0). FT = the FINE-task's confirmed winning loss;
    SkelRecall is added on top to target thin/tubular Nerves centerline recall."""

    def __init__(self, ft_kwargs, srec_kwargs, ce_kwargs, weight_ce=1.0,
                 weight_ft=1.0, weight_srec=1.0, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_ce = weight_ce
        self.weight_ft = weight_ft
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ft = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **ft_kwargs)
        self.srec = _TigerSoftSkeletonRecallLoss(
            apply_nonlin=softmax_helper_dim1, **srec_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = (target != self.ignore_label).bool()
            target_x = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_x = target
            mask = None
            num_fg = 1

        ft_loss = self.ft(net_output, target_x, loss_mask=mask) \
            if self.weight_ft != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        # srec returns -recall in [-1, 0]; shift to (1 + srec) -> perfect ~0.
        srec_loss = (1.0 + self.srec(net_output, target_x, loss_mask=mask)) \
            if self.weight_srec != 0 else 0
        return (self.weight_ce * ce_loss
                + self.weight_ft * ft_loss
                + self.weight_srec * srec_loss)


class _SkelRecallFTBuildLossMixin:
    """CE + Focal-Tversky + Skeleton-Recall (1:1:1)."""
    _ft_alpha = 0.3
    _ft_beta = 0.7
    _ft_gamma = 4.0 / 3.0
    _srec_smooth = 1e-5
    _weight_ce = 1.0
    _weight_ft = 1.0
    _weight_srec = 1.0

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()
        loss = CE_FocalTversky_and_SkeletonRecall_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'do_bg': False,
             'smooth': 1e-5, 'ddp': self.is_ddp,
             'alpha': self._ft_alpha, 'beta': self._ft_beta, 'gamma': self._ft_gamma},
            {'batch_dice': self.configuration_manager.batch_dice, 'do_bg': False,
             'smooth': self._srec_smooth, 'ddp': self.is_ddp},
            {},
            weight_ce=self._weight_ce, weight_ft=self._weight_ft,
            weight_srec=self._weight_srec,
            ignore_label=self.label_manager.ignore_label)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _do_i_compile(self):
        # SkeletonRecall uses skimage skeletonize + a python loop -> not
        # torch.compile-able (same as cbDice / BoundaryFT).
        return False


class nnUNetTrainer_250epochs_NoMirror_SkelRecallFT(
        _SkelRecallFTBuildLossMixin, _NoMirror250, nnUNetTrainer):
    """CE + Focal-Tversky + Skeleton-Recall (Kirchhoff ECCV'24), flip-safe, 250 ep.
    SkelRecall (tubed-GT-centerline recall on thin/tubular w3, esp. Nerves) added
    ON TOP of the FINE-task's confirmed winning Focal-Tversky recipe. Identical
    schedule to nnUNetTrainer_250epochs_NoMirror_FocalTversky -> clean loss-only
    ablation."""
    pass
