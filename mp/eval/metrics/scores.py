# ------------------------------------------------------------------------------
# Definition of a score metrics for classification and segmentation, taking
# in tp, tn, fn and fp as inputs. For segmentation, these refer to pixel/voxel
# values for one example.
# ------------------------------------------------------------------------------
from medpy.metric.binary import hd,hd95
import numpy as np
import torch

class ScoreAbstract:
    r"""Ab abstract definition of a metric that uses true positives, true 
    negatives, false negatives and false positives to calculate a value."""

    def __init__(self):
        self.name = self.__class__.__name__

    def eval(self, tp, tn, fn, fp):
        raise NotImplementedError

class ScoreHausdorff(ScoreAbstract):
    r"""Hausdorff distance."""

    def eval(self, target, pred):
        target = target.squeeze().cpu().numpy().astype(np.int)
        pred = pred.squeeze().cpu().numpy().astype(np.int)
        score = []
        for d in range(target.shape[-1]):
            target_d = target[...,d]
            pred_d = pred[...,d]
            if np.sum(target_d) == 0:
                continue
            if np.sum(pred_d) == 0:
                score.append(np.max(pred_d.shape))
                continue
            score.append(hd95(pred_d,target_d))
        return np.mean(score)
class ScoreDice(ScoreAbstract):
    r"""Dice score, inverce of a Dice loss except for the smoothing factor in
    the loss."""

    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fn + fp > 0:
                return 0.
            else:
                return 1.
        return (2 * tp) / (2 * tp + fp + fn)


class ScoreIoU(ScoreAbstract):
    r"""Intersection over Union."""

    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fn + fp > 0:
                return 0.
            else:
                return 1.
        return tp / (tp + fp + fn)


class ScorePrecision(ScoreAbstract):
    r"""Precision."""

    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tp / (tp + fp)


class ScorePPV(ScorePrecision):
    r"""Positive predictve value, equivalent to precision."""
    pass


class ScoreRecall(ScoreAbstract):
    r"""Recall."""

    def eval(self, tp, tn, fn, fp):
        if tp == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tp / (tp + fn)


class ScoreSensitivity(ScoreRecall):
    r"""Sensitivity, equivalent to recall."""
    pass


class ScoreTPR(ScoreRecall):
    r"""True positive rate, equivalent to recall."""
    pass


class ScoreSpecificity(ScoreAbstract):
    r"""Specificity."""

    def eval(self, tp, tn, fn, fp):
        if tn == 0:
            if fp > 0:
                return 0.
            else:
                return 1.
        return tn / (tn + fp)


class ScoreTNR(ScoreSpecificity):
    r"""True negative rate, equivalent to specificity."""
    pass
