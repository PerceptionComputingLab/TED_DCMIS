# ------------------------------------------------------------------------------
# Collection of loss metrics that can be used during training, including binary
# cross-entropy and dice. Class-wise weights can be specified.
# Losses receive a 'target' array with shape (batch_size, channel_dim, etc.)
# and channel dimension equal to nr. of classes that has been previously
# transformed (through e.g. softmax) so that values lie between 0 and 1, and an
# 'output' array with the same dimension and values that are either 0 or 1.
# The results of the loss is always averaged over batch items (the first dim).
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mp.eval.losses.loss_abstract import LossAbstract

import numpy as np


class LossDice(LossAbstract):
    r"""Dice loss with a smoothing factor."""

    def __init__(self, smooth=1.0, device="cuda:0"):
        super().__init__(device=device)
        self.smooth = smooth
        self.device = device
        self.name = "LossDice[smooth=" + str(self.smooth) + "]"

    def forward(self, output, target):
        output_flat = output.view(-1)
        target_flat = target.view(-1)
        intersection = (output_flat * target_flat).sum()
        return 1 - (
            (2.0 * intersection + self.smooth)
            / (output_flat.sum() + target_flat.sum() + self.smooth)
        )


class LossBCE(LossAbstract):
    r"""Binary cross entropy loss."""

    def __init__(self, device="cuda:0"):
        super().__init__(device=device)
        self.device = device
        self.bce = nn.BCELoss(reduction="mean")

    def forward(self, output, target):
        try:
            bce_loss = self.bce(output, target)
        except:
            print(output.max(), output.min())
            print(target.max(), target.min())
            print(torch.isnan(output).any())
            print(torch.isnan(target).any())
        return bce_loss


class LossAKT(LossAbstract):
    r"""Loss function for the AKT method."""

    def __init__(self, device="cuda:0", gugf=True):
        super().__init__(device=device)
        self.device = device
        self.epsilon = torch.from_numpy(np.array(1e-5))
        self.uncertainty = gugf

    def forward(self, output, target):
        probabilities_old = target
        target = target.argmax(dim=1, keepdim=True)  # pseudo label

        probabilities = output
        log_probs = torch.log(probabilities + self.epsilon.to(output.device))
        log_probs_target = log_probs.gather(
            dim=1, index=target.long()
        )  # pseudo label * log(prediction)
        net_predictions = log_probs.argmax(dim=1, keepdim=True)
        error_predictions = (
            (net_predictions != target).float().detach()
        )  # selective map
        error_predictions_num = torch.sum(error_predictions)
        if self.uncertainty:
            jointly = probabilities_old * probabilities
            # gvu is not keeping the same shape as the others, so broadcasting on the batch
            uncertainty = (
                torch.ones([1]).to(output.device) - torch.sum(jointly, dim=1)
            ).detach()
        else:
            uncertainty = torch.ones([1]).to(output.device)

        loss = -torch.sum(uncertainty * error_predictions * log_probs_target) / (
            error_predictions_num + 1
        )
        return loss


class LossCombined(LossAbstract):
    r"""A combination of several different losses."""

    def __init__(self, losses, weights, device="cuda:0"):
        super().__init__(device=device)
        self.losses = losses
        self.weights = weights
        # Set name
        self.name = "LossCombined["
        for loss, weight in zip(self.losses, self.weights):
            self.name += str(weight) + "x" + loss.name + "+"
        self.name = self.name[:-1] + "]"

    def forward(self, output, target):
        total_loss = torch.zeros(1).to(self.device)
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(output, target)
        return total_loss

    def get_evaluation_dict(self, output, target):
        eval_dict = super().get_evaluation_dict(output, target)
        for loss, weight in zip(self.losses, self.weights):
            loss_eval_dict = loss.get_evaluation_dict(output, target)
            for key, value in loss_eval_dict.items():
                eval_dict[key] = value
        return eval_dict


class LossDiceBCE(LossCombined):
    r"""A combination of Dice and Binary cross entropy."""

    def __init__(self, bce_weight=1.0, smooth=1.0, device="cuda:0"):
        super().__init__(
            losses=[LossDice(smooth=smooth), LossBCE()],
            weights=[1.0, bce_weight],
            device=device,
        )
