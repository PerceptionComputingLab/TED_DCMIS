import torch, torch.nn as nn
import torch, copy, math
from mp.eval.losses.losses_segmentation import LossDice

""" copy from https://github.com/MECLabTUDA/Lifelong-nnUNet"""


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    r"""Copied from https://github.com/fcdl94/MiB/blob/1c589833ce5c1a7446469d4602ceab2cdeac1b0e/utils/loss.py#L139."""

    def __init__(self, reduction="mean", alpha=1.0):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        new_cl = 1  # inputs.shape[1] - targets.shape[1] if inputs.shape[1] != targets.shape[1] else inputs.shape[1]
        targets = targets * self.alpha
        new_bkg_idx = torch.tensor(
            [0] + [x for x in range(targets.shape[1], inputs.shape[1])]
        ).to(inputs.device)
        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        if (
            inputs.size()[1] - new_cl
        ) > 1:  # Ensure that the outputs_no_bgk is not empty if new_cl is 1 and inputs.size()[1] is 1 eg.
            outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(
                dim=1
            )  # B, OLD_CL, H, W
        else:
            outputs_no_bgk = inputs[:, 1:] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = (
            torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1)
            - den
        )  # B, H, W
        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W
        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (
            labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)
        ) / targets.shape[1]
        if mask is not None:
            loss = loss * mask.float()
        if self.reduction == "mean":
            outputs = -torch.mean(loss)
        elif self.reduction == "sum":
            outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs


# -- Loss function for the PLOP approach -- #
class LossPLOP(nn.Module):
    # -- This implementation represents the method proposed in the paper https://arxiv.org/pdf/2011.11390.pdf -- #
    # -- Based on the implementation from here: https://github.com/arthurdouillard/CVPR2021_PLOP/blob/main/train.py -- #
    def __init__(self):
        """This Loss function is based on the PLOP approach. The loss function will be updated using the proposed method in the paper linked above.
        It needs the intermediate convolutional outputs of the previous models, the scales for weighting the result of previous tasks.
        """
        # -- Initialize -- #
        super(LossPLOP, self).__init__()

        self.ce = LossDice()

    def forward(self, x, x_o, y):
        x_o = copy.deepcopy(x_o)
        y = copy.deepcopy(y)
        foreground_label = (1.0 - torch.argmax(x_o, dim=1)) + y[:, 0, :, :]
        foreground_label = torch.where(
            foreground_label > 0.0,
            torch.ones([1]).to(x.device),
            torch.zeros([1]).to(x.device),
        )
        pseudo_label = y
        pseudo_label[:, 0, :, :] = foreground_label

        loss = self.ce(x, pseudo_label)

        return loss


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


def entropy(probabilities):
    r"""Computes the entropy per pixel.
    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
        Saporta et al.
        CVPR Workshop 2020
    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=0)
