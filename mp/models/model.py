# ------------------------------------------------------------------------------
# Class all model definitions should descend from.
# ------------------------------------------------------------------------------


import torch.nn as nn

from mp.models.segmentation.unet_fepegar import UNet2D
from mp.models.segmentation.vision_transformer import SwinUnet
import torch.optim as optim


class Model(nn.Module):
    r"""A model that descends from torch.nn.Model and includes methods to output
    a model summary, as well as the input_shape and output_shape fields used in
    other models and the logic to restore previous model states from a path.

    Args:
        input_shape tuple (int): Input shape with the form
            (channels, width, height, Opt(depth))
        output_shape (Obj): output shape, which takes different forms depending
            on the problem
    """

    def __init__(self, input_shape=(1, 32, 32), output_shape=2):
        super(Model, self).__init__()
        self.unet_old = None
        self.unet_new = None
        self.backbone = None
        self.input_shape = input_shape
        self.output_shape = output_shape

    def init_backbone(self):
        if self.backbone == "unet":
            self.unet_new = UNet2D(
                self.input_shape,
                self.nr_labels,
                dropout=self.unet_dropout,
                monte_carlo_dropout=self.unet_monte_carlo_dropout,
                preactivation=self.unet_preactivation,
            )
        elif self.backbone == "swinunet":
            self.unet_new = SwinUnet(
                input_shape=self.input_shape[1], nr_labels=self.nr_labels
            )
        self.unet_old = None

    def preprocess_input(self, x):
        r"""E.g. pretrained features. Override if needed."""
        return x

    def forward(self, x):
        r"""Forward pass of current U-Net

        Args:
            x (torch.Tensor): input batch

        Returns:
            (torch.Tensor): segmentated batch
        """
        return self.unet_new(x)

    def forward_old(self, x):
        r"""Forward pass of previous U-Net

        Args:
            x (torch.Tensor): input batch

        Returns:
            (torch.Tensor): segmentated batch
        """
        return self.unet_old(x)

    def freeze_unet(self, unet):
        r"""Freeze U-Net

        Args:
            unet (nn.Module): U-Net

        Returns:
            (nn.Module): U-Net with frozen weights
        """
        for param in unet.parameters():
            param.requires_grad = False
        return unet

    def freeze_decoder(self, unet):
        r"""Freeze U-Net decoder

        Args:
            unet (nn.Module): U-Net

        Returns:
            (nn.Module): U-Net with frozen decoder weights
        """
        for param in unet.decoder.parameters():
            param.requires_grad = False
        for param in unet.classifier.parameters():
            param.requires_grad = False
        return unet

    def finish(self):
        r"""Finish training, store current U-Net as old U-Net"""
        unet_new_state_dict = self.unet_new.state_dict()
        if next(self.unet_new.parameters()).is_cuda:
            device = next(self.unet_new.parameters()).device
        if self.backbone == "unet":
            self.unet_old = UNet2D(
                self.input_shape,
                self.nr_labels,
                dropout=self.unet_dropout,
                monte_carlo_dropout=self.unet_monte_carlo_dropout,
                preactivation=self.unet_preactivation,
            )
        elif self.backbone == "swinunet":
            self.unet_old = SwinUnet(
                input_shape=self.input_shape[1], nr_labels=self.nr_labels
            )
        self.unet_old.load_state_dict(unet_new_state_dict)
        self.unet_old = self.freeze_unet(self.unet_old)

        self.unet_old.to(device)

    def set_optimizers(self, optimizer=optim.SGD, lr=1e-4, weight_decay=1e-4):
        r"""Set optimizers for all modules

        Args:
            optimizer (torch.nn.optim): optimizer to use
            lr (float): learning rate to use
            weight_decay (float): weight decay
        """
        if optimizer == optim.SGD:
            self.unet_optim = optimizer(
                self.unet_new.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.unet_optim = optimizer(self.unet_new.parameters(), lr=lr)
