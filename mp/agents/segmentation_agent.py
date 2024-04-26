# ------------------------------------------------------------------------------
# A standard segmentation agent, which performs softmax in the outputs.
# ------------------------------------------------------------------------------

import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from mp.eval.evaluate import ds_losses_metrics
from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax
from mp.utils.pytorch.pytorch_load_restore import (
    load_model_state,
    save_model_state_dataparallel,
)
from mp.visualization.visualize_imgs import plot_3d_segmentation
from mp.utils.load_restore import pkl_dump, pkl_load


class SegmentationAgent(Agent):
    r"""An Agent for segmentation models."""

    def __init__(self, *args, **kwargs):
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
        super().__init__(*args, **kwargs)
        self.best_validation_value = 0.0
        self.best_validation_epoch = 0

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        outputs = softmax(outputs).clamp(min=1e-08, max=1.0 - 1e-08)
        return outputs

    def track_visualization(self, dataloader, save_path, epoch, config, phase="train"):
        r"""Creates visualizations and tracks them in tensorboard.

        Args:
            dataloader (Dataloader): dataloader to draw sample from
            save_path (string): path for the images to be saved (one folder up)
            epoch (int): current epoch
            config (dict): configuration dictionary from parsed arguments
            phase (string): either "test" or "train"
        """
        for imgs in dataloader:
            x_i, y_i = self.get_inputs_targets(imgs)
            x_i_seg = self.get_outputs(x_i)
            break

        # select sample with guaranteed segmentation label
        sample_idx = 0
        for i, y_ in enumerate(y_i):
            if len(torch.nonzero(y_[1])) > 0:
                sample_idx = i
                break
        x_i_img = x_i[sample_idx].unsqueeze(0)

        # segmentation
        x_i_seg = x_i_seg[sample_idx][1].unsqueeze(0).unsqueeze(0)
        threshold = 0.5
        x_i_seg_mask = (x_i_seg > threshold).int()
        y_i_seg_mask = y_i[sample_idx][1].unsqueeze(0).unsqueeze(0).int()

        save_path = os.path.join(save_path, "..", "imgs")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_pred = os.path.join(save_path, f"e_{epoch:06d}_{phase}_pred.png")
        save_path_label = os.path.join(save_path, f"e_{epoch:06d}_{phase}_label.png")

        plot_3d_segmentation(
            x_i_img,
            x_i_seg_mask,
            save_path=save_path_pred,
            img_size=(256, 256),
            alpha=0.5,
        )
        plot_3d_segmentation(
            x_i_img,
            y_i_seg_mask,
            save_path=save_path_label,
            img_size=(256, 256),
            alpha=0.5,
        )

        image = Image.open(save_path_pred)
        image = TF.to_tensor(image)
        self.writer_add_image(f"imgs_{phase}/pred", image, epoch)

        image = Image.open(save_path_label)
        image = TF.to_tensor(image)
        self.writer_add_image(f"imgs_{phase}/label", image, epoch)

    def get_background_shift(self, train_dataloader):

        image_all = []
        with torch.no_grad():
            for data in tqdm(train_dataloader):
                dicts = {}
                # get data
                inputs, target = self.get_inputs_targets(data)
                dicts["inputs"] = inputs.detach().cpu().numpy()
                dicts["target"] = target.detach().cpu().numpy()
                outputs = self.get_outputs(inputs)
                dicts["outputs"] = outputs.detach().cpu().numpy()
                image_all.append(dicts)
                # return image_all
        return image_all
