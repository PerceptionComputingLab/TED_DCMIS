import os
import time
from tqdm import tqdm
import torch
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.eval.losses.losses_distillation import LossPLOP


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class PLOPAgent(SegmentationAgent):

    def __init__(self, *args, **kwargs):
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
        super().__init__(*args, **kwargs)
        self.best_validation_epoch = 0
        self.best_validation_value = 0
        self.plop_loss = LossPLOP()

    def train(
        self,
        results,
        loss_f,
        train_dataloader,
        test_dataloader,
        config,
        init_epoch=0,
        nr_epochs=100,
        eval_datasets=dict(),
        save_path="",
        dataset_index=0,
        exp_path="",
    ):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.

        Args:
            results (mp.eval.result.Result): results object to track progress
            loss_f (mp.eval.losses.loss_abstract.LossAbstract): loss function for the segmenter
            train_dataloader (torch.utils.data.DataLoader): dataloader of training set
            test_dataloader (torch.utils.data.DataLoader): dataloader of test set
            eval_datasets (torch.utils.data.DataLoader): dataloader of evaluation set
            config (dict): configuration dictionary from parsed arguments
            init_epoch (int): initial epoch
            nr_epochs (int): number of epochs to train for
            save_path (str): save path for saving model, etc.
        """
        run_loss_print_interval = config["run_loss_print_interval"]
        save_interval = config["save_interval"]
        val_best = config["val_best"]
        self.agent_state_dict["epoch"] = init_epoch

        self.best_validation_value = 0.0
        self.best_validation_epoch = 0

        for epoch in range(init_epoch, nr_epochs):
            print("Epoch:", epoch)
            self.agent_state_dict["epoch"] = epoch

            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            acc = self.perform_training_epoch(
                loss_f, train_dataloader, config, epoch, print_run_loss=print_run_loss
            )
            if val_best:
                dice = self.track_validation_metrics(
                    idx=dataset_index,
                    loss_f=loss_f,
                    datasets=eval_datasets,
                    save_path=save_path,
                    epoch=epoch,
                    acc=acc,
                )
                print("validation dice:", dice, " lr:", get_lr(self.model.unet_optim))
                if dice > self.best_validation_value:
                    self.best_validation_value = dice
                    self.best_validation_epoch = epoch
                    self.save_state(save_path, epoch + 1)
            else:
                # Save agent and optimizer state
                if (epoch + 1) % save_interval == 0 and save_path is not None:
                    self.save_state(save_path, epoch + 1)

            self.model.unet_scheduler.step()

        if val_best:
            self.restore_state(exp_path, self.best_validation_epoch + 1)
            with open(os.path.join(exp_path, "val_track.txt"), "a+") as f:
                f.writelines(str(self.best_validation_epoch + 1) + "\n")
            print(
                "best epoch is ",
                self.best_validation_epoch + 1,
                "; best val dice is",
                self.best_validation_value,
            )

        self.track_metrics(nr_epochs, results, loss_f, eval_datasets)

        self.model.finish()

    def perform_training_epoch(
        self, loss_f, train_dataloader, config, epoch, print_run_loss=False
    ):
        r"""Perform a training epoch

        Args:
            loss_f (mp.eval.losses.loss_abstract.LossAbstract): loss function for the segmenter
            train_dataloader (torch.utils.data.DataLoader): dataloader of training set
            config (dict): configuration dictionary from parsed arguments
            print_run_loss (boolean): whether to print running loss

        Returns:
            acc (mp.eval.accumulator.Accumulator): accumulator holding losses
        """
        acc = Accumulator("loss")
        start_time = time.time()

        for data in tqdm(train_dataloader, disable=True):
            # Get data
            inputs, targets = self.get_inputs_targets(data)

            # Forward pass
            outputs = self.get_outputs(inputs)

            # Optimization step
            self.model.unet_optim.zero_grad()

            loss_seg = loss_f(outputs, targets)

            if self.model.unet_old != None:
                outputs_old = self.get_outputs_simple(inputs)
                loss_plop = self.plop_loss(outputs, outputs_old, targets)
            else:
                if loss_seg.is_cuda:
                    loss_plop = torch.zeros(1).to(loss_seg.get_device())
                else:
                    loss_plop = torch.zeros(1)

            loss = loss_seg + config["lambda_d"] * loss_plop
            loss.backward()

            self.model.unet_optim.step()

            acc.add("loss", float(loss.detach().cpu()), count=len(inputs))
            acc.add("loss_seg", float(loss_seg.detach().cpu()), count=len(inputs))
            acc.add("loss_plop", float(loss_plop.detach().cpu()), count=len(inputs))

        # self.model.unet_scheduler.step()

        if print_run_loss:
            print(
                "\nrunning loss: {} - plop {} - time/epoch {}".format(
                    acc.mean("loss"),
                    acc.mean("loss_plop"),
                    round(time.time() - start_time, 4),
                )
            )

        return acc

    def get_outputs_simple(self, inputs):
        r"""Applies a softmax transformation to the model outputs."""
        outputs = self.model.forward_old(inputs)
        outputs = softmax(outputs)
        return outputs
