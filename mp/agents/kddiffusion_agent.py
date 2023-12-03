import os
import time
from tqdm import tqdm
import torch

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax

from data_prep.diffusion_utils.dataset_diffusion import SimpleDataset
from torch.utils.data import DataLoader
import random


class KDDiffusionAgent(SegmentationAgent):

    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['ScoreDice', 'ScoreIoU', 'ScoreHausdorff']
        super().__init__(*args, **kwargs)

    def ablation(self, train_dataloader, dataset_list, config, dataset_index):

        pseudo_data = SimpleDataset(subset=dataset_list[dataset_index],
                                    size=(config['input_dim_hw'], config['input_dim_hw']))
        pseudo_data_loader = DataLoader(pseudo_data, batch_size=config['batch_size'], shuffle=True)

        features = {'outputs': [], 'outputs_old': [],
                    'outputs_new_pseudo_img': [], 'outputs_old_pseudo_img': [],
                    'outputs_new_injection_img': [], 'outputs_old_injection_img': [],
                    'outputs_new_ska_img': [], 'outputs_old_ska_img': [],
                    'outputs_new_aug_img': [], 'outputs_old_aug_img': [], }

        with torch.no_grad():
            for data in tqdm(train_dataloader, disable=True):
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass
                outputs = self.get_new_bottom_block(inputs)
                features['outputs'].append(outputs.to('cpu'))

                outputs_old = self.get_old_bottom_block(inputs)
                features['outputs_old'].append(outputs_old.to('cpu'))
                # load pseudo images
                pseudo_img = next(iter(pseudo_data_loader))
                try:
                    pseudo_img = pseudo_img.to(inputs.device)
                except Exception as e:
                    pseudo_data_loader = DataLoader(pseudo_data, batch_size=config['batch_size'], shuffle=True)
                    pseudo_img = next(iter(pseudo_data_loader))
                    pseudo_img = pseudo_img.to(inputs.device)

                outputs_new_pseudo_img = self.get_new_bottom_block(pseudo_img)
                features['outputs_new_pseudo_img'].append(outputs_new_pseudo_img.to('cpu'))

                outputs_old_pseudo_img = self.get_old_bottom_block(pseudo_img)
                features['outputs_old_pseudo_img'].append(outputs_old_pseudo_img.to('cpu'))
                # noise add
                noise_img = torch.randn_like(inputs).to(inputs.device) + inputs
                outputs_new_noise_img = self.get_new_bottom_block(noise_img)
                features['outputs_new_injection_img'].append(outputs_new_noise_img.to('cpu'))
                outputs_old_noise_img = self.get_old_bottom_block(noise_img)
                features['outputs_old_injection_img'].append(outputs_old_noise_img.to('cpu'))

                # noise only
                noise_img = torch.randn_like(inputs).to(inputs.device)
                outputs_new_noise_img = self.get_new_bottom_block(noise_img)
                features['outputs_new_ska_img'].append(outputs_new_noise_img.to('cpu'))
                outputs_old_noise_img = self.get_old_bottom_block(noise_img)
                features['outputs_old_ska_img'].append(outputs_old_noise_img.to('cpu'))

                # data augmentation
                aug_img = torch.flip(inputs, [2])
                aug_img = torch.rot90(aug_img, k=1, dims=[2, 3])
                aug_img = aug_img * random.uniform(0.8, 1.2)
                aug_img = aug_img.to(inputs.device)
                outputs_new_aug_img = self.get_new_bottom_block(aug_img)
                features['outputs_new_aug_img'].append(outputs_new_aug_img.to('cpu'))
                outputs_old_aug_img = self.get_old_bottom_block(aug_img)
                features['outputs_old_aug_img'].append(outputs_old_aug_img.to('cpu'))

                del inputs, targets, outputs_old, outputs_old_pseudo_img, outputs_old_noise_img, pseudo_img, noise_img,
        return features

    def get_old_bottom_block(self, inputs):
        return self.model.unet_old.bottom_block(self.model.unet_old.encoder(inputs)[1])

    def get_new_bottom_block(self, inputs):
        return self.model.unet_new.bottom_block(self.model.unet_new.encoder(inputs)[1])

    def train(self, results, loss_f, train_dataloader, test_dataloader, config, init_epoch=0, nr_epochs=100,
              eval_datasets=dict(), save_path='', dataset_index=0, exp_path=''):
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

        run_loss_print_interval = config['run_loss_print_interval']
        save_interval = config['save_interval']
        val_best = config['val_best']
        self.agent_state_dict['epoch'] = init_epoch

        self.best_validation_value = 0.
        self.best_validation_epoch = 0

        dataset_list = []
        for ds_name, _ in eval_datasets.items():
            if ds_name[0] not in dataset_list:
                dataset_list.append(ds_name[0])
        # make a dataloader on pseudo data
        if config['continual']:
            pseudo_data = SimpleDataset(subset=dataset_list[dataset_index],
                                        size=(config['input_dim_hw'], config['input_dim_hw']))
        else:
            pseudo_data = None

        for epoch in range(init_epoch, nr_epochs):
            print('Epoch:', epoch)
            self.agent_state_dict['epoch'] = epoch

            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            acc = self.perform_training_epoch(loss_f, train_dataloader, config, epoch,
                                              print_run_loss=print_run_loss,
                                              pseudo_data=pseudo_data)
            if val_best:
                dice = self.track_validation_metrics(dataset_index, loss_f, eval_datasets, save_path, epoch, acc)
                print('validation dice:', dice)
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
            with open(os.path.join(exp_path, 'val_track.txt'), 'a+') as f:
                f.writelines(str(self.best_validation_epoch + 1) + '\n')
            print('best epoch is ', self.best_validation_epoch + 1, '; best val dice is', self.best_validation_value)

        self.track_metrics(nr_epochs, results, loss_f, eval_datasets)

        self.model.finish()

    def perform_training_epoch(self, loss_f, train_dataloader, config, epoch, print_run_loss=False,
                               pseudo_data=None):
        r"""Perform a training epoch
        
        Args:
            loss_f (mp.eval.losses.loss_abstract.LossAbstract): loss function for the segmenter
            train_dataloader (torch.utils.data.DataLoader): dataloader of training set
            config (dict): configuration dictionary from parsed arguments
            print_run_loss (boolean): whether to print running loss
        
        Returns:
            acc (mp.eval.accumulator.Accumulator): accumulator holding losses
        """
        acc = Accumulator('loss')
        start_time = time.time()
        if pseudo_data:
            pseudo_data_loader = DataLoader(pseudo_data, batch_size=config['batch_size'], shuffle=True)
        else:
            pseudo_data_loader = None
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
                loss_distill_image = self.multi_class_cross_entropy_no_softmax(outputs, outputs_old)

                # load pseudo images
                pseudo_img = next(iter(pseudo_data_loader))
                try:
                    pseudo_img = pseudo_img.to(inputs.device)  # + torch.randn_like(pseudo_img).to(
                    #  inputs.device)  # v2 add noise
                except Exception as e:
                    pseudo_data_loader = DataLoader(pseudo_data, batch_size=config['batch_size'], shuffle=True)
                    pseudo_img = next(iter(pseudo_data_loader))
                    pseudo_img = pseudo_img.to(inputs.device)  # + torch.randn_like(pseudo_img).to(
                    #  inputs.device)  # v2 add noise
                outputs_new_pseudo_img = self.get_outputs(pseudo_img)
                outputs_old_pseudo_img = self.get_outputs_simple(pseudo_img)
                loss_distill_noise = self.multi_class_cross_entropy_no_softmax(outputs_new_pseudo_img,
                                                                               outputs_old_pseudo_img)
                loss_distill = loss_distill_noise + loss_distill_image
            else:
                if loss_seg.is_cuda:
                    loss_distill = torch.zeros(1).to(loss_seg.get_device())
                else:
                    loss_distill = torch.zeros(1)

            loss = loss_seg + config['lambda_d'] * loss_distill
            loss.backward()

            self.model.unet_optim.step()

            acc.add('loss', float(loss.detach().cpu()), count=len(inputs))
            acc.add('loss_seg', float(loss_seg.detach().cpu()), count=len(inputs))
            acc.add('loss_distill', float(loss_distill.detach().cpu()), count=len(inputs))

        # self.model.unet_scheduler.step()

        if print_run_loss:
            print('\nrunning loss: {} - distill {} - time/epoch {}'.format(acc.mean('loss'), acc.mean('loss_distill'),
                                                                           round(time.time() - start_time, 4)))

        return acc

    def multi_class_cross_entropy_no_softmax(self, prediction, target):
        r"""Stable Multiclass Cross Entropy with Softmax

        Args:
            prediction (torch.Tensor): network outputs w/ softmax
            target (torch.Tensor): label OHE

        Returns:
            (torch.Tensor) computed loss 
        """
        return (-(target * torch.log(prediction)).sum(dim=-1)).mean()

    def get_outputs_simple(self, inputs):
        r"""Applies a softmax transformation to the model outputs.
        """
        outputs = self.model.forward_old(inputs)
        outputs = softmax(outputs)
        return outputs
