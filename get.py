import torch.utils.data

from mp.data.datasets.ds_mr_prostate import Prostate
from mp.data.datasets.ds_mr_cardiac_mm import Cardiac
from mp.data.data import Data
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from torch.utils.data import DataLoader

from mp.models.continual.kd import KD
from mp.models.continual.mas import MAS
from mp.models.continual.mib import MIB
from mp.models.continual.plop import PLOP

from mp.eval.losses.losses_segmentation import LossDiceBCE, LossBCE, LossDice

from mp.agents.kd_agent import KDAgent
from mp.agents.mas_agent import MASAgent
from mp.agents.ewc_agent import EWCAgent
from mp.agents.mib_agent import MIBAgent
from mp.agents.plop_agent import PLOPAgent
from mp.agents.unet_agent import UNETAgent

from mp.agents.ska_agent import SKAAgent
from mp.agents.akt_agent import AKTAgent
from mp.agents.ted_agent import TEDAgent


def get_dataset(config, exp):
    data = Data()
    subset_list = []
    if config['dataset'] == 'prostate':
        subset_list = ['RUNMC', 'BMC', 'I2CVB', 'UCL', 'BIDMC', 'HK']
        for name in subset_list:
            dataset_domain = Prostate(subset=name)
            dataset_domain.name = name
            data.add_dataset(dataset_domain)
    elif config['dataset'] == 'mm':
        subset_list = ['Siemens', 'Philips', 'GE', 'Canon']
        target = {'i': 1, 'o': 2, 'r': 3}
        for name in subset_list:
            dataset_domain = Cardiac(subset=name, target=target[config['target_class']])
            dataset_domain.name = name
            data.add_dataset(dataset_domain)
    exp.set_data_splits(data)
    exp_run = exp.get_run(0, reload_exp_run=(
            config['resume_epoch'] is not None
    ))
    datasets = {}
    for item in data.datasets.items():
        ds_name, ds = item
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            data_ixs = data_ixs[:config['n_samples']]
            if len(data_ixs) > 0:
                datasets[(ds_name, split)] = PytorchSeg2DDataset(
                    dataset=ds, ix_lst=data_ixs, size=config['input_shape'], aug_key='none',
                    resize=(not config['no_resize'])
                )

    if config['approach'] in ['joint']:
        joint_dataset = torch.utils.data.ConcatDataset(datasets[(name, 'train')] for name in subset_list)
        train_dataloaders = DataLoader(dataset=joint_dataset, batch_size=config['batch_size'], shuffle=True,
                                       drop_last=False, pin_memory=False,
                                       num_workers=len(config['device_ids']) * config['n_workers'])
        joint_dataset = torch.utils.data.ConcatDataset(datasets[(name, 'test')] for name in subset_list)
        test_dataloaders = DataLoader(dataset=joint_dataset, batch_size=config['batch_size'], shuffle=False,
                                      drop_last=False, pin_memory=False,
                                      num_workers=len(config['device_ids']) * config['n_workers'])
        return [train_dataloaders], [test_dataloaders], datasets, exp_run, {'label_nr': data.nr_labels,
                                                                            'label_names': data.label_names}

    train_dataloaders = []
    test_dataloaders = []
    for name in subset_list:
        train_dataloaders.append(
            DataLoader(dataset=datasets[(name, 'train')], batch_size=config['batch_size'], shuffle=True,
                       drop_last=False, pin_memory=False, num_workers=len(config['device_ids']) * config['n_workers']))
        test_dataloaders.append(
            DataLoader(dataset=datasets[(name, 'test')], batch_size=config['batch_size'], shuffle=False,
                       drop_last=False, pin_memory=False, num_workers=len(config['device_ids']) * config['n_workers']))

    return train_dataloaders, test_dataloaders, datasets, exp_run, {'label_nr': data.nr_labels,
                                                                    'label_names': data.label_names}


def get_model(config, nr_labels):
    models = {'mas': MAS, 'ewc': MAS, 'kd': KD, 'mib': MIB, 'plop': PLOP, 'seq': MAS, 'joint': MAS,
              'ska': KD, 'akt': KD, 'ted': KD}
    model = models[config['approach']](input_shape=config['input_shape'], nr_labels=nr_labels,
                                       backbone=config['backbone'],
                                       unet_dropout=config['unet_dropout'],
                                       unet_monte_carlo_dropout=config['unet_monte_carlo_dropout'],
                                       unet_preactivation=config['unet_preactivation'])
    model.to(config['device'])

    return model


def get_loss_type(config):
    if config['loss_type'] == 'dice':
        return LossDice(device=config['device'])
    elif config['loss_type'] == 'bce':
        return LossBCE(device=config['device'])
    elif config['loss_type'] == 'dice_bce':
        loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=config['device'])
        # loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], device=config['device'])
        return loss_g


def get_agent(config, model, label_names):
    agents = {'mas': MASAgent, 'ewc': EWCAgent, 'kd': KDAgent, 'mib': MIBAgent, 'plop': PLOPAgent, 'seq': UNETAgent,
              'joint': UNETAgent, 'ska': SKAAgent, 'akt': AKTAgent, 'ted': TEDAgent}
    agent = agents[config['approach']](model=model, label_names=label_names, device=config['device'])
    return agent
