import os
import sys

from mp.experiments.experiment import Experiment
from args import parse_args_as_dict
from get import *
from mp.utils.helper_functions import seed_all
from mp.eval.losses.losses_segmentation import LossDice
from torchvision import transforms
import torch
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm

torch.set_num_threads(4)
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)

if __name__ == '__main__':

    transf = transforms.ToTensor()
    datasets_ = ['prostate']
    approaches = ['kddiffusion']
    backbones = ['unet']
    loss_f = LossDice()
    metrics = ['ScoreDice', 'ScoreIoU', 'ScoreHausdorff']

    font_path = 'times-new-roman.ttf'
    font_prop = fm.FontProperties(fname=font_path, size=16)

    for dataset_ in datasets_:
        for approach in approaches:
            for backbone in backbones:
                config['experiment_name'] = dataset_ + '-' + approach + '-' + backbone  # 'prostate-kddiffusion-unet'
                config['approach'] = approach
                config['dataset'] = dataset_
                config['resume_epoch'] = 40
                config['device-ids'] = '4'
                config['ablation'] = True
                print(config)

                exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=(
                    config['resume_epoch']
                ))
                train_dataloader, _, datasets, exp_run, label_inf = get_dataset(config, exp=exp)
                best_states_file = os.path.join(exp_run.paths['states'], 'val_track.txt')
                best_states = []
                with open(best_states_file, 'r') as f:
                    for line in f.readlines():
                        best_states.append(int(line.replace('\n', '')))

                model = get_model(config, nr_labels=label_inf['label_nr'])

                agent = get_agent(config, model=model, label_names=label_inf['label_names'])

                states_old = best_states[0]
                agent.restore_state(exp_run.paths['states'], states_old)
                agent.model.finish()  # change new model with old one, otherwise it fails to restore the state
                states_new = best_states[1]
                agent.restore_state(exp_run.paths['states'], states_new)

                dataset_list = []
                for ds_name, _ in datasets.items():
                    if ds_name[0] not in dataset_list:
                        dataset_list.append(ds_name[0])
                train_dataset = train_dataloader[1]
                features = agent.ablatcion(train_dataset, dataset_list, config, 1)

                X_old = features['outputs_old']
                X_old_ska = features['outputs_old_ska_img']
                X_old_diffusion = features['outputs_old_pseudo_img']
                X_old_injection = features['outputs_old_injection_img']
                X_old_aug = features['outputs_old_aug_img']

                X_new = features['outputs']
                X_new_ska = features['outputs_new_ska_img']
                X_new_diffusion = features['outputs_new_pseudo_img']
                X_new_injection = features['outputs_new_injection_img']
                X_new_aug = features['outputs_new_aug_img']

                del features
                X_old = torch.cat(X_old, dim=0)
                X_old_ska = torch.cat(X_old_ska, dim=0)
                X_old_diffusion = torch.cat(X_old_diffusion, dim=0)
                X_old_injection = torch.cat(X_old_injection, dim=0)
                X_old_aug = torch.cat(X_old_aug, dim=0)
                X_new = torch.cat(X_new, dim=0)
                X_new_ska = torch.cat(X_new_ska, dim=0)
                X_new_diffusion = torch.cat(X_new_diffusion, dim=0)
                X_new_injection = torch.cat(X_new_injection, dim=0)
                X_new_aug = torch.cat(X_new_aug, dim=0)

                X_old = X_old.detach().numpy().reshape(X_old.shape[0], -1)
                X_old_ska = X_old_ska.detach().numpy().reshape(X_old_ska.shape[0], -1)
                X_old_diffusion = X_old_diffusion.detach().numpy().reshape(X_old_diffusion.shape[0], -1)
                X_old_injection = X_old_injection.detach().numpy().reshape(X_old_injection.shape[0], -1)
                X_old_aug = X_old_aug.detach().numpy().reshape(X_old_aug.shape[0], -1)
                X_new = X_new.detach().numpy().reshape(X_new.shape[0], -1)
                X_new_ska = X_new_ska.detach().numpy().reshape(X_new_ska.shape[0], -1)
                X_new_diffusion = X_new_diffusion.detach().numpy().reshape(X_new_diffusion.shape[0], -1)
                X_new_injection = X_new_injection.detach().numpy().reshape(X_new_injection.shape[0], -1)
                X_new_aug = X_new_aug.detach().numpy().reshape(X_new_aug.shape[0], -1)

                out_dir = 'ska_augmentation'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                # distribution of the old knowledge
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_old)
                dict_ = {}
                dict_['x'] = X_pca[:, 0]
                dict_['y'] = X_pca[:, 1]
                df = pd.DataFrame(dict_)
                df.to_csv(os.path.join(out_dir, 'old_knowledge.csv'), index=False)

                # distribution of the old knowledge with ska augmentation
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_old_ska)
                dict_ = {}
                dict_['x'] = X_pca[:, 0]
                dict_['y'] = X_pca[:, 1]
                df = pd.DataFrame(dict_)
                df.to_csv(os.path.join(out_dir, 'old_knowledge_ska.csv'), index=False)

                # distribution of the old knowledge with diffusion augmentation
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_old_diffusion)
                dict_ = {}
                dict_['x'] = X_pca[:, 0]
                dict_['y'] = X_pca[:, 1]
                df = pd.DataFrame(dict_)
                df.to_csv(os.path.join(out_dir, 'old_knowledge_diffusion.csv'), index=False)

                # distribution of the old knowledge with injection augmentation
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_old_injection)
                dict_ = {}
                dict_['x'] = X_pca[:, 0]
                dict_['y'] = X_pca[:, 1]
                df = pd.DataFrame(dict_)
                df.to_csv(os.path.join(out_dir, 'old_knowledge_injection.csv'), index=False)

                # distribution of the old knowledge with augmentation
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_old_aug)
                dict_ = {}
                dict_['x'] = X_pca[:, 0]
                dict_['y'] = X_pca[:, 1]
                df = pd.DataFrame(dict_)
                df.to_csv(os.path.join(out_dir, 'old_knowledge_aug.csv'), index=False)
