import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mp.experiments.experiment import Experiment
from args import parse_args_as_dict
from get import *
from mp.utils.helper_functions import seed_all
from mp.eval.losses.losses_segmentation import LossDice
from mp.eval.evaluate import ds_metrics
from torchvision import transforms
import torch
import SimpleITK as sitk
import matplotlib.font_manager as fm

torch.set_num_threads(4)
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)


def dice_coefficient(y_true, y_pred):
    """计算Dice系数"""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1e-5) / (np.sum(y_true) + np.sum(y_pred) + 1e-5)


def iou(y_true, y_pred):
    """计算IOU"""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + 1e-5) / (union + 1e-5)


def hd95(y_true, y_pred):
    """计算Hausdorff Distance 95"""
    from scipy.spatial.distance import directed_hausdorff
    u_hausdorff = directed_hausdorff(y_true, y_pred)[0]
    v_hausdorff = directed_hausdorff(y_pred, y_true)[0]
    return np.percentile([u_hausdorff, v_hausdorff], 95)


def compute_metrics(y_true, y_pred):
    return {
        "Dice": dice_coefficient(y_true, y_pred),
        "IOU": iou(y_true, y_pred),
        "HD95": hd95(y_true, y_pred)
    }


if __name__ == '__main__':
    font_path = 'times-new-roman.ttf'
    font_prop = fm.FontProperties(fname=font_path, size=16)
    transf = transforms.ToTensor()
    datasets_ = ['mm']
    approaches = ['seq', 'kd', 'kdenhanced']
    dicelist = {k: [] for k in approaches}
    backbones = ['unet']
    loss_f = LossDice()
    metrics = ['ScoreDice', 'ScoreIoU', 'ScoreHausdorff']
    for dataset_ in datasets_:
        for approach in approaches:
            for backbone in backbones:
                config['experiment_name'] = dataset_ + '-i' + '-' + approach + '-' + backbone
                if approach == 'kdenhanced':
                    config['experiment_name'] = dataset_ + '-i' + '-' + approach + 'batch' + '-' + backbone
                # config['experiment_name'] = dataset_ + '-r' + '-' + approach + 'batch' + '-' + backbone  # 'mm'
                config['target_class'] = 'i'
                config['approach'] = approach
                config['dataset'] = dataset_
                config['resume_epoch'] = 40
                config['device-ids'] = '4'
                print(config)

            exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=(
                config['resume_epoch']  # TODO
            ))
            train_dataloader, test_dataloader, datasets, exp_run, label_inf = get_dataset(config, exp=exp)
            best_states_file = os.path.join(exp_run.paths['states'], 'val_track.txt')
            best_states = []
            with open(best_states_file, 'r') as f:
                for line in f.readlines():
                    best_states.append(int(line.replace('\n', '')))
            size = 150
            out_dir = 'image'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            model = get_model(config, nr_labels=label_inf['label_nr'])
            agent = get_agent(config, model=model, label_names=label_inf['label_names'])
            if approach == 'seq':
                states_old = best_states[-1]
                agent.restore_state(exp_run.paths['states'], states_old)
            else:
                states_new = best_states[-1]
                agent.restore_state(exp_run.paths['states'], states_new)
                agent.model.finish()  # change new model with old one, otherwise it fails to restore the state
                states_new = best_states[-1]
                agent.restore_state(exp_run.paths['states'], states_new)

            # illustrate the changing process of one sample from first domain
            for i in range(0, len(best_states)):
                continue
                states_new = best_states[i]
                agent.restore_state(exp_run.paths['states'], states_new)
                features = agent.get_background_shift(test_dataloader[0])
                feature = features[6]  # a batch sample from first domain

                image = feature['inputs'][0][0]
                target = feature['target'][0][1]
                new_prob = feature['outputs'][0]
                predicted = np.argmax(new_prob, axis=0)

                alpha_target = np.where(target == 1, 0.5, 0)
                alpha_predicted = np.where(predicted == 1, 0.5, 0)
                print(compute_metrics(alpha_target * 2, alpha_predicted * 2))
                dicelist[approach].append(compute_metrics(alpha_target * 2, alpha_predicted * 2)['Dice'])
                ground_truth_rgba = np.zeros((target.shape[0], target.shape[1], 4))
                ground_truth_rgba[..., 0] = 1
                ground_truth_rgba[..., 3] = alpha_target
                predicted_rgba = np.zeros((predicted.shape[0], predicted.shape[1], 4))
                predicted_rgba[..., 2] = 1
                predicted_rgba[..., 3] = alpha_predicted

                fig, ax = plt.subplots()
                ax.imshow(np.rot90(image, 3), cmap='gray')

                ax.imshow(np.rot90(ground_truth_rgba, 3))
                ax.imshow(np.rot90(predicted_rgba, 3))

                legend_ground_truth = plt.Rectangle((0, 0), 1, 1, fc=(1, 0, 0, 0.5))
                legend_predicted = plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1, 0.5))

                plt.title('Ground truth and prediction' + str(i), fontproperties=font_prop)
                plt.axis('off')
                plt.show()

                # for ground truth
                fig, ax = plt.subplots()
                ax.imshow(np.rot90(image, 3), cmap='gray')
                ax.imshow(np.rot90(ground_truth_rgba, 3))
                # legend_ground_truth = plt.Rectangle((0, 0), 1, 1, fc=(1, 0, 0, 0.5))
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            # illustrate the results of all domains after the last domain
            states_new = best_states[-1]
            agent.restore_state(exp_run.paths['states'], states_new)
            for i in range(0, len(best_states)):
                """
                prostate:
                domain 0: 6-1
                domain 1: 11-0
                domain 2: 7-3
                domain 3: 2-3
                domain 4: 5-1
                domain 5: 15-1
                mm-r:
                domain 0: 13-1
                domain 1: 143-2
                domain 2: 23-2
                domain 3: 49-2
                mm-i:
                domain 0: 28-1
                domain 1: 142-1
                domain 2: 51-2
                domain 3: 48-2
                mm-o:
                
                """
                if i != 3:
                    continue
                test_dataset = test_dataloader[i]
                features = agent.get_background_shift(test_dataset)

                feature = features[48]  # a batch sample id = 2

                image = feature['inputs'][2][0]  # sample id = 3
                target = feature['target'][2][1]
                new_prob = feature['outputs'][2]
                predicted = np.argmax(new_prob, axis=0)

                alpha_target = np.where(target == 1, 0.5, 0)
                alpha_predicted = np.where(predicted == 1, 0.5, 0)
                print(compute_metrics(alpha_target * 2, alpha_predicted * 2))
                dicelist[approach].append(compute_metrics(alpha_target * 2, alpha_predicted * 2)['Dice'])
                ground_truth_rgba = np.zeros((target.shape[0], target.shape[1], 4))
                ground_truth_rgba[..., 0] = 1
                ground_truth_rgba[..., 3] = alpha_target
                predicted_rgba = np.zeros((predicted.shape[0], predicted.shape[1], 4))
                predicted_rgba[..., 2] = 1
                predicted_rgba[..., 3] = alpha_predicted

                fig, ax = plt.subplots()
                ax.imshow(np.rot90(image, 3), cmap='gray')

                ax.imshow(np.rot90(ground_truth_rgba, 3))
                ax.imshow(np.rot90(predicted_rgba, 3))

                plt.title('domain ' + str(i) + " " + approach, fontproperties=font_prop)
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                fig, ax = plt.subplots()
                ax.imshow(np.rot90(image, 3), cmap='gray')
                ax.imshow(np.rot90(ground_truth_rgba, 3))
                plt.axis('off')
                plt.tight_layout()
    print(dicelist)
    for approach in approaches:
        print(approach, dicelist[approach])
