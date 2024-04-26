import os
import sys

import numpy as np
from mp.experiments.experiment import Experiment
from args import parse_args_as_dict
from get import *
from mp.utils.helper_functions import seed_all
from mp.eval.losses.losses_segmentation import LossDice
import torch

torch.set_num_threads(4)
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)

if __name__ == "__main__":

    datasets_ = ["prostate"]
    approaches = ["plop"]
    backbones = ["unet"]
    loss_f = LossDice()
    metrics = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
    for dataset_ in datasets_:
        for approach in approaches:
            for backbone in backbones:
                config["experiment_name"] = (
                    dataset_ + "-" + approach + "-" + backbone
                )  # 'prostate-kddiffusion-unet'
                # config['experiment_name'] = dataset_ + '-r' + '-' + approach + '-' + backbone  # 'mm'
                # config['target_class'] = 'r'
                config["approach"] = approach
                config["dataset"] = dataset_
                config["resume_epoch"] = 40
                config["device-ids"] = "4"
                print(config)

            exp = Experiment(
                config=config,
                name=config["experiment_name"],
                notes="",
                reload_exp=(config["resume_epoch"]),
            )
            train_dataloader, test_dataloader, datasets, exp_run, label_inf = (
                get_dataset(config, exp=exp)
            )
            best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")
            best_states = []
            with open(best_states_file, "r") as f:
                for line in f.readlines():
                    best_states.append(int(line.replace("\n", "")))
            size = 150
            out_dir = "background_shift"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            model = get_model(config, nr_labels=label_inf["label_nr"])
            agent = get_agent(config, model=model, label_names=label_inf["label_names"])
            if approach == "seq":
                states_old = best_states[-1]
                agent.restore_state(exp_run.paths["states"], states_old)
            else:
                states_new = best_states[-1]
                agent.restore_state(exp_run.paths["states"], states_new)
                agent.model.finish()  # change new model with old one, otherwise it fails to restore the state
                states_new = best_states[-1]
                agent.restore_state(exp_run.paths["states"], states_new)

            uncertainty = []
            for j in range(0, len(best_states)):

                test_dataset = test_dataloader[j]
                features = agent.get_background_shift(test_dataset)

                uncertainty_diff = []
                for i, feature in enumerate(features):
                    images = feature["inputs"]
                    targets = feature["target"]
                    new_probs = feature["outputs"]
                    for sample in range(images.shape[0]):
                        image = images[sample][0]
                        target = targets[sample][1]
                        new_prob = new_probs[sample]
                        predicted = np.argmax(new_prob, axis=0)
                        misclassified = target != predicted
                        entropy = -np.sum(new_prob * np.log(new_prob + 1e-6), axis=0)
                        entropy = entropy / np.max(entropy)

                        if np.sum(misclassified) == 0:
                            continue
                        else:
                            uncertainty_sample = np.sum(
                                entropy * misclassified
                            ) / np.sum(misclassified)
                            uncertainty_diff.append(uncertainty_sample)
                            continue

                uncertainty.append(np.mean(uncertainty_diff))

            print(approach + " uncertainty: ", uncertainty)

"""
result:
prostate
seq：0.38413072097203776, 0.4006589883840904, 0.4972383785915587, 0.38588250238986715, 0.5016571520403096, 0.5407959914510456
akt：0.45027472056881923, 0.42915558572119966, 0.5036675793635842, 0.44268759668510543, 0.5125576271031702, 0.49315115895113576
kd：0.32448884265258393, 0.357195225833993, 0.46751843421539085, 0.39263390044949403, 0.3645766436321109, 0.40715944010097743
mas：0.17628126541251454, 0.2999555688816964, 0.3721556845074497, 0.3116487403591346, 0.3663483144470182, 0.18687656238565886
plop: 0.17149389491680417, 0.28008838773710354, 0.37660046467268504, 0.31423761617641327, 0.279858345548347, 0.23724976540407225

mm-i
mas uncertainty:  [0.25301220896731097, 0.26642325367403247, 0.31340247622198636, 0.3227312696264851]
seq uncertainty:  [0.5760730471390291, 0.5842209353680831, 0.6373696909534111, 0.6527687312844521]
kd uncertainty:  [0.3883382876644014, 0.4114479870081355, 0.4867034385985664, 0.4634157418586244]
akt uncertainty:  [0.5217978568437198, 0.5513720186764558, 0.6086049904938543, 0.5610067867309436]
plop uncertainty:  [0.5118577664774409, 0.5198315103637569, 0.589350139238527, 0.5960363550752685]

mm-r
plop uncertainty:  [0.5446188450211803, 0.5989395012576127, 0.55961064535226, 0.604804952247668]
akt uncertainty:  [0.5911634932325895, 0.6395962415787626, 0.6481961155662849, 0.6426226344626859]
mas uncertainty:  [0.6013288058358217, 0.642397384142038, 0.6462454301268689, 0.6324799002935746]
seq uncertainty:  [0.4907239652518729, 0.533277628965283, 0.47660760555043963, 0.5153066662986938]
kd uncertainty:  [0.474258719705713, 0.5471400901365141, 0.5326042587843935, 0.5605235832594619]

mm-o
akt uncertainty:  [0.6368418745846033, 0.6431061970843159, 0.6865985209028855, 0.6363563235911008]
mas uncertainty:  [0.5950867197812835, 0.6179007992526409, 0.6348829639835377, 0.6098119841200924]
plop uncertainty:  [0.5388622997012137, 0.5617190445404212, 0.6148527685753614, 0.5683402070293917]
seq uncertainty:  [0.44292827268449264, 0.4539756019800355, 0.4818948037037849, 0.4735828782778371]
kd uncertainty:  [0.6215375353008024, 0.6212542319708404, 0.6624320063680352, 0.6274086879854167]

"""
