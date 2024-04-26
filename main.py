import os
import sys

from mp.experiments.experiment import Experiment
from mp.eval.result import Result
from mp.utils.tensorboard import create_writer
from args import parse_args_as_dict
from get import *
from mp.utils.helper_functions import seed_all
import torch.optim as optim
import torch

torch.set_num_threads(4)
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)
print(config)

exp = Experiment(
    config=config,
    name=config["experiment_name"],
    notes="",
    reload_exp=(config["resume_epoch"] is not None),
)
train_dataloader, test_dataloader, datasets, exp_run, label_inf = get_dataset(
    config, exp=exp
)
model = get_model(config, nr_labels=label_inf["label_nr"])
loss_f = get_loss_type(config)
results = Result()

agent = get_agent(config, model=model, label_names=label_inf["label_names"])
agent.summary_writer = create_writer(
    path=os.path.join(exp_run.paths["states"], ".."), init_epoch=0
)

nr_epochs = 0
config["continual"] = False
for idx, dataloader in enumerate(train_dataloader):
    init_epoch = nr_epochs
    nr_epochs = config["epochs"] + init_epoch

    if config["continual"]:
        model.set_optimizers(optim.Adam, lr=config["lr_2"])
    else:
        model.set_optimizers(optim.Adam, lr=config["lr"])

    model.unet_scheduler = optim.lr_scheduler.StepLR(
        model.unet_optim, step_size=1, gamma=0.99
    )
    agent.train(
        results=results,
        loss_f=loss_f,
        train_dataloader=dataloader,
        test_dataloader=dataloader,
        config=config,
        init_epoch=init_epoch,
        nr_epochs=nr_epochs,
        eval_datasets=datasets,
        save_path=exp_run.paths["states"],
        dataset_index=idx,
        exp_path=exp_run.paths["states"],
    )
    if config["approach"] not in ["seq"]:
        config["continual"] = True
if config["dataset"] == "mcm":
    config["dataset"] = "cardiac"
exp_run.finish(
    results=results, plot_metrics=["Mean_ScoreDice[{}]".format(config["dataset"])]
)
