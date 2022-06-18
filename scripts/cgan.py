from warnings import simplefilter
import warnings
import argparse
import yaml

import launcher.pytorch_util as ptu
from launcher import logger
from launcher.launcher_util import setup_logger
from utils.r2cnn_trainer import CGANTrain

simplefilter(action='ignore', category=DeprecationWarning)


def experiment(exp_specs, device):
    if exp_specs['log']:
        setup_logger(log_dir=exp_specs['log_dir'], variant=exp_specs, log_wandb=exp_specs['wandb'])
    exp_specs['device'] = device
    with CGANTrain(exp_specs) as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file", default="scripts/cgan.yaml")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        exp_specs = yaml.load(spec_file, Loader=yaml.Loader)

    log_dir = exp_specs["log_dir"]

    if exp_specs["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)
    else:
        device = "cpu"
    experiment(exp_specs, device)