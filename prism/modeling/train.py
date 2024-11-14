from pathlib import Path
import os

from loguru import logger
from tqdm import tqdm



from prism.config.config import (
    MODELS_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR,
      PROCESSED_DATA_DIR, sweep_config, tune_config
)
from prism.dataloader import dataset

from prism.modeling.model import CL as Model

import torch
from torch.optim.lr_scheduler import LambdaLR
import wandb

import numpy as np
import pandas as pd

from prism.utils.general import (
    get_repo_name, set_random_seed
)


class Trainer():
    def __init__(self, config=None):
        self.main_config = config

    def run_training(self):
        with wandb.init(mode='disabled' if self.main_config.get("debug", False) else 'online'):
            if self.main_config["seed"] is not None:
                set_random_seed(self.main_config["seed"])
            
            self.best_pred = np.inf
            self.best_accuracy = 0
            self.best_epoch = 0

            #self.build_dataset(config=wandb.config)
            self.build_model(config=wandb.config)
            self.build_optimizer(config=wandb.config)
            self.build_scheduler(config=wandb.config)
            self.train(config=wandb.config)


    def build_dataset(self, config):
        pass

    def build_optimizer(self, config):
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum
            )

        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )        
    
    def build_scheduler(self, config):
        # Create a combined scheduler with linear warmup and exponential decay
        
        if config.scheduler == 'LambdaLR':
            # Calculate the multiplier to scale from base_lr to max_lr
            lr_multiplier = config.max_lr / config.base_lr

            self.scheduler = LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: ((lr_multiplier - 1) * epoch / config.l_e + 1)
                if epoch < config.l_e else
                lr_multiplier * (config.l_b ** (epoch - config.l_e))
            )
        elif config.scheduler == "Cyclic":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer=self.optimizer,
                base_lr=config.base_lr,
                max_lr=config.max_lr,
                step_size_up=72,
                mode='exp_range',
                gamma=0.96,
                scale_mode='cycle',
                cycle_momentum=False
            )

    def build_model(self, config):
        self.model = Model(
            in_channels=config.in_channels,
            h_dim=config.h_dim,
            projection_dim=config.projection_dim
        ).to(self.main_config['device'])
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(self.main_config["device"])

if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_DIR / "csv", exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR / "hdf", exist_ok=True)

    trainer = Trainer(tune_config)
    if tune_config["tune"]:
        wandb.login(key=tune_config["wandb_key"])
        sweep_id = wandb.sweep(sweep_config, project=get_repo_name())
        wandb.agent(sweep_id, trainer.run_training, count=tune_config['count'])
    else:
        trainer.run_training()
    