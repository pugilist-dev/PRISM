# System imports
from pathlib import Path
import os

# I/O and logging
from loguru import logger
from tqdm import tqdm
import h5py
import wandb

# Configuration
from prism.config.config import (
    MODELS_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR,
      PROCESSED_DATA_DIR, sweep_config, tune_config, aug_params
)

# Data and proprocessing imports
from prism.dataloader import dataset

# Model Imports
import torch
from torch.optim.lr_scheduler import LambdaLR
from prism.modeling.model import CL as Model

# Ananlysis imports
import numpy as np
import pandas as pd

# Utils imports
from prism.utils.general import (
    get_repo_name, set_random_seed
)
from prism.utils.metrics import Metrics


class Trainer():
    def __init__(self, config=None, aug_params=None):
        self.main_config = config
        self.aug_params = aug_params

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
            projection_dim=config.projection_dim,
            aug_params = self.aug_params
        ).to(self.main_config['device'])
    
    def train(self, config):
        for epoch in range(config.epochs):
            self.model.train()
            self.model.encoder.train()
            train_loss = 0
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.main_config['device'])
                self.optimizer.zero_grad()
                z_i, z_j, _, _ = self.model(data)
                loss = self.model.loss(z_i, z_j, config.temperature)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()

            train_loss /= len(self.train_loader)
            print(f"Epoch {epoch} Loss: {train_loss}",end="\t")
            wandb.log({"train_loss": train_loss})

            val_loss, h_i = self.validate(config.temperature)
            self.best_pred = val_loss

            h_i = pd.DataFrame(h_i)
            metric = Metrics(h_i)
            ari = metric.ari_score_cos
            nmi = metric.nmi
            ari_07 = metric.ari_score_cos_07
            nmi_07 = metric.nmi_07
            predicted = metric.cos_predicted_labels
            h_i["predicted"] = predicted
            h_i["predicted_0.7"] = metric.cos_predicted_labels_res_07
            wandb.log({
                "ari": ari, "ari_07": ari_07,
                "nmi": nmi, "nmi_07": nmi_07
            })

            if ari > self.best_ari:
                self.best_ari = ari
                self.best_epoch = epoch
                self.save_checkpoint(self.model, epoch,
                    dir=self.main_config['model_path'])
                wandb.log({
                    "best_ari": ari, "best_ari_07": ari_07,
                    "best_nmi": nmi, "best_nmi_07": nmi_07
                })

                if epoch > 0:
                    self.remove_most_recent_file(self.main_config['output_csv'])
                    self.remove_most_recent_file(self.main_config['output_hdf'])

                fname = f"ep_{epoch}_loss_{self.best_pred:.4f}_ari_{self.best_ari:.4f}"
                fcsv = os.path.join(self.main_config['output_csv'], f'{fname}.csv')
                fhdf = os.path.join(self.main_config['output_hdf'], f'{fname}.hdf5')
                h_i.to_csv(fcsv)
                with h5py.File(fhdf, 'w') as f:
                    f.create_dataset('images', data=self.val_images)
                    f.create_dataset('masks', data=self.val_masks)

            print(f"ARI: {ari}\tNMI: {nmi}")
    
    def remove_most_recent_file(self, path):
        file_delete = [
                entry.name for entry in sorted(
                os.scandir(path),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
        ][0]
        file_delete = os.path.join(path, file_delete)
        os.remove(file_delete)

    def validate(self, temp):
        self.model.eval()
        self.model.encoder.eval()
        val_loss = 0
        h_i_list = []
        h_score = [str(a) for a in range(self.model.h_dim)]

        for batch_idx, (data, label) in enumerate(self.val_loader):
            data  = data.to(self.main_config['device'])
            z_i, z_j, _, _ = self.model(data)
            loss = self.model.loss(z_i, z_j, temp)
            val_loss += loss.item()
            h_i = self.model.encoder(data)
            h_i = h_i.detach().cpu().numpy()
            label = label.cpu().numpy()

            for i in range(h_i.shape[0]):
                h_data = {h_score[j]: h_i[i][j] for j in range(self.model.h_dim)}
                h_data["label"] = str(int(label[i]))
                h_i_list.append(h_data)

        val_loss /= len(self.val_loader)
        print(f"Validation Loss: {val_loss}")
        wandb.log({"val_loss": val_loss})
        return val_loss, h_i_list
    
    def save_checkpoint(self, model, epoch, dir):
        if epoch > 0:
            self.remove_most_recent_file(dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        fname = f"ep_{epoch}-loss_{self.best_pred:.4f}-ari_{self.best_ari:.4f}.pth"
        fname = os.path.join(dir, fname)
        torch.save(checkpoint, fname)

if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_DIR / "csv", exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR / "hdf", exist_ok=True)

    trainer = Trainer(tune_config, aug_params)
    if tune_config["tune"]:
        wandb.login(key=tune_config["wandb_key"])
        sweep_id = wandb.sweep(sweep_config, project=get_repo_name())
        wandb.agent(sweep_id, trainer.run_training, count=tune_config['count'])
    else:
        trainer.run_training()
    