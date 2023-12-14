# must import pytorch_lightning before numpy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import argparse
import os
import os.path as osp
from pprint import pprint

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from dataloader import MPIIGazeDataset

from quantize import KMeansQuantizer

os.environ["WANDB__SERVICE_WAIT"] = "60"

def euler_to_vec(theta, phi):
    x = -1 * np.cos(theta) * np.sin(phi)
    y = -1 * np.sin(theta)
    z = -1 * np.cos(theta) * np.cos(phi)
    vec = np.array([x, y, z])
    vec = vec / np.linalg.norm(vec)
    return vec
    
def extra_preprocess(x):
    return (x * 255 - 128).clamp(-128, 127).to(torch.int8)

class TrainModel(pl.LightningModule):
    def __init__(self, args, model=None, kmeans_quantizer=None):
        super(TrainModel, self).__init__()
        if model is not None:
            self.model = model
        elif args.model == "MyModelv7":
            self.model = models.MyModelv7(arch=args.arch)
        elif args.model == 'MyModelv8':
            self.model = models.MyModelv8(arch=args.arch)
        else:
            raise NotImplementedError
        self.criterion = getattr(torch.nn, args.criterion)()
        self.args = args
        self.kmeans_quantizer = kmeans_quantizer
    
    def calc_angle_error(self, preds, gts):
        # in degree
        preds = np.deg2rad(preds.detach().cpu())
        gts = np.deg2rad(gts.detach().cpu())
        errors = []
        for pred, gt in zip(preds, gts):
            pred_vec = euler_to_vec(pred[0], pred[1])
            gt_vec = euler_to_vec(gt[0], gt[1])
            error = np.rad2deg(np.arccos(np.clip(np.dot(pred_vec, gt_vec), -1.0, 1.0)))
            errors.append(error)
        return errors
    
    def on_train_epoch_end(self, *args, **kwargs):
        if self.kmeans_quantizer is not None: 
            self.kmeans_quantizer.apply(self.model, update_centroids=True)

    def forward(self, left_eye, right_eye, face):
        return self.model(left_eye, right_eye, face)

    def training_step(self, batch, batch_idx):
        left, right, face, label = batch

        left = left.expand(left.shape[0], 3, left.shape[2], left.shape[3]).cuda()
        right = right.expand(right.shape[0], 3, right.shape[2], right.shape[3]).cuda()
        face = face.expand(face.shape[0], 3, face.shape[2], face.shape[3]).cuda()

        # if self.kmeans_quantizer is None:
        #     left = extra_preprocess(left)
        #     right = extra_preprocess(right)
        #     face = extra_preprocess(face)

        output = self.model(left, right, face)
        loss = self.criterion(output, label)
        if batch_idx % 100 == 0:
            angle_error = np.mean(self.calc_angle_error(output, label))
            self.log("train_angle_error", angle_error, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        left, right, face, label = batch
        
        left = left.expand(left.shape[0], 3, left.shape[2], left.shape[3]).cuda()
        right = right.expand(right.shape[0], 3, right.shape[2], right.shape[3]).cuda()
        face = face.expand(face.shape[0], 3, face.shape[2], face.shape[3]).cuda()

        # if self.kmeans_quantizer is None:
        #     left = extra_preprocess(left)
        #     right = extra_preprocess(right)
        #     face = extra_preprocess(face)

        output = self.model(left, right, face)
        loss = self.criterion(output, label)
        angle_error = np.mean(self.calc_angle_error(output, label))
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_angle_error", angle_error, on_epoch=True, logger=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.args.optimizer)(self.model.parameters(), **self.args.optimizer_parameters)
        return optimizer
    
    def train_dataloader(self):
        trainset = MPIIGazeDataset(self.args.dataset_dir, is_train=True)
        trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        return trainloader

    def val_dataloader(self):
        valset = MPIIGazeDataset(self.args.dataset_dir, is_train=False)
        valloader = DataLoader(valset, batch_size=4*self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        return valloader
    
    def test_dataloader(self):
        testset = MPIIGazeDataset(self.args.dataset_dir, is_train=False)
        testloader = DataLoader(testset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        return testloader


def train_kmeans(model, args):
    km = KMeansQuantizer(model, bitwidth=8) 
    km.apply(model, update_centroids=False)

    trainer_model = TrainModel(args, model=model, kmeans_quantizer=km)

    if args.logger == 'wandb':
        mylogger = WandbLogger(project=args.project, 
                            log_model=False, 
                            entity="steveli",
                            )
        mylogger.log_hyperparams(args)
        mylogger.watch(trainer_model, None, 10000, log_graph=False)
    else:
        raise NotImplementedError

    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir,
                                        filename='{epoch}-{val_loss:.4f}-{val_angle_error:.2f}',
                                        monitor='val_angle_error',
                                        save_last=True,
                                        save_top_k=3,
                                        verbose=False)

    trainer = Trainer(default_root_dir=args.ckpt_dir,
                    devices=4,
                    precision=32,
                    callbacks=[checkpoint_callback],
                    max_epochs=args.epoch,
                    benchmark=True,
                    logger=mylogger,
                    strategy=DDPStrategy(find_unused_parameters=True)
                    )

    if args.resume:
        trainer.fit(trainer_model, ckpt_path=osp.join(args.ckpt_dir, "last.ckpt"))
    else:
        trainer.fit(trainer_model)
    trainer.validate(trainer_model)
 
def linear_trainer(model, args):
    trainer_model = TrainModel(args, model=model)

    if args.logger == 'wandb':
        mylogger = WandbLogger(project="linear_quantized_gaze", 
                            log_model=False, 
                            entity="steveli",
                            )
        mylogger.log_hyperparams(args)
        mylogger.watch(trainer_model, None, 10000, log_graph=False)
    else:
        raise NotImplementedError

    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir,
                                        filename='{epoch}-{val_loss:.4f}-{val_angle_error:.2f}',
                                        monitor='val_angle_error',
                                        save_last=True,
                                        save_top_k=3,
                                        verbose=False)

    trainer = Trainer(default_root_dir=args.ckpt_dir,
                    devices=4,
                    precision=32,
                    callbacks=[checkpoint_callback],
                    max_epochs=args.epoch,
                    benchmark=True,
                    logger=mylogger,
                    strategy=DDPStrategy(find_unused_parameters=True)
                    )

    if args.resume:
        trainer.fit(trainer_model, ckpt_path=osp.join(args.ckpt_dir, "last.ckpt"))
    else:
        trainer.fit(trainer_model)
    trainer.validate(trainer_model)


if __name__ == "__main__":
    pl.seed_everything(47)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, type=str, default="./configs/config.yaml")
    parser.add_argument('--resume', action="store_true", dest="resume")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_args = yaml.load(f, Loader=yaml.FullLoader)
    yaml_args.update(vars(args))
    args = argparse.Namespace(**yaml_args)
    model = models.MyModelv7(arch=args.arch).cuda()
    train_kmeans(model=model, args=args)

    # linear_trainer(model=model, args=args)

    