#!/usr/bin/env python
# coding: utf-8


# !pip install torchsummary
# !pip install lpips


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from hydra.utils import instantiate

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
from datetime import datetime
import os
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from collections.abc import Iterable
import glob
import numpy as np
from PIL import Image

from modules.dataset import StyleGANFaces
# from modules.loss import Pix2PixHDLoss
from modules.networks import VGG19
from utils import parse_config, get_lr_lambda, weights_init, freeze_encoder, show_tensor_images

import lpips as LPIPS





            
class Pix2PixHD(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        self.config = config
        
        self.encoder = instantiate(config.encoder).to(config.device).apply(weights_init)
        self.generator = instantiate(config.generator).to(config.device).apply(weights_init)
        self.discriminator = instantiate(config.discriminator).to(config.device).apply(weights_init)
        self.vgg = VGG19().to(config.device)
        self.lpips = LPIPS.LPIPS(net='alex').to(config.device)
        
        self.automatic_optimization = False
        
        # Scaling factors for individual losses 
        # 
        self.lambda0 = self.config.losses.lambda_adv
        self.lambda1 = self.config.losses.lambda_fm
        self.lambda2 = self.config.losses.lambda_vgg
        self.lambda3 = self.config.losses.lambda_lpips
        # Keep ratio of composite loss, but scale down max to 1.0
        norm_weight_to_one=True
        scale = max(self.lambda0, self.lambda1, self.lambda2) if norm_weight_to_one else 1.0
        self.lambda0 = self.lambda0 / scale
        self.lambda1 = self.lambda1 / scale
        self.lambda2 = self.lambda2 / scale
        
        self.num_seen_examples = 0 # How many data examples have we seen (i.e. images)
        
        
    def forward(self, img_A, img_B):
        feature_map = self.encoder(torch.cat((img_A, img_B), dim=1))
        x_fake = self.generator(torch.cat((img_A, img_B, feature_map), dim=1))
        return x_fake
    
    
    def configure_optimizers(self):
        if args.high_res:
            g_optimizer = torch.optim.Adam(
                list(self.generator.parameters()),
                **self.config.optim,
            )
        else:
            g_optimizer = torch.optim.Adam(
                list(self.generator.parameters()) + list(self.encoder.parameters()),
                **self.config.optim,
            )
            
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(
            g_optimizer,
            get_lr_lambda(self.config.train.epochs, self.config.train.decay_after),
        )
        
        d_optimizer = torch.optim.Adam(
            list(self.discriminator.parameters()),
            **self.config.optim
        )
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(
            d_optimizer,
            get_lr_lambda(self.config.train.epochs, self.config.train.decay_after),
        )
        
#         return (
#             {'optimizer': g_optimizer, 'lr_scheduler': g_scheduler},
#             {'optimizer': d_optimizer, 'lr_scheduler': d_scheduler}
#         )
        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

    
    def transfer(self, X, Y, XtoY):
        '''
        Perform the pix2pix transfer
        '''
        x_real = XtoY
        feature_map = self.encoder(torch.cat((X, Y), dim=1))
        x_fake = self.generator(torch.cat((X, Y, feature_map), dim=1))

        # Get necessary outputs for loss/backprop for both generator and discriminator
        #
#         fake_preds_for_g = discriminator(x_fake)
#         fake_preds_for_d = discriminator(x_fake.detach())
#         real_preds_for_d = discriminator(x_real.detach())
        fake_preds_for_g = self.discriminator(torch.cat((X, Y, x_fake), dim=1))
        fake_preds_for_d = self.discriminator(torch.cat((X, Y, x_fake.detach()), dim=1))
        real_preds_for_d = self.discriminator(torch.cat((X, Y, x_real.detach()), dim=1))

        g_loss = (
            self.lambda0 * self.adv_loss(fake_preds_for_g, False) + \
            self.lambda1 * self.fm_loss(real_preds_for_d, fake_preds_for_g) / self.discriminator.n_discriminators + \
            self.lambda2 * self.vgg_loss(x_fake, x_real)  + \
    #         0.5 * enc_loss(feature_map, img_B) + \
            self.lambda3 * self.lpips_loss(x_fake, x_real)
        )

        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False)
        )
        
        return g_loss, d_loss, x_fake, feature_map
        
    
    def training_step(self, batch, batch_idx):
#         g_optimizer, d_optimizer  = self.optimizers(use_pl_optimizer=False)
        g_optimizer, d_optimizer  = self.optimizers()
        
        img_A, img_B, img_AtoB, img_BtoA = batch
        img_A = img_A 
        img_B = img_B 
        img_AtoB = img_AtoB 
        img_BtoA = img_BtoA
        

        ### TODO:
        ### - What is best way to train? Update after both transfers have occurred,
        ###   or after each?
        ###
        
        # Compute gradients after each tranfer. Respects batch_size.
        # ------------------------------------------------------------------
        #
        # Transfer A -> B
        #
        g_loss, d_loss, _, _ = self.transfer(X=img_A, Y=img_B, XtoY=img_AtoB)
        g_loss = g_loss.mean()
        d_loss = d_loss.mean()
        self.num_seen_examples += img_A.shape[0]
        g_optimizer.zero_grad()
        self.manual_backward(g_loss)
        g_optimizer.step()
        d_optimizer.zero_grad()
        self.manual_backward(d_loss)
        d_optimizer.step()        
        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True, logger=True)
        
        # Transfer B -> A
        #
        g_loss, d_loss, _, _ = self.transfer(X=img_B, Y=img_A, XtoY=img_BtoA)
        g_loss = g_loss.mean()
        d_loss = d_loss.mean()
        self.num_seen_examples += img_A.shape[0]
        g_optimizer.zero_grad()
        self.manual_backward(g_loss)
        g_optimizer.step()
        d_optimizer.zero_grad()
        self.manual_backward(d_loss)
        d_optimizer.step()        
        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True, logger=True)        
        
        
#         # Compute gradients and step after both A -> B and B -> A have occurred.
#         # ------------------------------------------------------------------
#         #
#         # Transfer A -> B
#         #
#         g_loss, d_loss, _, _ = self.transfer(X=img_A, Y=img_B, XtoY=img_AtoB)
#         self.num_seen_examples += img_A.shape[0]
#         g_total = g_loss.mean()
#         d_total = d_loss.mean()
        
#         # Transfer B -> A
#         #
#         g_loss, d_loss, _, _ = self.transfer(X=img_B, Y=img_A, XtoY=img_BtoA)
#         self.num_seen_examples += img_A.shape[0]
#         g_total = g_loss.mean()
#         d_total = d_loss.mean()
        
#         g_optimizer.zero_grad()
#         self.manual_backward(g_total)
#         g_optimizer.step()
#         d_optimizer.zero_grad()
#         self.manual_backward(d_total)
#         d_optimizer.step()        
    
#         self.log_dict({"g_loss": g_total, "d_loss": d_total}, prog_bar=True, logger=True)
#         self.log("g_loss", g_total, prog_bar=True)
        
        
        # Step scheduler every `n` epochs
        #
        n = 1
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % n == 0:
            for scheduler in self.lr_schedulers():
                scheduler.step()
        
    
    def validation_step(self, batch, batch_idx):
        img_A, img_B, img_AtoB, img_BtoA = batch
        img_A = img_A #.to(device)
        img_B = img_B #.to(device)
        img_AtoB = img_AtoB #.to(device)
        img_BtoA = img_BtoA #.to(device)
        
        with torch.no_grad():
            g_loss_forward, d_loss_forward, x_fake_forward, features_forward = self.transfer(X=img_A, Y=img_B, XtoY=img_AtoB)
            g_loss_back, d_loss_back, x_fake_back, features_back = self.transfer(X=img_B, Y=img_A, XtoY=img_BtoA)
    
    
        val_g_loss = g_loss_forward.mean() + g_loss_back.mean()
        val_d_loss = d_loss_forward.mean() + g_loss_back.mean()
        self.log("val_g_loss", val_g_loss, prog_bar=True, logger=True)
        self.log("val_d_loss", val_d_loss, prog_bar=True, logger=True)
#         self.log("dis_loss", dis_loss, prog_bar=True)
#         self.log_dict({"gen_loss": gen_loss, "dis_loss": dis_loss}, prog_bar=True, logger=True)
        
        # Save snapshot of results on validation data
        #
#         if self.trainer.is_last_batch:
        if (batch_idx == 0) and (self.local_rank == self.config.gpu_ids[0]):
            res = torch.cat(
                [img_A, img_B, img_AtoB, x_fake_forward, features_forward, img_BtoA, x_fake_back, features_back],
                dim=0
            ).detach().cpu()
            torchvision.utils.save_image(
                res,
                fp=f'{self.trainer.logger.log_dir}/snapshot_seen{self.num_seen_examples}_epoch{self.current_epoch}.png',
                normalize=True,
                scale_each=True
            )
    
    # Available losses for training.
    # -----------------------------------------------------
    #
    def vgg_loss(self, x_real, x_fake):
        ''' Computes perceptual loss with VGG network from real and fake images '''    
        vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        vgg_real = self.vgg(x_real)
        vgg_fake = self.vgg(x_fake)

        vgg_loss = 0.0
        for real, fake, weight in zip(vgg_real, vgg_fake, vgg_weights):
            vgg_loss += weight * F.l1_loss(real.detach(), fake)

        return vgg_loss


    def fm_loss(self, real_preds, fake_preds):
        ''' Computes feature matching loss from nested lists of fake and real outputs from discriminator '''
        fm_loss = 0.0
        for real_features, fake_features in zip(real_preds, fake_preds):
            for real_feature, fake_feature in zip(real_features, fake_features):
                fm_loss += F.l1_loss(real_feature.detach(), fake_feature)

        return fm_loss


    def adv_loss(self, discriminator_preds, is_real):
        ''' Computes adversarial loss from nested list of fakes outputs from discriminator '''
        target = torch.ones_like if is_real else torch.zeros_like

        adv_loss = 0.0
        for preds in discriminator_preds:
            pred = preds[-1]
            adv_loss += F.mse_loss(pred, target(pred))

        return adv_loss


    def enc_loss(self, f_map, img_orig):
        return F.l1_loss(f_map, img_orig)


    def lpips_loss(self, x_fake, x_real):
        return self.lpips(x_fake.detach(), x_real)
        
            

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--path_data', type=str, required=True,
                        help='Path to training and validation data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers for dataloader (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device we are using for training (default: CUDA)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size (default: 1)')
    parser.add_argument('--tag', type=str, required=True,
                        help='Tag to append to training/logging dir for identification (e.g. mixlevel-middle)')
    parser.add_argument('--gpu_ids', type=lambda s: [int(item) for item in s.split(',')], required=True,
                        help="GPU ids for training (e.g. --gpu_ids 0,1,2)")
    parser.add_argument('--precision', type=int, default=32,
                        help="Precision used in training (default: 32)")
    parser.add_argument('--experiment', type=str, default=None,
                        help='(Optional) define experiment name to log to.')
    parser.add_argument('--high_res', action='store_true', default=False)
    return parser.parse_args()

    
    
if __name__ == "__main__":
    args = parse_arguments()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)
        config = parse_config(config)

    config.device = args.device
    config.gpu_ids = args.gpu_ids
    config.high_res = args.high_res
    config.args = OmegaConf.to_yaml(OmegaConf.from_cli())
    config.experiment = args.experiment
    print(config)

    
    # Setup logging.
    #
#     log_dir = os.path.join(config.train.log_dir, datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    log_dir = f"{config.train.log_dir}/{args.tag}"
#     os.makedirs(log_dir, mode=0o775, exist_ok=True)
    config.log_dir = log_dir
    logger = TensorBoardLogger(
        f"{log_dir}/tb_logs",
        name=args.experiment
    )
    

    parent_path = args.path_data # "../dataset/image-to-image-50k-256px"
    print(">> training data:", parent_path)
    print(">> logging to:", log_dir)
    print(">> precision:", args.precision)
    print(">> batch size:", args.batch_size)
    print(">> gpu_ids:", args.gpu_ids)
    
    
    # Setup datasets for training and validation.
    #
    train_dataset = StyleGANFaces(
        path_A= f"{parent_path}/train/imagesA",
        path_B= f"{parent_path}/train/imagesB",
        path_AtoB = f"{parent_path}/train/images_AtoB",
        path_BtoA = f"{parent_path}/train/images_BtoA"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=True,
    )
    
    val_dataset = StyleGANFaces(
        path_A= f"{parent_path}/validation/imagesA",
        path_B= f"{parent_path}/validation/imagesB",
        path_AtoB = f"{parent_path}/validation/images_AtoB",
        path_BtoA = f"{parent_path}/validation/images_BtoA"
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1 if args.high_res else 8,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=True,
    )

    # summary(encoder, (3, 256, 256))
    # summary(generator, (6, 256, 256))
    # summary(discriminator, (3, 256, 256))
    
    # Setup checkpoints.
    #
    checkpoint_callback = ModelCheckpoint(
#         monitor='val_loss',
#         dirpath=log_dir,
#         save_top_k=3,
        filename="pix2pixhd-{epoch:03d}" + f"-{args.tag}",
        every_n_epochs=1, #config.train.ckpt_every_n
    )
    
    print("checkpoint", checkpoint_callback.dirpath)
    
    # Define the trainer.
    #
    trainer = Trainer(
#         default_root_dir=log_dir,
        gpus=args.gpu_ids, # specify which GPUs to use explicitly.
        auto_select_gpus=True,
        accelerator="ddp",
        plugins=DDPPlugin(find_unused_parameters=False),
        precision=args.precision,
        weights_summary="top",
        logger=logger,
        max_epochs=config.train.epochs,
        callbacks=[checkpoint_callback],
        val_check_interval=0.25, # Run validation every 0.25 of an epoch
    ) 

#     # For testing.
#     #
#     trainer = Trainer(
#         gpus=args.gpu_ids, # specify which GPUs to use explicitly.
#         auto_select_gpus=True,
#         accelerator="ddp",
#         plugins=DDPPlugin(find_unused_parameters=False),
#         precision=args.precision,
#         weights_summary="top",
#         logger=logger,
#         max_epochs=config.train.epochs,
#         callbacks=[checkpoint_callback],
#         limit_train_batches=0.2,
#         limit_val_batches=0.2,        
#     #         fast_dev_run=True
#     ) 
    
#     print("Trainer.logger", trainer.logger)
#     print("Trainer.logger.save_dir", trainer.logger.save_dir)
#     print("Trainer.logger.version", trainer.logger.version)
#     print("Trainer.logger.sub_dir", trainer.logger.sub_dir)
#     print("Trainer.logger.log_dir", trainer.logger.log_dir)
#     print("Trainer.logger.root_dir", trainer.logger.root_dir)
#     print("Trainer.default_root_dir", trainer.default_root_dir)

      
    # Setup the model.    
    #
    model = Pix2PixHD(config)
    
    # Train model on dataset.
    #
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader
    )
    
    # Explicitly save the training configuration file.
    #
    fp = f'{trainer.logger.log_dir}/training_config.yaml'
    OmegaConf.save(config=config, f=fp)

    






