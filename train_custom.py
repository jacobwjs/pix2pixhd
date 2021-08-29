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


def scale_width(img, target_width, method):
    ''' Scales an image to target_width while retaining aspect ratio '''
    w, h = img.size
    if w == target_width: return img
    target_height = target_width * h // w
    return img.resize((target_width, target_height), method)



# Scaling factors for losses 
# 
lambda0 = 1.0
lambda1 = 10. 
lambda2 = 10.
# Keep ratio of composite loss, but scale down max to 1.0
norm_weight_to_one=True
scale = max(lambda0, lambda1, lambda2) if norm_weight_to_one else 1.0
lambda0 = lambda0 / scale
lambda1 = lambda1 / scale
lambda2 = lambda2 / scale


def vgg_loss(vgg, x_real, x_fake):
    ''' Computes perceptual loss with VGG network from real and fake images '''    
    vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    
    vgg_real = vgg(x_real)
    vgg_fake = vgg(x_fake)

    vgg_loss = 0.0
    for real, fake, weight in zip(vgg_real, vgg_fake, vgg_weights):
        vgg_loss += weight * F.l1_loss(real.detach(), fake)

    return vgg_loss


def fm_loss(real_preds, fake_preds):
    ''' Computes feature matching loss from nested lists of fake and real outputs from discriminator '''
    fm_loss = 0.0
    for real_features, fake_features in zip(real_preds, fake_preds):
        for real_feature, fake_feature in zip(real_features, fake_features):
            fm_loss += F.l1_loss(real_feature.detach(), fake_feature)
    
    return fm_loss


def adv_loss(discriminator_preds, is_real):
    ''' Computes adversarial loss from nested list of fakes outputs from discriminator '''
    target = torch.ones_like if is_real else torch.zeros_like

    adv_loss = 0.0
    for preds in discriminator_preds:
        pred = preds[-1]
        adv_loss += F.mse_loss(pred, target(pred))
    
    return adv_loss


def enc_loss(f_map, img_orig):
    return F.l1_loss(f_map, img_orig)


def lpips_loss(lpips, x_fake, x_real):
    return lpips(x_fake.detach(), x_real)


def forward_loss(img_A, img_B, img_AtoB, encoder, generator, discriminator, vgg, lpips):
    # Forward call of loss.
    #
    x_real = img_AtoB

#     feature_map = encoder(img_B)
#     x_fake = generator(torch.cat((img_A, feature_map), dim=1))
    feature_map = encoder(torch.cat((img_A, img_B), dim=1))
    x_fake = generator(torch.cat((img_A, img_B, feature_map), dim=1))
#     print(feature_map.shape)
#     print(x_fake.shape)

    # Get necessary outputs for loss/backprop for both generator and discriminator
#     fake_preds_for_g = discriminator(x_fake)
#     fake_preds_for_d = discriminator(x_fake.detach())
#     real_preds_for_d = discriminator(x_real.detach())
    fake_preds_for_g = discriminator(torch.cat((img_A, img_B, x_fake), dim=1))
    fake_preds_for_d = discriminator(torch.cat((img_A, img_B, x_fake.detach()), dim=1))
    real_preds_for_d = discriminator(torch.cat((img_A, img_B, x_real.detach()), dim=1))

    g_loss = (
        lambda0 * adv_loss(fake_preds_for_g, False) + \
        lambda1 * fm_loss(real_preds_for_d, fake_preds_for_g) / discriminator.n_discriminators + \
        lambda2 * vgg_loss(vgg, x_fake, x_real)  + \
#         0.5 * enc_loss(feature_map, img_B) + \
        2.0 * lpips_loss(lpips, x_fake, x_real)
    )

    d_loss = 0.5 * (
        adv_loss(real_preds_for_d, True) + \
        adv_loss(fake_preds_for_d, False)
    )

    return g_loss, d_loss, x_fake.detach(), feature_map.detach()



def train(
    args, config, log_dir,
    train_loader, val_loader,
    encoder, generator, discriminator, vgg, lpips):
    
    if args.high_res:
        g_optimizer = torch.optim.Adam(
            list(generator.parameters()), **config.optim,
        )
    else:
        g_optimizer = torch.optim.Adam(
            list(generator.parameters()) + list(encoder.parameters()), **config.optim,
        )
    d_optimizer = torch.optim.Adam(list(discriminator.parameters()), **config.optim)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(
        g_optimizer,
        get_lr_lambda(config.train.epochs, config.train.decay_after),
    )
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(
        d_optimizer,
        get_lr_lambda(config.train.epochs, config.train.decay_after),
    )
    
    start_epoch = 0
    if config.resume_checkpoint is not None:
        state_dict = torch.load(config.resume_checkpoint)

        encoder.load_state_dict(state_dict['e_model_dict'])
        generator.load_state_dict(state_dict['g_model_dict'])
        discriminator.load_state_dict(state_dict['d_model_dict'])
        g_optimizer.load_state_dict(state_dict['g_optim_dict'])
        d_optimizer.load_state_dict(state_dict['d_optim_dict'])
        start_epoch = state_dict['epoch']

        msg = 'high-res' if args.high_res else 'low-res'
        print(f'Starting {msg} training from checkpoints')
        
    elif args.high_res:
        state_dict = config.pretrain_checkpoint
        if state_dict is not None:
            encoder.load_state_dict(torch.load(state_dict['e_model_dict']))
            encoder = freeze_encoder(encoder)
            generator.g1.load_state_dict(torch.load(state_dict['g_model_dict']))
            print('Starting high-res training from pretrained low-res checkpoints')
        else:
            print('Starting high-res training from scratch (no valid checkpoint detected)')

    else:
        print('Starting low-res training from random initialization')


    num_seen_examples = 0
    for epoch in tqdm(range(start_epoch, config.train.epochs)):
        # training epoch
        # ---------------------------------------------
        #
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        epoch_steps = 0
        if not args.high_res:
            encoder.train()

        generator.train()
        discriminator.train()
        
        pbar = tqdm(train_loader, position=0, desc='train [G loss: -.----][D loss: -.----]')
        for batch in pbar:
            img_A, img_B, img_AtoB, img_BtoA = batch
            img_A = img_A.to(device)
            img_B = img_B.to(device)
            img_AtoB = img_AtoB.to(device)
            img_BtoA = img_BtoA.to(device)
    
                
            g_loss, d_loss, x_fake, encoder_feats = forward_loss(
                img_A,
                img_B,
                img_AtoB,
                encoder,
                generator,
                discriminator,
                vgg,
                lpips
            )
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            
            mean_g_loss += g_loss.item() 
            mean_d_loss += d_loss.item() 
            
            # Normalize by number of steps since we are accumulating.
            # NOTE: Only used for updating progress display.
            #
            epoch_steps += 1
#             mean_g_loss /= epoch_steps
#             mean_d_loss /= epoch_steps
            
            pbar.set_description(
                desc=f'train [G loss: {mean_g_loss/epoch_steps:.4f}][D loss: {mean_d_loss/epoch_steps:.4f}]'
            )   
            
            num_seen_examples += img_A.shape[0]
            if (num_seen_examples % config.train.snapshot_every_n_examples) == 0:
                # Save snapshot of results up to now in training.
                #
                res = torch.cat([img_A, img_B, img_AtoB, x_fake, encoder_feats], dim=0)
                torchvision.utils.save_image(
                    res,
                    fp=f'{log_dir}/train_snapshot_seen{num_seen_examples}_epoch{epoch}.png',
                    normalize=True
                )
            
        g_scheduler.step()
        d_scheduler.step()
        
      
        # validation epoch
        # ---------------------------------------------
        #
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        epoch_steps = 0
        if not args.high_res:
            encoder.eval()
            
        generator.eval()
        discriminator.eval()
        
        pbar = tqdm(val_loader, position=0, desc='val [G loss: -.----][D loss: -.----]')
        for (img_A, img_B, img_AtoB, img_BtoA) in pbar:
            img_A = img_A.to(device)
            img_B = img_B.to(device)
            img_AtoB = img_AtoB.to(device)
            img_BtoA = img_BtoA.to(device)

            with torch.no_grad():
#                 with torch.cuda.amp.autocast(enabled=(device=='cuda')):
#                     g_loss, d_loss, x_fake = loss(
#                         x_real, labels, insts, bounds, encoder, generator, discriminator,
#                     )
                g_loss, d_loss, x_fake, encoder_feats = forward_loss(
                    img_A,
                    img_B,
                    img_AtoB,
                    encoder,
                    generator,
                    discriminator,
                    vgg,
                    lpips
                )
    
#             if epoch_steps == 0:
#                 # Save images on first pass.
#                 res = torch.cat([img_A, img_B, img_AtoB, x_fake], dim=0)
#                 torchvision.utils.save_image(res, fp=f'{log_dir}/snapshot_epoch{epoch}.png', normalize=True)
            
            mean_g_loss += g_loss.mean().item()
            mean_d_loss += d_loss.mean().item()
            
            # Normalize by number of steps since we are accumulating.
            # NOTE: Only used for updating progress display.
            #
            epoch_steps += 1
#             mean_g_loss /= epoch_steps
#             mean_d_loss /= epoch_steps
            pbar.set_description(
                desc=f'val [G loss: {mean_g_loss/epoch_steps:.4f}][D loss: {mean_d_loss/epoch_steps:.4f}]'
            )
            
            
        if (epoch % config.train.save_ckpt_every) == 0:
            print("Checkpointing model...")
            torch.save({
                'e_model_dict': encoder.state_dict(),
                'g_model_dict': generator.state_dict(),
                'd_model_dict': discriminator.state_dict(),
                'g_optim_dict': g_optimizer.state_dict(),
                'd_optim_dict': d_optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(log_dir, f'ckpt_epoch{epoch}.pt'))
            
            # Save snapshot of results on validation data every time we checkpoint
            #
            res = torch.cat([img_A, img_B, img_AtoB, x_fake, encoder_feats], dim=0)
            torchvision.utils.save_image(
                res,
                fp=f'{log_dir}/ckpt_snapshot_seen{num_seen_examples}_epoch{epoch}.png',
                normalize=True
            )


            

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='Path to config file')
    parser.add_argument('--path_data', type=str, required=True,
                        help='Path to training and validation data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers for dataloader (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device we are using for training (default: CUDA)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size (default: 1)')
    parser.add_argument('--high_res', action='store_true', default=False)
    return parser.parse_args()

    
if __name__ == "__main__":
    
    args = parse_arguments()
    
    parent_path = args.path_data # "../dataset/image-to-image-50k-256px"
    train_dataset = StyleGANFaces(
        path_A= f"{parent_path}/train/imagesA",
        path_B= f"{parent_path}/train/imagesB",
        path_AtoB = f"{parent_path}/train/images_AtoB",
        path_BtoA = f"{parent_path}/train/images_BtoA"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
#         sampler=None,
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
        shuffle=True,
#         sampler=None,
        num_workers=args.num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=True,
    )

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)
        config = parse_config(config)

    print(config)
    
    log_dir = os.path.join(config.train.log_dir, datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    os.makedirs(log_dir, mode=0o775, exist_ok=False)

    device = args.device
    encoder = instantiate(config.encoder).to(device).apply(weights_init)
    generator = instantiate(config.generator).to(device).apply(weights_init)
    discriminator = instantiate(config.discriminator).to(device).apply(weights_init)
    vgg = VGG19().to(device)
    lpips = LPIPS.LPIPS(net='alex').to(device)

    # summary(encoder, (3, 256, 256))
    # summary(generator, (6, 256, 256))
    # summary(discriminator, (3, 256, 256))
    
    train(
        args,
        config,
        log_dir,
        train_dataloader,
        val_dataloader,
        encoder,
        generator,
        discriminator,
        vgg,
        lpips
    )






