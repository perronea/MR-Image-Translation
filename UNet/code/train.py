#! /usr/bin/env python

import os
from pathlib import Path
from tqdm import tqdm
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
from torch import optim
import torch.nn as nn
import torchvision
import torchio as tio

from tio_preprocessing import TIOPreprocessing
from model import UNet

def train(tio_data, device):
    lr = 0.01
    epochs = 200

    net = UNet()
    #net.load_state_dict(torch.load('T1toT2_model_gpu_epoch-67.pth', map_location=device))

    net.to(device=device)
    print(net.parameters())

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        #epoch = epoch + 67
        print(epoch)
        net.train()

        # Train model        
        epoch_training_loss = 0
        #with tqdm(total=tio_data.num_training_subjects, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in tio_data.training_loader_patches:
            T1 = batch['T1w']['data']
            T2 = batch['T2w']['data']
            T1 = T1.to(device=device, dtype=torch.float32)
            T2 = T2.to(device=device, dtype=torch.float32)
            T2_pred = net(T1)
            loss = criterion(T2_pred, T2)
            epoch_training_loss += loss.item()
            
            #pbar.set_postfix(**{'loss (batch)': loss.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #pbar.update(T1.shape[0])
        print(epoch_training_loss)
        training_losses += [epoch_training_loss]

        # Validate model
        epoch_validation_loss = 0
        for batch in tio_data.validation_loader_patches:
            T1 = batch['T1w']['data']
            T2 = batch['T2w']['data']
            T1 = T1.to(device=device, dtype=torch.float32)
            T2 = T2.to(device=device, dtype=torch.float32)
            T2_pred = net(T1)
            loss = criterion(T2_pred, T2)
            epoch_validation_loss += loss.item()
        print(epoch_validation_loss)
        validation_losses += [epoch_validation_loss]
        print("Max memory allocated: {}".format(torch.cuda.max_memory_allocated(device=device)))
            
        # Save model at end of each epoch
        model_name = 'T1toT2_model_gpu_epoch-%s.pth' % str(epoch)
        torch.save(net.state_dict(), model_name)

    return


def test(tio_data, device, net):


    subject = random.choice(tio_data.validation_set)
    input_tensor = subject.T1w.data[0]
    patch_size = 180, 180, 180
    patch_overlap = 4, 4, 4
    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(
        grid_sampler, batch_size=tio_data.validation_batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler)

    net.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['T1w'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            pred_T2 = net(inputs)
            print(pred_T2.shape)
    
    #one_batch = next(iter(tio_data.validation_loader_patches))
    patch_size=100
    k = int(patch_size // 2)
    batch_T1 = patches_batch['T1w']['data'][...,k]
    batch_T2 = patches_batch['T2w']['data'][...,k]
    batch_pred_T2 = pred_T2[..., k].to(torch.device('cpu'))
    slices = torch.cat((batch_T1, batch_T2, batch_pred_T2))
    image_path = 'batch_patches.png'
    torchvision.utils.save_image(
        slices,
        image_path,
        nrow=tio_data.training_batch_size,
        normalize=True,
        scale_each=True
    )
    #display.Image(image_path)
    return


def main():

    tio_data = TIOPreprocessing(data_dir='/home/faird/shared/projects/3D_MRI_GAN/data/BCP_head', T1_landmarks_path='/home/faird/shared/projects/3D_MRI_GAN/T1w_landmarks.npy', T2_landmarks_path='/home/faird/shared/projects/3D_MRI_GAN/T2w_landmarks.npy')
    #tio_data = TIOPreprocessing(data_dir='/home/faird/shared/projects/3D_MRI_GAN/tio_unet/training_data/ECHO_pre')
    tio_data.load_data()
    #tio_data.find_histogram_landmarks(train=False)
    tio_data.compose_preprocessing()
    tio_data.split_training_data()
    tio_data.generate_patches()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_new_model = True

    if train_new_model == True:
        train(tio_data, device)
    else:
        net = UNet()
        net.load_state_dict(torch.load('T1toT2_model_gpu_epoch-125.pth', map_location=device))
        net.to(device=device)
        net.eval
        test(tio_data, device, net)
                

if __name__ == '__main__':
    main()



