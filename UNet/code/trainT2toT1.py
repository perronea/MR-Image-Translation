#! /usr/bin/env python

import os
from pathlib import Path
from tqdm import tqdm
import random
import argparse
import yaml
import glob

import numpy as np
#import matplotlib.pyplot as plt
#from scipy import stats
import nibabel as nib

import torch
from torch import optim
import torch.nn as nn
import torchvision
import torchio as tio

from tio_preprocessing import TIOPreprocessing
from model import UNet

def train(tio_data, device, **params):
    lr = params['learning_rate']
    epochs = params['epochs']
    weights_path = params['weights_path']

    net = UNet()
    # Load previous weights if the path already exists
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, map_location=device))

    net.to(device=device)
    print(net.parameters())

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=params['weight_decay'], momentum=params['momentum'])
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        net.train()

        # Train model        
        epoch_training_loss = 0
        #with tqdm(total=tio_data.num_training_subjects, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in tio_data.training_loader_patches:
            T1 = batch['T1w']['data']
            T2 = batch['T2w']['data']
            T1 = T1.to(device=device, dtype=torch.float32)
            T2 = T2.to(device=device, dtype=torch.float32)
            T1_pred = net(T2)
            loss = criterion(T1_pred, T1)
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
            T1_pred = net(T2)
            loss = criterion(T1_pred, T1)
            epoch_validation_loss += loss.item()
        print(epoch_validation_loss)
        validation_losses += [epoch_validation_loss]
        print("Max memory allocated: {}".format(torch.cuda.max_memory_allocated(device=device)))
            
        # Save model at end of each epoch
        model_name = os.path.join(params['output'], 'T2toT1_model_gpu_epoch-%s.pth' % str(epoch))
        torch.save(net.state_dict(), model_name)

    return


def test(tio_data, model, device):

    net = UNet()
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device=device)
    net.eval

    subject = random.choice(tio_data.validation_set)
    input_tensor = subject.T2w.data[0]
    patch_size = 140, 140, 140
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
            inputs = patches_batch['T2w'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            pred_T1 = net(inputs)
            print(pred_T1.shape)
    
    #one_batch = next(iter(tio_data.validation_loader_patches))
    patch_size=140
    k = int(patch_size // 2)
    batch_T1 = patches_batch['T1w']['data'][...,k]
    batch_T2 = patches_batch['T2w']['data'][...,k]
    batch_pred_T1 = pred_T1[..., k].to(torch.device('cpu'))
    slices = torch.cat((batch_T2, batch_T1, batch_pred_T1))
    image_path = 'batch_T2toT1_patches.png'
    torchvision.utils.save_image(
        slices,
        image_path,
        nrow=tio_data.training_batch_size,
        normalize=True,
        scale_each=True
    )
    #display.Image(image_path)
    return

def generate(tio_data, model, device, params):
    net = UNet()
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device=device)

    for subject in tio_data:

        patch_size = 140, 140, 140
        patch_overlap = 16, 16, 16
        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size,
            patch_overlap,
        )
        patch_loader = torch.utils.data.DataLoader(
            grid_sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

        net.eval()
        with torch.no_grad():
            for patches_batch in patch_loader:
                inputs = patches_batch['T2w'][tio.DATA].to(device)
                locations = patches_batch[tio.LOCATION]
                pred_T2 = net(inputs)
                aggregator.add_batch(pred_T2, locations)
            
        prediction = aggregator.get_output_tensor()

        T1_pred_subject = tio.Subject(T1w=tio.ScalarImage(tensor=prediction))
        T1_landmarks_dict = {'T1w': np.load(params['T1_landmarks_path'])}
        T1_hist_transform = tio.HistogramStandardization(T1_landmarks_dict)
        transform = tio.Compose([T1_hist_transform])
        T1_std = transform(T1_pred_subject)
        prediction_data = T1_std.T1w.data[0,:,:,:].numpy()
        prediction_data = prediction_data.clip(0)

    return(prediction_data)

def get_cli_args():
    
    parser = argparse.ArgumentParser(
        description="UNet for estimating T1w and T2w images"
    )
    parser.add_argument(
        "--config",
        type=str,
        help=("Path to a yaml file containing all of the required parameters")
    )

    
    return parser

def read_params(config_file):

    with open(config_file, 'rb') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    params = {
        'training_data': None,
        'T1_landmarks_path': None,
        'T2_landmarks_path': None,
        'test_data': None,
        'weights_path': None,
        'train': True,
        'generate': False,
        'learning_rate': 0.01,
        'epochs': 50,
        'weight_decay': 0.0000001,
        'momentum': 0.9,
        'patch_size': 140,
        'patch_overlap': 4
    }

    for key, value in conf.items():
        params[key] = value

    print(params)

    return(params)

def main():

    parser = get_cli_args()
    args = parser.parse_args()

    params = read_params(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if params['train']:
        print('Training')
        tio_data = TIOPreprocessing(data_dir=params['training_data'], T1_landmarks_path=params['T1_landmarks_path'], T2_landmarks_path=params['T2_landmarks_path'])    
        tio_data.load_data()
        tio_data.find_histogram_landmarks(train=False)
        tio_data.compose_preprocessing()
        tio_data.split_training_data()
        tio_data.generate_patches()
        train(tio_data, device, **params)
    elif params['generate']:
        print('Generating')

        # First create transform
        T2_landmarks_path=params['T2_landmarks_path']
        if T2_landmarks_path:
            landmarks_dict = {'T2w': np.load(T2_landmarks_path)}
            histogram_transform = tio.HistogramStandardization(landmarks_dict)
        composed_transform = tio.Compose([histogram_transform])
        
        # List out image paths and create tio dataset
        T2_img_paths = sorted(glob.glob(os.path.join(params['test_data'], '*.nii.gz')))
        for T2_path in T2_img_paths:
            subject = tio.Subject(T2w=tio.ScalarImage(T2_path))
            tio_data = tio.SubjectsDataset([subject], transform=composed_transform)
            prediction_data = generate(tio_data, params['weights_path'], device, params)
            orig_img = nib.load(T2_path)
            orig_affine = orig_img.affine
            prediction_data = (prediction_data - prediction_data.min()) * (orig_img.get_fdata().max() / prediction_data.max())
            prediction_img = nib.Nifti1Image(prediction_data, orig_affine)
            synth_img_path = T2_path.replace('/pre/', '/synth/').replace('_T2w.nii.gz','_desc-synthetic_T1w.nii.gz')
            nib.save(prediction_img, synth_img_path)
            print('Generated {}'.format(synth_img_path))


        #net = UNet()
        #net.load_state_dict(torch.load(params['model_path'], map_location=device))
        #net.to(device=device)
        #net.eval
        #test(tio_data, device, net)
                

if __name__ == '__main__':
    main()



