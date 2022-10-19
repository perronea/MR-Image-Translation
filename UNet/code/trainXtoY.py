#! /usr/bin/env python

import os
from pathlib import Path
from tqdm import tqdm
import random
import argparse
import yaml
import glob

import numpy as np
import nibabel as nib

import torch
from torch import optim
import torch.nn as nn
import torchvision
import torchio as tio
from torch.utils.data import DataLoader

import GPUtil

from tio_preprocessing import TIOPreprocessing
from model import UNet

def train(tio_data, device, params):
    lr = params['learning_rate']
    epochs = params['epochs']
    weights_path = params['weights_path']

    net = UNet()

    # Load previous weights if the path already exists
    if os.path.exists(weights_path):
        net.load_state_dict(torch.load(weights_path, map_location=device))

    net.to(device=device)
    print(net.parameters)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=params['weight_decay'], momentum=params['momentum'])
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []

    training_data_loader = DataLoader(tio_data.training_set, batch_size=1)

    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch))
        net.train()

        # Train model        
        epoch_training_loss = 0
        #with tqdm(total=tio_data.num_training_subjects, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        #for batch in tio_data.training_loader_patches:
        for batch in training_data_loader:
            # Load batch into GPU
            T1 = batch['T1w']['data']
            T2 = batch['T2w']['data']
            T1 = T1.to(device=device, dtype=torch.float32)
            T2 = T2.to(device=device, dtype=torch.float32)
            
            # Make prediction
            T2_pred = net(T1)
            #print('GPU Utilization after making prediction')
            #GPUtil.showUtilization()
            # Calculate MSE between prediction and target
            loss = criterion(T2_pred, T2)
            print('Batch Loss: {}'.format(loss.item()))
            epoch_training_loss += loss.item()
            
            #pbar.set_postfix(**{'loss (batch)': loss.item()})
            
            # Reset gradient to zero before updating the network so the same gradient isn't applied twice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('GPU Utilization after updating epoch')
            #GPUtil.showUtilization()

            # delete variables on GPU to prevent OOM
            del T1
            del T2
            del T2_pred
            
            #pbar.update(T1.shape[0])
        print('Mean Epoch Loss: {}'.format(epoch_training_loss/40))
        training_losses += [epoch_training_loss/40]

        # Validate model
        #epoch_validation_loss = 0
        #for batch in tio_data.validation_loader_patches:
        #    T1 = batch['T1w']['data']
        #    T2 = batch['T2w']['data']
        #    T1 = T1.to(device=device, dtype=torch.float32)
        #    T2 = T2.to(device=device, dtype=torch.float32)
        #    T2_pred = net(T1)
        #    loss = criterion(T2_pred, T2)
        #    epoch_validation_loss += loss.item()
        #print(epoch_validation_loss)
        #validation_losses += [epoch_validation_loss]
        #print("Max memory allocated: {}".format(torch.cuda.max_memory_allocated(device=device)))
            
        # Save model at end of each epoch
        model_name = os.path.join(params['output'], 'T1toT2_model_gpu_epoch-%s.pth' % str(epoch))
        torch.save(net.state_dict(), model_name)

    return


def test(tio_data, model, device):
    
    net = UNet()
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device=device)
    net.eval

    subject = random.choice(tio_data.validation_set)
    input_tensor = subject.T1w.data[0]
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
            inputs = patches_batch['T1w'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            pred_T2 = net(inputs)
            print(pred_T2.shape)
    
    #one_batch = next(iter(tio_data.validation_loader_patches))
    patch_size=140
    k = int(patch_size // 2)
    batch_T1 = patches_batch['T1w']['data'][...,k]
    batch_T2 = patches_batch['T2w']['data'][...,k]
    batch_pred_T2 = pred_T2[..., k].to(torch.device('cpu'))
    slices = torch.cat((batch_T1, batch_T2, batch_pred_T2))
    image_path = 'batch_T1toT2_patches.png'
    torchvision.utils.save_image(
        slices,
        image_path,
        nrow=tio_data.training_batch_size,
        normalize=True,
        scale_each=True
    )
    #display.Image(image_path)
    return

def generate(image_paths, model, device, params):
    net = UNet()
    net.load_state_dict(torch.load(model, map_location=device))
    net.to(device=device)

    net.eval()

    with torch.no_grad():
        data_loader = DataLoader(tio_data, batch_size=1)
        for path in image_paths:
            input_T1 = tio.ScalarImage(path)['data'].to(device)
            input_T1 = input_T1[None] # Prepend tensor with batch placeholder dimension
            T2_pred = net(input_T1)
            pred_data = T2_pred.cpu().detach().numpy()
            pred_data_for_nii = pred_data[0,0,:,:,:]
            pred_data_for_nii = pred_data_for_nii - pred_data_for_nii.min()
            new_image = nib.Nifti1Image(pred_data_for_nii, affine=np.eye(4))
            new_name = path.replace('pre', 'synth')
            nib.save(new_image, 'test_desc-synthetic_T2w.nii.gz')
            

    return


def get_cli_args():
    
    parser = argparse.ArgumentParser(
        description="UNet for estimating T1w and T2w images"
    )
    parser.add_argument(
        "--config",
        type=str,
        help=("Path to a yaml file containing all of the required parameters")
    )
    parser.add_argument(
        "--img",
        type=str,
        help=("Path to an antatomical image for translation")
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
        tio_data = TIOPreprocessing(data_dir=params['training_data'])
        tio_data.load_data()
        tio_data.split_training_data()
        train(tio_data, device, params)

    elif params['generate']:
        print('Generating')
        
        # First create transform
        T1_landmarks_path=params['T1_landmarks_path']
        if T1_landmarks_path:
            landmarks_dict = {'T1w': np.load(T1_landmarks_path)}
            histogram_transform = tio.HistogramStandardization(landmarks_dict)
        composed_transform = tio.Compose([histogram_transform])
        
        # List out image paths and create tio dataset
        if args.img:
            T1_img_paths = [args.img]
        else:
            T1_img_paths = sorted(glob.glob(os.path.join(params['test_data'], '*.nii.gz')))
        for T1_path in T1_img_paths:
            subject = tio.Subject(T1w=tio.ScalarImage(T1_path))
            tio_data = tio.SubjectsDataset([subject], transform=composed_transform)
            prediction_data = generate(tio_data, params['weights_path'], device, params)
            orig_img = nib.load(T1_path)
            orig_affine = orig_img.affine
            #prediction_data = (prediction_data - prediction_data.min()) * (orig_img.get_fdata().max() / prediction_data.max())
            #prediction_img = nib.Nifti1Image(prediction_data, orig_affine)
            prediction_img = nib.Nifti1Image(prediction_data, orig_affine)
            synth_img_path = T1_path.replace('/pre/', '/synth/').replace('_T1w.nii.gz','_desc-synthetic_T2w.nii.gz')
            nib.save(prediction_img, synth_img_path)
            print('Generated {}'.format(synth_img_path))   
                

if __name__ == '__main__':
    main()



