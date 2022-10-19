import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import nibabel as nib
import torch
from torch import optim
import torch.nn as nn
import torchvision
import torchio as tio

from tio_preprocessing import TIOPreprocessing

from model import UNet

#subject_path = '/home/faird/shared/data/INDIA/sub-06IND001B/ses-001/anat/sub-06IND001B_ses-001_T1w.nii.gz'
#output = '/home/faird/shared/projects/3D_MRI_GAN/tio_unet/data/sub-100619_ses-17mo_T2w_p1.nii.gz'
#subject_path = '/home/faird/shared/projects/3D_MRI_GAN/tio_unet/data/sub-06IND003G_ses-003_T1w_acpcDenoiseN4dc.nii.gz'
#output = '/home/faird/shared/projects/3D_MRI_GAN/tio_unet/data/sub-06IND003G_ses-003_T2w.nii.gz'
#subject_path = '/home/faird/shared/projects/3D_MRI_GAN/tio_unet/data/sub-06IND001B_ses-001_T1w_acpcDenoiseN4dc.nii.gz'
#output = '/home/faird/shared/projects/3D_MRI_GAN/tio_unet/data/sub-06IND001B_ses-001_T2w.nii.gz'
#subject_path = '/home/faird/shared/projects/3D_MRI_GAN/tio_unet/data/sub-06IND161G_ses-002_T1w_acpcDenoiseN4dc.nii.gz'
#output = '/home/faird/shared/projects/3D_MRI_GAN/tio_unet/data/sub-06IND161G_ses-002_T2w_std.nii.gz'

subject_path = sys.argv[1]
output = sys.argv[2]

new_subject = tio.Subject(T1w=tio.ScalarImage(subject_path))

T1_landmarks = np.load('/home/faird/shared/projects/3D_MRI_GAN/tio_unet/experiments/ECHO_51/ECHO_T1w_landmarks.npy')
#T1_landmarks = np.load('/home/faird/shared/projects/3D_MRI_GAN/T1w_landmarks.npy')
landmarks_dict = {'T1w': T1_landmarks}
histogram_transform = tio.HistogramStandardization(landmarks_dict)

znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)

transform = tio.Compose([histogram_transform, znorm_transform])
#transform = tio.Compose([znorm_transform])
znormed = transform(new_subject)

net = UNet()
#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load('/home/faird/shared/projects/3D_MRI_GAN/tio_unet/experiments/ECHO_51/T1toT2_model_gpu_epoch-130.pth', map_location=device))
#net.load_state_dict(torch.load('/home/faird/shared/projects/3D_MRI_GAN/tio_unet/trained_models/T1toT2_model_gpu_epoch-137.pth', map_location=device))
#net.load_state_dict(torch.load('/home/faird/shared/projects/3D_MRI_GAN/tio_unet/T1toT2_model_gpu_epoch-199.pth', map_location=device))
net.to(device=device)

patch_size = 182, 182, 182
patch_overlap = 16, 16, 16
grid_sampler = tio.inference.GridSampler(
    znormed,
    patch_size,
    patch_overlap,
)
patch_loader = torch.utils.data.DataLoader(
    grid_sampler, batch_size=1)
aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

net.eval()
with torch.no_grad():
    for patches_batch in patch_loader:
        inputs = patches_batch['T1w'][tio.DATA].to(device)
        locations = patches_batch[tio.LOCATION]
        pred_T2 = net(inputs)
        aggregator.add_batch(pred_T2, locations)
        
prediction = aggregator.get_output_tensor()

T2_pred_subject = tio.Subject(T2w=tio.ScalarImage(tensor=prediction))
#T2_landmarks = np.load('/home/faird/shared/projects/3D_MRI_GAN/T2w_landmarks.npy')
T2_landmarks = np.load('/home/faird/shared/projects/3D_MRI_GAN/tio_unet/experiments/ECHO_51/ECHO_T2w_landmarks.npy')
T2_landmarks_dict = {'T2w': T2_landmarks}
T2_hist_transform = tio.HistogramStandardization(T2_landmarks_dict)
transform = tio.Compose([T2_hist_transform])
T2_std = transform(T2_pred_subject)
prediction_data = T2_std.T2w.data[0,:,:,:].numpy()
prediction_data = prediction_data.clip(0)
#prediction_data = T2_pred_subject.T2w.data[0,:,:,:].numpy()
#prediction_data = prediction_data.clip(0)

orig_img = nib.load(subject_path)
orig_affine = orig_img.affine

#prediction_data = prediction[0,:,:,:].data.numpy()
#prediction_data = (prediction_data - prediction_data.min()) * orig_img.get_fdata().max()
prediction_data = (prediction_data - prediction_data.min()) * (orig_img.get_fdata().max() / prediction_data.max())


pred_img = nib.Nifti1Image(prediction_data, orig_affine)
nib.save(pred_img, output) 


