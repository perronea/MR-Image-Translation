#! /usr/bin/env python

import os

import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats

import torch
import torchvision
import torchio as tio
import torch.nn.functional as F
from pathlib import Path


class TIOPreprocessing:
    def __init__(self, data_dir='data/MR',
                       T1_landmarks_path=None,
                       T2_landmarks_path=None):

        # Get paths to training data
        self.data_dir = data_dir
        self.T1_dir = Path(os.path.join(self.data_dir, "T1w"))
        self.T2_dir = Path(os.path.join(self.data_dir, "T2w"))
        self.T1_img_paths = sorted(self.T1_dir.glob('*.nii.gz')) 
        self.T2_img_paths = sorted(self.T2_dir.glob('*.nii.gz')) 

        # Training data must be paired
        assert len(self.T1_img_paths) == len(self.T2_img_paths)

        self.subjects = []
        self.dataset = None

        #self.load_training_data()

        self.T1_landmarks_path = T1_landmarks_path
        self.T2_landmarks_path = T2_landmarks_path
        self.T1_landmarks = None
        self.T2_landmarks = None

        #self.histogram_landmarks()

        self.znorm_transform = None

        self.training_transform = None
        self.validation_transform = None

        #self.compose_preprocessing()

        self.training_split_ratio = 0.8

        #self.split_training_data()


        patch_based_training = True
        if patch_based_training:
            self.training_batch_size = 1
            self.validation_batch_size = 1
            self.patch_size = 140
            self.samples_per_volume = 4
            self.max_queue_length = 20
            self.num_workers = 1
            #self.generate_patches()


    def load_data(self):

        for (T1_path, T2_path) in zip(self.T1_img_paths, self.T2_img_paths):
            subject = tio.Subject(
                T1w=tio.ScalarImage(T1_path),
                T2w=tio.ScalarImage(T2_path)
            )
            self.subjects.append(subject)
        self.dataset = tio.SubjectsDataset(self.subjects)
        print('Dataset size:', len(self.dataset), 'subjects')
        subject_1 = self.dataset[0]
        print(subject_1)
        print(subject_1.T1w)
        print(subject_1.T2w)

        return


    def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):

        values = tensor.numpy().ravel()
        kernel = stats.gaussian_kde(values)
        positions = np.linspace(values.min(), values.max(), num=num_positions)
        histogram = kernel(positions)
        kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
        if label is not None:
            kwargs['label'] = label
        axis.plot(positions, histogram, **kwargs)

        return


    def find_histogram_landmarks(self, train=True):

        if train:
            self.T1_landmarks = tio.HistogramStandardization.train(
                    self.T1_img_paths,
                    output_path = self.T1_landmarks_path,
                    )
            self.T2_landmarks = tio.HistogramStandardization.train(
                    self.T2_img_paths,
                    output_path = self.T2_landmarks_path,
            )
        elif os.path.exists(self.T1_landmarks_path) and os.path.exists(self.T2_landmarks_path):
            self.T1_landmarks = np.load(self.T1_landmarks_path)
            self.T2_landmarks = np.load(self.T2_landmarks_path)
        else:
            print('ERROR Missing .npy file containing histogram landmarks for T1w/T2w images')
            sys.exit(1)

        np.set_printoptions(suppress=True, precision=3)
        print('\nTrained T1w landmarks', self.T1_landmarks)
        print('\nTrained T2w landmarks', self.T2_landmarks)

        return


    def znormalize(self):
        
        self.znorm_transform = tio.ZNormalization(masking_method=tio.Znormalization.mean)

        return


    def compose_preprocessing(self):

        self.training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.HistogramStandardization({'T1w':self.T1_landmarks, 'T2w':self.T2_landmarks}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomFlip(axes='LR'),
            tio.OneOf({
                tio.RandomAffine(): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            })
        ])
        self.validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.HistogramStandardization({'T1w':self.T1_landmarks, 'T2w':self.T2_landmarks}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])
        """
        self.training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomFlip(axes='LR'),
            tio.OneOf({
                tio.RandomAffine(): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            })
        ])
        self.validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])
        """
    
        return


    def split_training_data(self):

        self.num_subjects = len(self.dataset)
        self.num_training_subjects = int(self.training_split_ratio * self.num_subjects)
        self.num_validation_subjects = self.num_subjects - self.num_training_subjects

        self.num_split_subjects = self.num_training_subjects, self.num_validation_subjects
        self.training_subjects, self.validation_subjects = torch.utils.data.random_split(self.subjects, self.num_split_subjects)

        self.training_set = tio.SubjectsDataset(
            self.training_subjects, transform=self.training_transform)

        self.validation_set = tio.SubjectsDataset(
            self.validation_subjects, transform=self.validation_transform)

        print('Training set: {} subjects'.format(len(self.training_set)))
        print('Validation set: {} subjects'.format(len(self.validation_set)))

        return


    def generate_patches(self):

        self.sampler = tio.data.UniformSampler(self.patch_size)

        self.patches_training_set = tio.Queue(
            subjects_dataset=self.training_set,
            max_length=self.max_queue_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        self.patches_validation_set = tio.Queue(
            subjects_dataset=self.validation_set,
            max_length=self.max_queue_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.num_workers,
            shuffle_subjects=False,
            shuffle_patches=False
        )

        self.training_loader_patches = torch.utils.data.DataLoader(
            self.patches_training_set, batch_size=self.training_batch_size)
        self.validation_loader_patches = torch.utils.data.DataLoader(
            self.patches_validation_set, batch_size=self.validation_batch_size)

        return

    def visualize_training_data(self):

        print("TODO")

        return


def main():

    print("TODO: dev main")

    return

        
if __name__ == "__main__":
    main()



