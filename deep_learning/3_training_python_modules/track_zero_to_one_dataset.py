import os
import glob
import numpy as np
import csv
from osgeo import gdal
import random

import torch
import torchvision


class TrackDataset(object):
    def __init__(self, base_dir, train_datatype=torch.float32,
                 train=False, max_1d_shape=None):

        # NOTE: It seems Deeplabv3 require the image to be in [0,1]
        #   and normalized using mean = [0.485, 0.456, 0.406] and
        #   std = [0.229, 0.224, 0.225].
        #   (https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
        # NOTE ALSO that this version put the images in [0,1], but do not
        #   normalize using mean and std above.
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]
        # self.dataset_mean = dataset_mean
        # self.dataset_std = dataset_std

        self.base_dir = base_dir
        self.train_datatype = train_datatype
        self.train = train
        self.max_1d_shape = max_1d_shape

        self.min_1d_shape = 224
        # self.center_crop_1d_shape = 224

        # not sure if np.array is necessary, but there may be
        #   memory leaks from list
        self.image_filename_array = np.array(list(sorted(glob.glob(os.path.join(
            base_dir, 'images', '*.tif')))))
        self.mask_filename_array = np.array(list(sorted(glob.glob(os.path.join(
            base_dir, 'masks', '*.tif')))))

        for image_str, mask_str in zip(self.image_filename_array,
                                       self.mask_filename_array):
            if (image_str[-10:] != mask_str[-10:]):
                self.image_filename_array = np.empty((0,))
                self.mask_filename_array = np.empty((0,))

        
    def __getitem__(self, idx):
        # image_file = os.path.join(self.base_dir, 'images', self.image_filename_array[idx])
        
        try:
            ds = gdal.Open(self.image_filename_array[idx], gdal.GA_ReadOnly)
        except RuntimeError as e:
            print('Unable to open INPUT.tif')
            print(e)
            sys.exit(1)
            
        array_list = []
        
        for band_no in range(1, ds.RasterCount + 1):
            
            try:
                band = ds.GetRasterBand(band_no)
            except RuntimeError as e:
                print('Band ( %i ) not found' % default_band)
                print(e)
                
            array_list.append(band.ReadAsArray())
            
        ds = None
        
        image_array = np.array(array_list)
        
        # mask_file = os.path.join(self.base_dir, "masks", self.mask_filename_array[idx])

        # print(self.mask_filename_array[idx], flush=True)
        
        try:
            ds = gdal.Open(self.mask_filename_array[idx], gdal.GA_ReadOnly)
        except RuntimeError as e:
            print('Unable to open INPUT.tif')
            print(e)
            sys.exit(1)
            
        # mask has only one band
        band_no = 1
        
        try:
            band = ds.GetRasterBand(band_no)
        except RuntimeError as e:
            print('Band ( %i ) not found' % default_band)
            print(e)
            
        mask_array = band.ReadAsArray().astype(np.int32)
        
        ds = None

        if ((mask_array.shape[0] < self.min_1d_shape)
            | (mask_array.shape[1] < self.min_1d_shape)):
            print('ERROR: mask_array.shape[0] < self.min_1d_shape or',
                  'mask_array.shape[1] < self.min_1d_shape')
            sys.exit(1)
        
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, 0)
        
        if mask_array.ndim == 2:
            mask_array = np.expand_dims(mask_array, 0)

        # float32 is needed by some transforms
        image_array = torch.as_tensor(image_array, dtype=torch.float32)
        mask_array = torch.as_tensor(mask_array, dtype=torch.int16)

        if self.train:

            rot_par = random.uniform(0, 1)
            if (rot_par > 0.5):
                rotation_parameter \
                    = torchvision.transforms.RandomRotation.get_params((-45,45))
                image_array = torchvision.transforms.functional.rotate(
                    image_array, rotation_parameter)
                mask_array = torchvision.transforms.functional.rotate(
                    mask_array, rotation_parameter)
            
            h_flip_par = random.uniform(0, 1)
            if (h_flip_par > 0.5):
                image_array = torchvision.transforms.functional.hflip(
                    image_array)
                mask_array = torchvision.transforms.functional.hflip(mask_array)

            resize_par = random.uniform(0, 1)
            if (resize_par > 0.9):
                resize_factor = random.uniform(0.8, 1.2)
                resize_shape = np.round(np.array(
                    image_array.numpy().shape[-2:]) * resize_factor).astype(int)
                resize_shape = tuple(np.where(resize_shape < self.min_1d_shape,
                                              self.min_1d_shape, resize_shape))

                image_array = torchvision.transforms.functional.resize(
                    image_array, resize_shape)
                mask_array = torchvision.transforms.functional.resize(
                    mask_array, resize_shape)

            blur_par = random.uniform(0, 1)
            if (blur_par > 0.8):
                kernel_size = 3
                image_array = torchvision.transforms.functional.gaussian_blur(
                    image_array, kernel_size)


        # change to [0,1]
        image_array = image_array / 255

        if self.train:

            # NOTE: this seems to require normalized tensors ([0, 1]?)
            sharpness_par = random.uniform(0, 1)
            if (sharpness_par > 0.8):
                sharpness_factor = random.uniform(0.7, 1.3)
                image_array \
                    = torchvision.transforms.functional.adjust_sharpness(
                        image_array, sharpness_factor)

            brightness_par = random.uniform(0, 1)
            if (brightness_par > 0.8):
                brightness_factor = random.uniform(0.7, 1.3)
                image_array \
                    = torchvision.transforms.functional.adjust_brightness(
                        image_array, brightness_factor)

            contrast_par = random.uniform(0, 1)
            if (contrast_par > 0.8):
                contrast_factor = random.uniform(0.7, 1.3)
                image_array \
                    = torchvision.transforms.functional.adjust_contrast(
                        image_array, contrast_factor)

            gamma_par = random.uniform(0, 1)
            if (gamma_par > 0.8):
                gamma_factor = random.uniform(0.7, 1.3)
                image_array \
                    = torchvision.transforms.functional.adjust_gamma(
                        image_array, gamma_factor)

        if self.max_1d_shape is not None:
            resize_shape = np.round(np.array(
                image_array.numpy().shape[-2:])).astype(int)
            if np.any(resize_shape > self.max_1d_shape):
                resize_shape = tuple(
                    np.where(resize_shape > self.max_1d_shape,
                             self.max_1d_shape, resize_shape))

                image_array = torchvision.transforms.functional.resize(
                    image_array, resize_shape)
                mask_array = torchvision.transforms.functional.resize(
                    mask_array, resize_shape)


        # train_datatype for train, float32 else
        if self.train:
            image_array = torch.as_tensor(image_array,
                                          dtype=self.train_datatype)

        return image_array, mask_array.squeeze()
    
    def __len__(self):
        return len(self.image_filename_array)


