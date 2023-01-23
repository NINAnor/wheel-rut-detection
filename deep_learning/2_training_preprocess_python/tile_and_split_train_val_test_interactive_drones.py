import os
import csv
import random
import numpy as np

from osgeo import gdal
from shutil import copyfile

def write_array_to_tif(new_file, transform, projection, band_count,
                       value_array_list,
                       nodata_value_list, datatype, metadata_dict=None):
    """
    Trying to save tif using Create()
    """
    
    if (band_count != len(value_array_list)):
        print('ERROR: band_count != len(value_array_list)')
    elif (band_count != len(nodata_value_list)):
        print('ERROR: band_count != len(nodata_value_list)')
    else:
        driver = gdal.GetDriverByName('GTiff')
        new_ds = driver.Create(new_file,
                               value_array_list[0].shape[1],
                               value_array_list[0].shape[0],
                               band_count, datatype,
                               ['BIGTIFF=YES', 'TILED=YES', 'COMPRESS=DEFLATE'])
        
        new_ds.SetGeoTransform(transform)
        new_ds.SetProjection(projection)
        
        for i in np.arange(band_count):
            band = new_ds.GetRasterBand(int(i+1))
            
            nodata_value = nodata_value_list[i]
            if nodata_value is not None:
                band.SetNoDataValue(nodata_value)
            
            band.WriteArray(value_array_list[i])
        
        if metadata_dict:
            for k, v in metadata_dict.items():
                new_ds.SetMetadataItem(str(k),str(v))
        
        new_ds = None


img_target_size = 300

data_dir = '1_training_preprosess_bash'

balsfjord_drone_file = os.path.join(data_dir, 'drone',
                                    '2020_10_balsfjord_rgb_low_res.tiff')
balsfjord_mask_file = os.path.join(data_dir, 'ground_truth',
                                    'balsfjord_drone_track_mask.tif')

rjukan_drone_file = os.path.join(data_dir, 'drone',
                                    '2020_10_rjukan_rgb.tif')
rjukan_mask_file = os.path.join(data_dir, 'ground_truth',
                                    'rjukan_drone_track_mask.tif')

dataset_dir = os.path.join(data_dir, 'pytorch_datasets',
                           'semantic_segmentation_drone_300_rgb')


train_dir = os.path.join(dataset_dir, 'train')
if not os.path.isdir(os.path.join(train_dir, 'images')):
    os.makedirs(os.path.join(train_dir, 'images'))

if not os.path.isdir(os.path.join(train_dir, 'masks')):
    os.makedirs(os.path.join(train_dir, 'masks'))

val_dir = os.path.join(dataset_dir, 'val')
if not os.path.isdir(os.path.join(val_dir, 'images')):
    os.makedirs(os.path.join(val_dir, 'images'))

if not os.path.isdir(os.path.join(val_dir, 'masks')):
    os.makedirs(os.path.join(val_dir, 'masks'))

test_dir = os.path.join(dataset_dir, 'test')
if not os.path.isdir(os.path.join(test_dir, 'images')):
    os.makedirs(os.path.join(test_dir, 'images'))

if not os.path.isdir(os.path.join(test_dir, 'masks')):
    os.makedirs(os.path.join(test_dir, 'masks'))

rgb_band_count = 3
red_band = 1
green_band = 2
blue_band = 3
alpha_band = 4
mask_band = 1

file_id = 0

for image_file, mask_file in zip(
        [balsfjord_drone_file, rjukan_drone_file],
        [balsfjord_mask_file, rjukan_mask_file]):
    
    try:
        ds = gdal.Open(image_file, gdal.GA_ReadOnly)
    except RuntimeError as e:
        print('Unable to open INPUT.tif')
        print(e)
        
    img_projection = ds.GetProjection()
    img_transform = ds.GetGeoTransform()
    
    try:
        band = ds.GetRasterBand(red_band)
    except RuntimeError as e:
        print('Band ( %i ) not found' % red_band)
        print(e)
        
    red_nodata_value = band.GetNoDataValue()
    red_array = band.ReadAsArray()    
    
    try:
        band = ds.GetRasterBand(green_band)
    except RuntimeError as e:
        print('Band ( %i ) not found' % green_band)
        print(e)
        
    green_nodata_value = band.GetNoDataValue()
    green_array = band.ReadAsArray()
    
    try:
        band = ds.GetRasterBand(blue_band)
    except RuntimeError as e:
        print('Band ( %i ) not found' % blue_band)
        print(e)
        
    blue_nodata_value = band.GetNoDataValue()
    blue_array = band.ReadAsArray()
    
    try:
        band = ds.GetRasterBand(alpha_band)
    except RuntimeError as e:
        print('Band ( %i ) not found' % alpha_band)
        print(e)
        
    alpha_nodata_value = band.GetNoDataValue()
    alpha_array = band.ReadAsArray()
    
    ds = None
    
    try:
        ds = gdal.Open(mask_file, gdal.GA_ReadOnly)
    except RuntimeError as e:
        print('Unable to open INPUT.tif')
        print(e)
        
    mask_projection = ds.GetProjection()
    mask_transform = ds.GetGeoTransform()
    
    if (mask_projection != img_projection):
        sys.exit(1)
        
    if (mask_transform != img_transform):
        sys.exit(1)
        
    try:
        band = ds.GetRasterBand(mask_band)
    except RuntimeError as e:
        print('Band ( %i ) not found' % mask_band)
        print(e)
        
    mask_nodata_value = band.GetNoDataValue()
    mask_array = band.ReadAsArray()
    
    ds = None
    
    
    ns_tile_number = int(np.round(mask_array.shape[0] / img_target_size))
    ew_tile_number = int(np.round(mask_array.shape[1] / img_target_size))
    
    slice_list = []
    current_ns_index = 0
    for ns_split_array in np.array_split(mask_array, ns_tile_number, axis=0):
        ns_slice = slice(current_ns_index, 
                         current_ns_index + ns_split_array.shape[0])
        current_ns_index += ns_split_array.shape[0]
        
        current_ew_index = 0
        for ew_split_array in np.array_split(ns_split_array, ew_tile_number,
                                             axis=1):
            ew_slice = slice(current_ew_index,
                             current_ew_index + ew_split_array.shape[1])
            current_ew_index += ew_split_array.shape[1]
            
            slice_list.append((ns_slice, ew_slice))
            
            
    for slice_tuple in slice_list:
        ns_slice, ew_slice = slice_tuple
        
        if (np.any(mask_array[slice_tuple])
            and np.all(alpha_array[slice_tuple] != 0)):
            
            print(file_id, end='\r')
            
            current_transform = (
                mask_transform[0] + (ew_slice.start * mask_transform[1]),
                mask_transform[1],
                mask_transform[2],
                mask_transform[3] + (ns_slice.start * mask_transform[5]),
                mask_transform[4],
                mask_transform[5])
            
            split_value = random.uniform(0, 1)
            if (split_value < 0.7):
                current_image_savefile = os.path.join(
                    train_dir, 'images',
                    'image_' + str(file_id).zfill(6) + '.tif')
                current_mask_savefile = os.path.join(
                    train_dir, 'masks',
                    'mask_' + str(file_id).zfill(6) + '.tif')
            elif ((split_value >= 0.7) & (split_value < 0.85)):
                current_image_savefile = os.path.join(
                    val_dir, 'images',
                    'image_' + str(file_id).zfill(6) + '.tif')
                current_mask_savefile = os.path.join(
                    val_dir, 'masks',
                    'mask_' + str(file_id).zfill(6) + '.tif')
            else:
                current_image_savefile = os.path.join(
                    test_dir, 'images',
                    'image_' + str(file_id).zfill(6) + '.tif')
                current_mask_savefile = os.path.join(
                    test_dir, 'masks', 'mask_' + str(file_id).zfill(6) + '.tif')
                
            write_array_to_tif(
                current_image_savefile, current_transform, mask_projection,
                rgb_band_count,
                [red_array[slice_tuple],
                 green_array[slice_tuple],
                 blue_array[slice_tuple]],
                [red_nodata_value, green_nodata_value, blue_nodata_value],
                gdal.GDT_Byte)
            write_array_to_tif(
                current_mask_savefile, current_transform, mask_projection,
                1, [mask_array[slice_tuple]],
                [mask_nodata_value],
                gdal.GDT_Byte)
            
            file_id += 1
    
    red_array = None
    green_array = None
    blue_array = None
    alpha_array = None
    mask_array = None
    print()

