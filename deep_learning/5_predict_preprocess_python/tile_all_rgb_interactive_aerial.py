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


image_target_size = 300

data_dir = 'your_data_dir'

balsfjord_aerial_file = os.path.join(data_dir, 'aerial',
                                    'Eksport-nib_balsfjord.tif')

rjukan_aerial_file = os.path.join(data_dir, 'aerial',
                                    'Eksport-nib.tif')

output_dir = 'your_output_dir'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rgb_band_count = 3
red_band = 1
green_band = 2
blue_band = 3

file_id = 0

for name, image_file in zip(['balsfjord', 'rjukan'],
                            [balsfjord_aerial_file, rjukan_aerial_file]):
    
    save_dir = os.path.join(output_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    try:
        ds = gdal.Open(image_file, gdal.GA_ReadOnly)
    except RuntimeError as e:
        print('Unable to open INPUT.tif')
        print(e)
        
    image_projection = ds.GetProjection()
    image_transform = ds.GetGeoTransform()
    
    image_srs = ds.GetSpatialRef()
    res = image_srs.AutoIdentifyEPSG()
    if res == 0:
        image_epsg_code = int(image_srs.GetAuthorityCode(None))
    else:
        print('Could not determine SRID')
        sys.exit( 1 )
    
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
    
    ds = None
    
    data_mask_array = ((red_array != red_nodata_value) &
                       (green_array != green_nodata_value) &
                       (blue_array != blue_nodata_value))
    
    ns_tile_number = int(np.round(red_array.shape[0] / image_target_size))
    ew_tile_number = int(np.round(red_array.shape[1] / image_target_size))
    
    slice_list = []
    current_ns_index = 0
    for ns_split_array in np.array_split(red_array, ns_tile_number, axis=0):
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
        
        if (np.any(data_mask_array[slice_tuple])):
            
            print(file_id, end='\r')
            
            current_transform = (
                image_transform[0] + (ew_slice.start * image_transform[1]),
                image_transform[1],
                image_transform[2],
                image_transform[3] + (ns_slice.start * image_transform[5]),
                image_transform[4],
                image_transform[5])
            
            current_image_savefile = os.path.join(
                    save_dir, 'image_' + str(file_id).zfill(6) + '.tif')
            
            red_part_array = red_array[slice_tuple]
            green_part_array = green_array[slice_tuple]
            blue_part_array = blue_array[slice_tuple]
            
            write_array_to_tif(
                current_image_savefile, current_transform, image_projection,
                rgb_band_count,
                [red_part_array,
                 green_part_array,
                 blue_part_array],
                [red_nodata_value, green_nodata_value, blue_nodata_value],
                gdal.GDT_Byte)
            
            file_id += 1
    
    red_array = None
    green_array = None
    blue_array = None
    data_mask_array = None
    print()

