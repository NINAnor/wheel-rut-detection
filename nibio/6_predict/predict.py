#!/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import sys
import numpy as np

from osgeo import gdal, ogr, osr

import torch
import torchvision

def usage():
    print('Usage: predict.py \\')
    print('    -model_file model_file \\')
    print('    -input_tif_dir input_tif_dir \\')
    print('    -result_save_dir result_save_dir \\')
    print('    -prediction_tif_prefix prediction_tif_prefix \\')


def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    model_file = None
    input_tif_dir = None
    result_save_dir = None

    # optional
    prediction_tif_prefix = 'prediction_'

    # maybe optional later
    num_classes = 2
    in_channels = 3
    result_nodata_value = 0
    default_band = 1

    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '-model_file':
            i = i + 1
            model_file = sys.argv[i]

        elif arg == '-input_tif_dir':
            i = i + 1
            input_tif_dir = sys.argv[i]

        elif arg == '-result_save_dir':
            i = i + 1
            result_save_dir = sys.argv[i]

        elif arg == '-prediction_tif_prefix':
            i = i + 1
            prediction_tif_prefix = sys.argv[i]

        else:
            print('Unrecognised command option: %s' % arg)
            usage()
            sys.exit( 1 )
        
        i = i + 1
    
    if ((model_file is None)
        | (not model_file)
        | (input_tif_dir is None)
        | (not input_tif_dir)
        | (result_save_dir is None)
        | (not result_save_dir)):

        print('check your argument list')
        print('')
        print('model_file: ', model_file)
        print('input_tif_dir: ', input_tif_dir)
        print('result_save_dir: ', result_save_dir)

        usage()
        sys.exit( 1 )


    if (not os.path.isfile(model_file)):
        print('ERROR: model_file is not a file: ' + model_file)
        usage()
        sys.exit( 1 )

    if (not os.path.isdir(input_tif_dir)):
        print('ERROR: input_tif_dir is not a dir: ' + input_tif_dir)
        usage()
        sys.exit( 1 )


    # Note: Loading av modeller kan være tricky. 

    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')

    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        pretrained_backbone=False,
        num_classes=num_classes)

    model.to(device)

    model.load_state_dict(torch.load(model_file, map_location=device))

    model.to(device)
    model.eval()

    input_tif_file_list = glob.glob(os.path.join(
        input_tif_dir, '*.tif'))

    for input_tif_file in input_tif_file_list:

        print('Predicting', input_tif_file, flush=True)

        result_savefile = os.path.join(
            result_save_dir, prediction_tif_prefix + input_tif_file[-10:])

        image_array, projection, transform, tif_epsg_code \
            = get_array_from_tif(input_tif_file, in_channels)
        image_array = np.expand_dims(image_array, 0)
        image_array = torch.as_tensor(image_array)
        image_array = image_array.cuda()

        out_dict = model(image_array)

        out_array = out_dict['out'].cpu().squeeze().detach().numpy()

        prediction_array = out_array.argmax(axis=0).astype(np.uint8)

        write_array_to_tif(
            result_savefile, transform, projection,
            default_band, [prediction_array],
            [result_nodata_value],
            gdal.GDT_Byte)


def get_array_from_tif(tif_file, in_channels):
    
    try:
        ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
        
    if (ds.RasterCount < in_channels):
        print(f'ERROR: ds.RasterCount < self.in_channels',
              f'({ds.RasterCount} < {self.in_channels}.')
        exit(1)
        
    array_list = []
    
    for band_no in range(1, in_channels + 1):
        
        try:
            band = ds.GetRasterBand(band_no)
        except RuntimeError as e:
            print('Band ( %i ) not found' % default_band)
            print(e)
            
        array_list.append(band.ReadAsArray().astype(np.float32))
    
    projection = ds.GetProjection()
    transform = ds.GetGeoTransform()

    tif_srs = ds.GetSpatialRef()
    res = tif_srs.AutoIdentifyEPSG()
    if res == 0:
        tif_epsg_code = int(tif_srs.GetAuthorityCode(None))
    else:
        tif_proj4 = tif_srs.ExportToProj4()

        if ((re.search('proj=utm', tif_proj4) is not None)
            & (re.search('zone=33', tif_proj4) is not None)
            & (re.search('ellps=GRS80', tif_proj4) is not None)):

            tif_epsg_code = 25833
            tif_srs.ImportFromEPSG(tif_epsg_code)
        else:
            print('Could not determine SRID')
            sys.exit( 1 )
    
    ds = None
    
    image_array = np.array(array_list)
    
    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, 0)
        
    image_array = torch.as_tensor(image_array, dtype=torch.float32)

    # change to [0,1]
    image_array = image_array / 255.0

    # dataset_mean = [0.485, 0.456, 0.406]
    # dataset_std = [0.229, 0.224, 0.225]

    # image_array = torchvision.transforms.functional.normalize(
    #     image_array,
    #     dataset_mean, dataset_std)

    return image_array, projection, transform, tif_epsg_code



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



if __name__ == '__main__':
    sys.exit(main())
