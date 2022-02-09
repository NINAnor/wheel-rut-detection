#!/bin/bash

# Script for re-processing raw drone imagery with Docker installation of OpenDroneMap on a 48 cores, 378 GB memory server with 1.8TB local SSD storage

# For processing large datasets, more adustments with regards to memory limitations might be necessary. See:
# https://github.com/OpenDroneMap/odm-benchmarks/issues/10
# https://docs.opendronemap.org/large/#

basedir="/data/scratch"
resultdir="/data/P-Prosjekter/15215600_autmatisert_kartlegging_av_barmarksskader/ODM_results"
cd $basedir


# Download data
wget --header="Accept: text/html" --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" "https://fjernmaalingsdata.file.core.windows.net/barmarksskader/2021_barmarksskader/2020_10_andoya_space_center/2020_10_andoya_space_center.7z?st=2021-09-30T08%3A48%3A00Z&se=2021-12-31T09%3A48%3A00Z&sp=rl&sv=2018-03-28&sr=f&sig=QuB6Om7H4ilqQvN3ttDPGWYLWvVSXshXfuSXCMCamCY%3D"
# Rename to get a proper suffix (otherwise p7zip will complain)
mv '2020_10_andoya_space_center.7z?st=2021-09-30T08:48:00Z&se=2021-12-31T09:48:00Z&sp=rl&sv=2018-03-28&sr=f&sig=QuB6Om7H4ilqQvN3ttDPGWYLWvVSXshXfuSXCMCamCY=' 2020_10_andoya_space_center.7z

# Upack the archive
p7zip -d 2020_10_andoya_space_center.7z

# Unify subfolder structure for both areas
mv Rjukan/P4PRO\ RGB/Day1 Rjukan/P4P_Rjukan_Day1
mv Rjukan/P4PRO\ RGB/Day2 Rjukan/P4P_Rjukan_Day2
rm -r Rjukan/P4PRO\ RGB


# Prepare data in tasks: split by area, day and sensor (multispectral + RGB)
# RGB images of the P4M sensor have a significantly lower resolution (1600x1330 pixels) compared to the RGB images from the P4P sensor (4864x3648 pixels)
# So, RGB images from the P4M sensor are dropped from processing

for area in Skutviksvatnet Rjukan
do
    for d in 1 2
    do
        for s in P M
        do
            acquisition="P4${s}_${area}_Day${d}"
            mkdir "${basedir}/$acquisition"
            mkdir "${basedir}/${acquisition}/images"
            cd "${basedir}/${acquisition}/images"
            # RGB is stored in JPG, Multispectral in TIF
            if [ "$s" == "P" ] ; then
                suffix=JPG
            else
                suffix=TIF
            fi
            # Unfortunately, images need to be copied as linked images (ln -s) are not read properly
            # Image names are not unique across subdirectories, thus renaming them here
            find ${basedir}/${area}/ -wholename "*P4${s}_*_Day${d}*/*.${suffix}" | awk -v FS="/" '{print("cp " $0 " ./" $6 "_" $7 "\0")}' | xargs -0 -P 40 -I{} bash -c "{}"
            cd "$basedir"
        done
    done
done

# Handle corrupt images
# Fix corrupt JPG image data
jpegtran -perfect -copy all -outfile DJI_0421_fixed.JPG P4P_Rjukan_Day1/102MEDIA_DJI_0421.JPG
mv DJI_0421_fixed.JPG P4P_Rjukan_Day1/102MEDIA_DJI_0421.JPG
# Remove image with lack of GPS data (0,0 coordinates)
rm P4P_Rjukan_Day1/images/100MEDIA_DJI_0709.JPG


# Define target resolution
res=5
for acquisition in Skutviksvatnet_Day1 Rjukan_Day2 Rjukan_Day1 Skutviksvatnet_Day2 
do
    for s in P M
    do
        acq=P4${s}_${acquisition}
        echo $acq
        if [ $(ls -1 $basedir/$acq/images | wc -l) -gt 800 ] ; then
            # The 378 GB RAM machine runs out of memory at ~800-900 images so splitting becomes necessry
            # --pc-tile causes artifacts within the DSM/DTM
            split="--split 150 --split-overlap 100"
        fi
        if [ $s == P ] ; then
            # Run RGB tasks
            # Focus is on extracting a high quality DSM/DTM
            sudo docker run -ti --rm -v ${basedir}/${acq}/\:/datasets/code opendronemap/odm --time --verbose --optimize-disk-space $split --dsm --dtm --dem-resolution $res --ignore-gsd --pc-quality ultra --feature-quality ultra --texturing-data-term area --cog --orthophoto-resolution $res --project-path /datasets &> ${basedir}/${acq}.log
        else
            # Run Multispectral Task
            # See: https://community.opendronemap.org/t/dji-phantom-4-pro-multipectral-w-o-rtk/4978/12
            # https://docs.opendronemap.org/multispectral/
            # https://www.opendronemap.org/2020/02/odm-0-9-8-adds-multispectral-16bit-tiffs-support-and-moar/
            sudo docker run -ti --rm -v ${basedir}/${acq}/\:/datasets/code opendronemap/odm --time --verbose --optimize-disk-space --radiometric-calibration camera --dsm  --dtm --dem-resolution $res --ignore-gsd --pc-quality ultra --feature-quality ultra --texturing-data-term area --cog --orthophoto-resolution $res --texturing-skip-global-seam-leveling --project-path /datasets &> ${basedir}/${acq}.log
        fi
        mkdir ${resultdir}/${acq}
        # Copy Benchmark
        cp ${basedir}/${acq}/benchmark.txt ${resultdir}/${acq}/
        # Copy log
        cp ${basedir}/${acq}/log.json ${resultdir}/${acq}/
        # Copy camera info
        cp ${basedir}/${acq}/cameras.json ${resultdir}/${acq}/
        # Copy report
        cp ${basedir}/${acq}/odm_report/* ${resultdir}/${acq}/
        # Copy DSM
        cp ${basedir}/${acq}/odm_dem/dsm.tif ${resultdir}/${acq}/${acq}_dsm.tif
        # Copy DTM
        cp ${basedir}/${acq}/odm_dem/dtm.tif ${resultdir}/${acq}/${acq}_dtm.tif
        # Copy Orthofoto
        cp ${basedir}/${acq}/odm_orthophoto/odm_orthophoto.tif ${resultdir}/${acq}/${acq}_orthophoto.tif
    done
done
