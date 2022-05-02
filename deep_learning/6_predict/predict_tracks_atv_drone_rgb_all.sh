python_venv="your_python_venv"
source "${python_venv}/bin/activate"

# NOTE: to predict, cd to project dir and copy predict.py there
#       (or another dir with modified files)

project_dir="your_project_dir"
model_dir="${project_dir}/model_save_dir"
model_file="${model_dir}/your_model_file"

base_data_dir="your_data_dir"

predictions_dir="your_predictions_dir"
mkdir -p "${predictions_dir}"

image_dir="balsfjord"

input_dir="${base_data_dir}/drone_image_tiles_all/${image_dir}"
output_dir="${predictions_dir}/${image_dir}"
mkdir -p "${output_dir}"

prediction_tif_prefix="prediction_best_"

python predict.py \
         -model_file  "${model_file}" \
         -input_tif_dir "${input_dir}" \
         -result_save_dir "${output_dir}" \
         -prediction_tif_prefix "${prediction_tif_prefix}"

output_vrt="${predictions_dir}/predictions_${image_dir}_drone_20220221.vrt"
gdalbuildvrt -overwrite "${output_vrt}" "${output_dir}"/*.tif
output_tif="${predictions_dir}/predictions_${image_dir}_drone_20220221.tif"
gdal_translate -co "COMPRESS=DEFLATE" \
               -co "TILED=YES" \
               "${output_vrt}" "${output_tif}"


image_dir="rjukan_1"

input_dir="${base_data_dir}/drone_image_tiles_all/${image_dir}"
output_dir="${predictions_dir}/${image_dir}"
mkdir -p "${output_dir}"

prediction_tif_prefix="prediction_best_"

python predict.py \
         -model_file  "${model_file}" \
         -input_tif_dir "${input_dir}" \
         -result_save_dir "${output_dir}" \
         -prediction_tif_prefix "${prediction_tif_prefix}"

output_vrt="${predictions_dir}/predictions_${image_dir}_drone_20220221.vrt"
gdalbuildvrt -overwrite "${output_vrt}" "${output_dir}"/*.tif
output_tif="${predictions_dir}/predictions_${image_dir}_drone_20220221.tif"
gdal_translate -co "COMPRESS=DEFLATE" \
               -co "TILED=YES" \
               "${output_vrt}" "${output_tif}"


image_dir="rjukan_2"

input_dir="${base_data_dir}/drone_image_tiles_all/${image_dir}"
output_dir="${predictions_dir}/${image_dir}"
mkdir -p "${output_dir}"

prediction_tif_prefix="prediction_best_"

python predict.py \
         -model_file  "${model_file}" \
         -input_tif_dir "${input_dir}" \
         -result_save_dir "${output_dir}" \
         -prediction_tif_prefix "${prediction_tif_prefix}"

output_vrt="${predictions_dir}/predictions_${image_dir}_drone_20220221.vrt"
gdalbuildvrt -overwrite "${output_vrt}" "${output_dir}"/*.tif
output_tif="${predictions_dir}/predictions_${image_dir}_drone_20220221.tif"
gdal_translate -co "COMPRESS=DEFLATE" \
               -co "TILED=YES" \
               "${output_vrt}" "${output_tif}"


# rjukan combined

rjukan_1_tif="${predictions_dir}/predictions_rjukan_1_drone_20220221.tif"
rjukan_2_tif="${predictions_dir}/predictions_rjukan_2_drone_20220221.tif"
rjukan_drone_tif="${predictions_dir}/predictions_rjukan_drone_20220221.tif"
gdal_merge.py -init 0 -a_nodata 0 \
              -co "COMPRESS=DEFLATE" \
              -co "TILED=YES" \
              -o "${rjukan_drone_tif}" "${rjukan_1_tif}" "${rjukan_2_tif}"

