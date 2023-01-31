project_dir="."
model_dir="${project_dir}/model_save_dir"
model_file="${model_dir}/your_model_file"

base_data_dir="your_data_dir"

predictions_dir="your_predictions_dir"
mkdir -p "${predictions_dir}"

image_dir="balsfjord"

input_dir="${base_data_dir}/aerial_image_tiles_all/${image_dir}"
output_dir="${predictions_dir}/${image_dir}"
mkdir -p "${output_dir}"

prediction_tif_prefix="prediction_best_"

python predict.py \
         -model_file  "${model_file}" \
         -input_tif_dir "${input_dir}" \
         -result_save_dir "${output_dir}" \
         -prediction_tif_prefix "${prediction_tif_prefix}"

output_vrt="${predictions_dir}/predictions_${image_dir}_aerial_20220222.vrt"
gdalbuildvrt -overwrite "${output_vrt}" "${output_dir}"/*.tif
output_tif="${predictions_dir}/predictions_${image_dir}_aerial_20220222.tif"
gdal_translate -co "COMPRESS=DEFLATE" \
               -co "TILED=YES" \
               "${output_vrt}" "${output_tif}"


image_dir="rjukan"

input_dir="${base_data_dir}/aerial_image_tiles_all/${image_dir}"
output_dir="${predictions_dir}/${image_dir}"
mkdir -p "${output_dir}"

prediction_tif_prefix="prediction_best_"

python predict.py \
         -model_file  "${model_file}" \
         -input_tif_dir "${input_dir}" \
         -result_save_dir "${output_dir}" \
         -prediction_tif_prefix "${prediction_tif_prefix}"

output_vrt="${predictions_dir}/predictions_${image_dir}_aerial_20220222.vrt"
gdalbuildvrt -overwrite "${output_vrt}" "${output_dir}"/*.tif
output_tif="${predictions_dir}/predictions_${image_dir}_aerial_20220222.tif"
gdal_translate -co "COMPRESS=DEFLATE" \
               -co "TILED=YES" \
               "${output_vrt}" "${output_tif}"

