db_host="your_db_host"
dbname="your_dbname"
db_user="your_db_user"

db_schema="your_db_schema"
db_fasit_table="your_fasit_table"

db_epsg_code="25832"
balsfjord_drone_tif_epsg="32634"
rjukan_drone_tif_epsg="32632"
balsfjord_aerial_tif_epsg="25834"
rjukan_aerial_tif_epsg="25832"

track_radius="0.6"

base_data_dir="your_dir"

balsfjord_gpkg_file="${base_data_dir}/ground_truth_gpkg/balsfjord_sf_withtile.gpkg"
rjukan_gpkg_file="${base_data_dir}/ground_truth_gpkg/rjukan_sf_withtile.gpkg"

# balsfjord_high_res_tif="${base_data_dir}/drone/2020_10_balsfjord_rgb.tiff"

balsfjord_drone_tif="${base_data_dir}/drone/2020_10_balsfjord_rgb_low_res.tiff"
rjukan_drone_tif="${base_data_dir}/drone/2020_10_rjukan_rgb.tif"

balsfjord_aerial_tif="${base_data_dir}/aerial/Eksport-nib_balsfjord.tif"
rjukan_aerial_tif="${base_data_dir}/aerial/Eksport-nib.tif"

# balsfjord_high_res_burn_tif="${base_data_dir}/ground_truth/balsfjord_track_mask_high_res.tif"

balsfjord_drone_burn_tif="${base_data_dir}/ground_truth/balsfjord_drone_track_mask.tif"
rjukan_drone_burn_tif="${base_data_dir}/ground_truth/rjukan_drone_track_mask.tif"

balsfjord_aerial_burn_tif="${base_data_dir}/ground_truth/balsfjord_aerial_track_mask.tif"
rjukan_aerial_burn_tif="${base_data_dir}/ground_truth/rjukan_aerial_track_mask.tif"

# gdalwarp -tr 0.07 -0.07 -r 'near' \
#          -co "BIGTIFF=YES" \
#          -co "TILED=YES" \
#          -co "COMPRESS=DEFLATE" \
#          "${balsfjord_high_res_tif}" "${balsfjord_drone_tif}"

# # copy to /space
# gdaladdo -ro 2020_10_balsfjord_rgb_low_res.tiff --config BIGTIFF YES \
#          --config BIGTIFF_OVERVIEW YES --config TILED YES \
#          --config COMPRESS_OVERVIEW DEFLATE 4 16 64 256 1024
# # mode is worse
# gdaladdo -ro 2020_10_balsfjord_rgb_low_res_mode.tiff --config BIGTIFF YES \
#          --config BIGTIFF_OVERVIEW YES --config TILED YES \
#          --config COMPRESS_OVERVIEW DEFLATE 4 16 64 256 1024

psql -h "${db_host}" -d "${dbname}" -U "${db_user}" -1 -c "
    DROP TABLE IF EXISTS ${db_schema}.${db_fasit_table};
    CREATE UNLOGGED TABLE ${db_schema}.${db_fasit_table} (
        track_id serial,
        track_type text,
        image_type text,
        geo geometry(Polygon, ${db_epsg_code}),
        CONSTRAINT ${db_fasit_table}_pk PRIMARY KEY (track_id)
      );

    CREATE INDEX ON ${db_schema}.${db_fasit_table} USING gist (geo);"


tmp_track_table="tmp_track"
psql -h "${db_host}" -d "${dbname}" -U "${db_user}" -1 -c "
    DROP TABLE IF EXISTS ${db_schema}.${tmp_track_table};"

ogr2ogr -f "PostgreSQL" \
        PG:"host=${db_host} dbname=${dbname} user=${db_user}" \
        -a_srs "EPSG:${db_epsg_code}" \
        -lco "unlogged=yes" \
        -lco "precision=no" \
        -lco "schema=${db_schema}" \
        -nln "${tmp_track_table}" \
        "${balsfjord_gpkg_file}"

psql -h "${db_host}" -d "${dbname}" -U "${db_user}" -1 -c "
    INSERT INTO ${db_schema}.${db_fasit_table} (track_type, image_type, geo)
        SELECT type AS track_type,
               CASE WHEN (tileend_2 = 0) THEN 'aerial'
                    WHEN (tileend_2 = 1) THEN 'drone'
                    ELSE 'unknown'
               END AS image_type,
               ST_Buffer(geom, ${track_radius}) AS geo
          FROM ${db_schema}.${tmp_track_table};

    DROP TABLE IF EXISTS ${db_schema}.${tmp_track_table};"


ogr2ogr -f "PostgreSQL" \
        PG:"host=${db_host} dbname=${dbname} user=${db_user}" \
        -a_srs "EPSG:${db_epsg_code}" \
        -lco "unlogged=yes" \
        -lco "precision=no" \
        -lco "schema=${db_schema}" \
        -nln "${tmp_track_table}" \
        "${rjukan_gpkg_file}"

psql -h "${db_host}" -d "${dbname}" -U "${db_user}" -1 -c "
    INSERT INTO ${db_schema}.${db_fasit_table} (track_type, image_type, geo)
        SELECT type AS track_type,
               CASE WHEN (tileend_2 = 0) THEN 'aerial'
                    WHEN (tileend_2 = 1) THEN 'drone'
                    ELSE 'unknown'
               END AS image_type,
               ST_Buffer(geom, ${track_radius}) AS geo
          FROM ${db_schema}.${tmp_track_table};

    DROP TABLE IF EXISTS ${db_schema}.${tmp_track_table};"


# drone balsfjord

# empty tif to burn on
gdal_calc.py --overwrite --type='Byte' --quiet \
             -A "${balsfjord_drone_tif}" \
             --A_band=1 \
             --outfile="${balsfjord_drone_burn_tif}" \
             --co "COMPRESS=DEFLATE" \
             --co "TILED=YES" \
             --calc "(0)"

gdal_rasterize -at -b 1 -burn 1 -sql "
    SELECT ST_Transform(geo, ${balsfjord_drone_tif_epsg})
      FROM ${db_schema}.${db_fasit_table}
     WHERE track_type = 'damage'
       AND image_type = 'drone'" \
  PG:"host='${db_host}' dbname='${dbname}' user='${db_user}' port='5432'" \
  "${balsfjord_drone_burn_tif}"


# drone rjukan

# empty tif to burn on
gdal_calc.py --overwrite --type='Byte' --quiet \
             -A "${rjukan_drone_tif}" \
             --A_band=1 \
             --outfile="${rjukan_drone_burn_tif}" \
             --co "COMPRESS=DEFLATE" \
             --co "TILED=YES" \
             --calc "(0)"

gdal_rasterize -at -b 1 -burn 1 -sql "
    SELECT ST_Transform(geo, ${rjukan_drone_tif_epsg})
      FROM ${db_schema}.${db_fasit_table}
     WHERE track_type = 'damage'
       AND image_type = 'drone'" \
  PG:"host='${db_host}' dbname='${dbname}' user='${db_user}' port='5432'" \
  "${rjukan_drone_burn_tif}"


# aerial balsfjord

# empty tif to burn on
gdal_calc.py --overwrite --type='Byte' --quiet \
             -A "${balsfjord_aerial_tif}" \
             --A_band=1 \
             --outfile="${balsfjord_aerial_burn_tif}" \
             --co "COMPRESS=DEFLATE" \
             --co "TILED=YES" \
             --calc "(0)"

gdal_rasterize -at -b 1 -burn 1 -sql "
    SELECT ST_Transform(geo, ${balsfjord_aerial_tif_epsg})
      FROM ${db_schema}.${db_fasit_table}
     WHERE track_type = 'damage'
       AND image_type = 'aerial'" \
  PG:"host='${db_host}' dbname='${dbname}' user='${db_user}' port='5432'" \
  "${balsfjord_aerial_burn_tif}"


# aerial rjukan

# empty tif to burn on
gdal_calc.py --overwrite --type='Byte' --quiet \
             -A "${rjukan_aerial_tif}" \
             --A_band=1 \
             --outfile="${rjukan_aerial_burn_tif}" \
             --co "COMPRESS=DEFLATE" \
             --co "TILED=YES" \
             --calc "(0)"

gdal_rasterize -at -b 1 -burn 1 -sql "
    SELECT ST_Transform(geo, ${rjukan_aerial_tif_epsg})
      FROM ${db_schema}.${db_fasit_table}
     WHERE track_type = 'damage'
       AND image_type = 'aerial'" \
  PG:"host='${db_host}' dbname='${dbname}' user='${db_user}' port='5432'" \
  "${rjukan_aerial_burn_tif}"



