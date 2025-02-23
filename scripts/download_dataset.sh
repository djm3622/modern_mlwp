#!/bin/bash

# Define base paths for different resolutions
BASE_PATH_64="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
BASE_PATH_240="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
BASE_PATH_1400="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-1440x721_equiangular_with_poles_conservative.zarr"

# Determine output path based on degree argument
case "$1" in
    "5.625") OUTPUT_PATH="$PWD/ERA5/5.625deg"; BASE_PATH="$BASE_PATH_64";;
    "1.5") OUTPUT_PATH="$PWD/ERA5/1.5deg"; BASE_PATH="$BASE_PATH_240";;
    "0.25") OUTPUT_PATH="$PWD/ERA5/0.25deg"; BASE_PATH="$BASE_PATH_1400";;
    *) echo "Invalid degree specified. Use 5.625, 1.5, or 0.25."; exit 1;;
esac

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Define the list of files to copy
FILES=(
    ".zattrs"
    ".zgroup"
    ".zmetadata"
    "10m_u_component_of_wind"
    "10m_v_component_of_wind"
    "2m_temperature"
    "mean_sea_level_pressure"
    "surface_pressure"
    "temperature"
    "land_sea_mask"
    "time"
    "u_component_of_wind"
    "v_component_of_wind"
    "vertical_velocity"
    "level"
    "specific_humidity"
    "geopotential"
    "latitude"
    "longitude"
    "geopotential_at_surface"
    "total_precipitation_6hr"
    "total_column_water"
    "standard_deviation_of_orography"
    "slope_of_sub_gridscale_orography"
)

# Download dataset
gsutil -m cp -r ${FILES[@]/#/${BASE_PATH}/} "$OUTPUT_PATH"

echo "Download complete!"
