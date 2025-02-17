#!/bin/bash

# Define base paths for different resolutions
BASE_PATH_64="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
BASE_PATH_240="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

# Define output paths
OUTPUT_PATH_64="$1/5.625deg"
OUTPUT_PATH_240="$1/1.5deg"

# Create directories for output
mkdir -p "${OUTPUT_PATH_64}"
mkdir -p "${OUTPUT_PATH_240}"

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

# Function to download dataset
copy_dataset() {
    local BASE_PATH="$1"
    local OUTPUT_PATH="$2"
    
    gsutil -m cp -r ${FILES[@]/#/${BASE_PATH}/} "$OUTPUT_PATH"
}

# Download both resolutions
copy_dataset "$BASE_PATH_64" "$OUTPUT_PATH_64"
copy_dataset "$BASE_PATH_240" "$OUTPUT_PATH_240"

echo "Download complete!"
