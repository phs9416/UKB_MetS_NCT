#!/bin/bash

mkdir -p zip_file 
mkdir -p unzip_file
cd zip_file

BASE_DIR="/Bulk/Brain MRI/Connectomes"
PROJECT_ID="FBA-Metabolic"

for i in {10..60}; do
	REMOTE_PATH="$BASE_DIR/$i/*_31024_2_0.zip"
	echo "Downloading from: $PROJECT_ID:$REMOTE_PATH"
	dx download --no-progress --lightweight "$PROJECT_ID:$REMOTE_PATH"
done

cd ..

ZIP_DIR="zip_file" 
UNZIP_PARENT_DIR="unzip_file"  

for ZIP_FILE in "$ZIP_DIR"/*.zip; do
    if [ -f "$ZIP_FILE" ]; then
        UNZIP_DIR="$UNZIP_PARENT_DIR/$(basename "${ZIP_FILE%.zip}")"  
        mkdir -p "$UNZIP_DIR"
        unzip -q "$ZIP_FILE" -d "$UNZIP_DIR"
    fi
done

dx upload --no-progress unzip_file -r --path /Structural_connectomes/
