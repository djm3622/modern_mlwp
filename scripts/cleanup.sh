#!/bin/bash

# Check if the user provided directories as arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <directory1> [directory2] ..."
    exit 1
fi

# Loop through all provided directories and delete them if they exist
for DIR in "$@"; do
    if [ -d "$DIR" ]; then
        echo "Deleting directory: $DIR"
        rm -rf "$DIR"
    else
        echo "Directory not found: $DIR"
    fi
done

echo "Cleanup complete!"