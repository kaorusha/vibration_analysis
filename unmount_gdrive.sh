#!/bin/bash

# Usage: ./unmount_gdrive.sh <mount_point>
# Example: ./unmount_gdrive.sh ./gdrive
if [ -z "$1" ]; then
    echo "Usage: $0 <mount_point>"
    exit 1
fi
MOUNT_POINT="$1" # local mount point for Google Drive

# Check if the mount point is actually mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo "Unmounting Google Drive from $MOUNT_POINT..."
    fusermount -u "$MOUNT_POINT"
    if [ $? -eq 0 ]; then
        echo "Successfully unmounted."
    else
        echo "Failed to unmount. Trying lazy unmount..."
        fusermount -uz "$MOUNT_POINT"
        if [ $? -eq 0 ]; then
            echo "Successfully lazy unmounted."
        else
            echo "Failed to unmount Google Drive."
            exit 1
        fi
    fi
else
    echo "$MOUNT_POINT is not mounted."
fi