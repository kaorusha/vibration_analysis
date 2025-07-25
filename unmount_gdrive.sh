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
        # delete the mount point directory after unmounting
        if [ -d "$MOUNT_POINT" ]; then
        # Check if the directory is empty before removing
            if [ -z "$(ls -A "$MOUNT_POINT" 2>/dev/null)"]; then
                echo "Mount point directory $MOUNT_POINT is empty. Removing..."
                rmdir "$MOUNT_POINT"
                if [ $? -eq 0 ]; then
                    echo "Success: Mount point directory $MOUNT_POINT removed."
                else
                    echo "Warning: Failed to remove mount point directory $MOUNT_POINT. Please check permissions."
                fi
            else
                echo "Mount point directory $MOUNT_POINT is not empty after unmount."
                echo "Please manually remove it if desired."
                echo "Contents: $(ls -A "$MOUNT_POINT" 2>/dev/null)"
            fi
        else
            echo "Warning: Mount point directory $MOUNT_POINT does not exist."
        fi
    else
        echo "Failed to unmount. Trying lazy unmount..."
        fusermount -uz "$MOUNT_POINT"
        if [ $? -eq 0 ]; then
            echo "Successfully lazy unmounted."
            # delete the mount point directory after unmounting
            if [ -d "$MOUNT_POINT" ]; then
            # Check if the directory is empty before removing
                if [ -z "$(ls -A "$MOUNT_POINT" 2>/dev/null)"]; then
                    echo "Mount point directory $MOUNT_POINT is empty. Removing..."
                    rmdir "$MOUNT_POINT"
                    if [ $? -eq 0 ]; then
                        echo "Success: Mount point directory $MOUNT_POINT removed."
                    else
                        echo "Warning: Failed to remove mount point directory $MOUNT_POINT. Please check permissions."
                    fi
                else
                    echo "Mount point directory $MOUNT_POINT is not empty after unmount."
                    echo "Please manually remove it if desired."
                    echo "Contents: $(ls -A "$MOUNT_POINT" 2>/dev/null)"
                fi
            else
                echo "Warning: Mount point directory $MOUNT_POINT does not exist."
            fi
        else
            echo "Failed to unmount Google Drive."
            exit 1
        fi
    fi
else
    echo "$MOUNT_POINT is not mounted."
fi