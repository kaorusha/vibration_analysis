#!/bin/bash

# Usage: ./mount_gdrive.sh <mount_point>
# Example: ./mount_gdrive.sh ./gdrive
GDRIVE_REMOTE="google_drive_mlflow"
MLFLOW_SUBDIR="mlruns"
MOUNT_POINT="$1" # local mount point for Google Drive

if [ -z "$MOUNT_POINT" ]; then
    echo "Usage: $0 <mount_point>"
    exit 1
fi

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "rclone is not installed. Installing..."
    sudo apt update && sudo apt install -y rclone
fi

# Configure rclone if not already configured
if ! rclone listremotes | grep -q $GDRIVE_REMOTE; then
    echo "No $GDRIVE_REMOTE remote found. Please run 'rclone config' to set up Google Drive."
    exit 1
fi

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Mount point $MOUNT_POINT not exist, creating..."
    mkdir -p "$MOUNT_POINT"
    if [ $? -ne 0 ]; then
        echo "Failed to create mount point $MOUNT_POINT. Please check permissions."
        exit 1
    fi
fi

# Check if the mount point is already mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo "Mount point $MOUNT_POINT is already mounted."
    exit 0
fi

# Mount Google Drive using rclone
echo "Mounting Google Drive to $MOUNT_POINT ..."
rclone mount "$GDRIVE_REMOTE:$MLFLOW_SUBDIR" "$MOUNT_POINT" \
    --vfs-cache-mode writes \
    --allow-other \
    --daemon
# Note: Uncomment /etc/fuse.conf to allow other users to access the mount

if [ $? -eq 0 ]; then
    echo "Google Drive mounted at $MOUNT_POINT successfully."
    echo "You can now set MLflow tracking URI to file://$MOUNT_POINT"
else
    echo "Failed to mount Google Drive. Please check the rclone configuration or network connection."
    exit 1
fi