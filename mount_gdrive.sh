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

# Check if the mount point is already mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo "Mount point $MOUNT_POINT is already mounted."
    exit 0
fi

# Configure rclone if not already configured
if ! rclone listremotes | grep -q $GDRIVE_REMOTE; then
    echo "No rcclone remote named $GDRIVE_REMOTE found."
    echo "Please run 'rclone config' to set up Google Drive."
    exit 1
fi

# --- Pre-mount checks ---

# Ensure the mount point does not exist before creating it, or is empty
if [ -d "$MOUNT_POINT" ] && [ "$(ls -A $MOUNT_POINT)" ]; then
    echo "Mount point $MOUNT_POINT already exists and is not empty."
    echo "Please use an empty directory to avoid conflicts."
    exit 1
else 
    echo "Creating mount point: '$MOUNT_POINT'..."
    mkdir -p "$MOUNT_POINT"
    if [ $? -ne 0 ]; then
        echo "Failed to create mount point $MOUNT_POINT. Please check permissions."
        exit 1
    fi
fi

# Mount Google Drive using rclone
echo "Mounting Google Drive '$GDRIVE_REMOTE:$MLFLOW_SUBDIR' to '$MOUNT_POINT' ..."
rclone mount "$GDRIVE_REMOTE:$MLFLOW_SUBDIR" "$MOUNT_POINT" \
    --vfs-cache-mode writes \
    --allow-other \
    --daemon \
    --log-level INFO 
# Note: Uncomment user_allow_other in the file /etc/fuse.conf to allow other users to access the mount

if [ $? -eq 0 ]; then
    echo "Google Drive mounted at $MOUNT_POINT successfully."
    echo "You can now set MLflow tracking URI to file://$MOUNT_POINT"
else
    echo "Failed to mount Google Drive. Please check the rclone configuration or network connection."
    exit 1
fi