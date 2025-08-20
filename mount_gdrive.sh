# Problems with the mounting approach:

# 1.  **File lock conflicts**: MLflow and rclone operate on the same filesystem simultaneously.
# 2.  **Performance issues**: Writing data via the mount point is slow.
# 3.  **Stability problems**: Network interruptions can affect local operations.
# 4.  **UI Read anomalies**:
#       * **Filesystem inconsistencies**: Mismatches between the mounted and local filesystems.
#       * **Metadata desynchronization**: File timestamps and permissions change during the rclone sync process.

# Solution: Secure Synchronization After Training
pkill -f "mlflow"  # Stop MLflow
rclone config reconnect google_drive_mlflow:mlruns  # Refresh credentials
rclone sync ./mlruns google_drive_mlflow:mlruns --progress  # Sync to the cloud
mlflow ui --backend-store-uri ./mlruns  # Restart the UI