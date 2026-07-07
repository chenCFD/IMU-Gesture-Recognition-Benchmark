import os
import numpy as np
import pandas as pd
import glob

# Paths configuration
SOURCE_DIR = '../datasets/suveenellawela/hand-gesture-classification-2-imu-glove/versions/3/final_raw_data' 

os.makedirs("not_split", exist_ok=True)

FEAT_DIR = './not_split/feat_imu2'
TAR_DIR = './not_split/tar_imu2'

# Create output directories if they don't exist
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(TAR_DIR, exist_ok=True)


def get_label(filename):
    """
    Parses the filename to return a one-hot encoded label for the gesture.
    """
    name = filename.lower()
    
    if 'left' in name:
        return [1, 0, 0, 0, 0, 0, 0]
    elif 'right' in name:
        return [0, 1, 0, 0, 0, 0, 0]
    elif 'anti' in name: 
        return [0, 0, 0, 1, 0, 0, 0]
    elif 'clockwise' in name: 
        return [0, 0, 1, 0, 0, 0, 0]
    elif 'up' in name:
        return [0, 0, 0, 0, 1, 0, 0]
    elif 'down' in name:
        return [0, 0, 0, 0, 0, 1, 0]
    elif 'shake' in name:
        return [0, 0, 0, 0, 0, 0, 1]
    
    return None


def main():
    # Recursively find all CSV files in the source directory
    csv_files = glob.glob(os.path.join(SOURCE_DIR, '**/*.csv'), recursive=True)
    
    if not csv_files:
        print(f"[Error] No CSV files found in {SOURCE_DIR}. Please check the path.")
        return

    processed_count = 0
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        filename_without_ext = os.path.splitext(filename)[0]

        # Extract label from filename
        label = get_label(filename)
        if label is None:
            print(f"[Warning] Failed to determine gesture label from filename '{filename}'. Skipped.")
            continue

        try:
            df = pd.read_csv(filepath)
            
            # Extract features (e.g., first 24 timesteps, 12 IMU channels)
            data = df.iloc[:24, 1:13].values.astype(np.float32)

            # Zero-padding if the sequence length is less than 24
            if data.shape[0] < 24:
                print(f"[Info] Padding sequence: actual length {data.shape[0]} is less than 24.")
                pad_width = 24 - data.shape[0]
                data = np.pad(data, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)

            # Define saving paths
            feat_filename = f"feat_{filename_without_ext}.npy"
            tar_filename  = f"tar_{filename_without_ext}.npy"
            
            feat_path = os.path.join(FEAT_DIR, feat_filename)
            tar_path = os.path.join(TAR_DIR, tar_filename)

            # Save features and targets as numpy arrays
            np.save(feat_path, data)
            np.save(tar_path, np.array(label, dtype=np.float32))

            processed_count += 1
            print(f"Successfully exported: {feat_filename} (Shape: {data.shape}) | Label: {label}")

        except Exception as e:
            print(f"[Error] Exception occurred while processing file '{filename}': {e}")

    print("-" * 30)
    print(f"Dataset preprocessing completed! Successfully converted {processed_count} files.")
    print(f"Features saved to: {os.path.abspath(FEAT_DIR)}")
    print(f"Targets saved to: {os.path.abspath(TAR_DIR)}")


if __name__ == "__main__":
    main()