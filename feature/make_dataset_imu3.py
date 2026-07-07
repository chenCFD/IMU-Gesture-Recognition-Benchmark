import os
import numpy as np
import pandas as pd
import glob

# ================= Configuration =================
DATA_DIR = '../datasets/harrisonlou/imu-glove/versions/1/rosbag/data/data_clean'
LABEL_DIR = '../datasets/harrisonlou/imu-glove/versions/1/rosbag/data/label'

os.makedirs("not_split", exist_ok=True)

FEAT_DIR = './not_split/feat_imu3'
TAR_DIR = './not_split/tar_imu3'

# Create output directories if they don't exist
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(TAR_DIR, exist_ok=True)

# Label Mapping (Value -> 8-dimensional One-hot vector)
LABEL_MAP = {
    0:  [1, 0, 0, 0, 0, 0, 0, 0],
    1:  [0, 1, 0, 0, 0, 0, 0, 0],
    2:  [0, 0, 1, 0, 0, 0, 0, 0],
    3:  [0, 0, 0, 1, 0, 0, 0, 0],
    4:  [0, 0, 0, 0, 1, 0, 0, 0],
    5:  [0, 0, 0, 0, 0, 1, 0, 0],
    6:  [0, 0, 0, 0, 0, 0, 1, 0],
    10: [0, 0, 0, 0, 0, 0, 0, 1]
}


def main():
    # Retrieve all CSV files within the data_clean directory
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    
    if not data_files:
        print(f"[Error] No CSV files found in {DATA_DIR}.")
        return

    processed_count = 0
    skipped_count = 0

    for data_path in data_files:
        # Generate base filename by removing extension and '_data' suffix
        raw_filename = os.path.basename(data_path)
        base_name = raw_filename.replace('_data.csv', '')
        label_path = os.path.join(LABEL_DIR, f"{base_name}_label.csv")

        # Verify if the corresponding label file exists
        if not os.path.exists(label_path):
            print(f"[Skip] Corresponding label file not found: {label_path}")
            continue

        try:
            # --- 1. Feature Processing ---
            # Read feature data (assuming header is at row 0)
            df_data = pd.read_csv(data_path)
            
            # Validation: sequence must contain at least 90 rows of data
            if len(df_data) < 90:
                print(f"[Skip] Insufficient rows (< 90): {raw_filename} (Contains only {len(df_data)} rows)")
                skipped_count += 1
                continue

            # Column Selection Configuration based on sensor layout:
            # A(0): timestamp, B-G(1-6): Imu0, L-Q(11-16): Imu1, V-AA(21-26): Imu2
            # Select the first 90 rows (iloc 0:90)
            cols = list(range(1, 7)) + list(range(11, 17)) + list(range(21, 27))
            feat_data = df_data.iloc[0:90, cols].values.astype(np.float32)

            # Ensure the output shape matches exactly (90, 18)
            if feat_data.shape != (90, 18):
                print(f"[Warning] Invalid shape dimensions for {base_name}: {feat_data.shape}")
                continue

            # --- 2. Label Processing ---
            # Read the label file
            df_label = pd.read_csv(label_path)
            
            if df_label.empty:
                print(f"[Skip] Label file is empty: {label_path}")
                continue

            # Extract the ground truth label from the first row and first column
            label_value = df_label.iloc[0, 0]

            # Convert to integer and validate against the map keys
            label_key = int(label_value)
            if label_key not in LABEL_MAP:
                print(f"[Warning] Unknown label value '{label_key}' found in file: {os.path.basename(label_path)}")
                continue
            
            tar_data = np.array(LABEL_MAP[label_key], dtype=np.float32)

            # --- 3. Save Files ---
            feat_output_path = os.path.join(FEAT_DIR, f"feat_{base_name}.npy")
            tar_output_path = os.path.join(TAR_DIR, f"tar_{base_name}.npy")

            np.save(feat_output_path, feat_data)
            np.save(tar_output_path, tar_data)

            processed_count += 1
            print(f"Successfully processed: {base_name} | Label: {label_key}")

        except Exception as e:
            print(f"[Error] Exception occurred while processing {base_name}: {e}")

    print("-" * 30)
    print(f"Processing complete! Successfully converted {processed_count} file groups. Skipped {skipped_count} files due to insufficient data.")


if __name__ == "__main__":
    main()