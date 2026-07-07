import os
import numpy as np
import pandas as pd


csv_path = "../datasets/dilharajayawardhane/6-axis-motion-gesture-dataset-hand-waves-and-flicks/versions/1/gesture_dataset.csv"

os.makedirs("not_split", exist_ok=True)

feat_dir = "./not_split/feat_imu1"
tar_dir = "./not_split/tar_imu1"

os.makedirs(feat_dir, exist_ok=True)
os.makedirs(tar_dir, exist_ok=True)

label_map = {
    "wave_right": 0,
    "wave_left": 1,
    "flick_up": 2,
    "flick_down": 3,
    "idle": 4
}

num_classes = 5


df = pd.read_csv(csv_path)


groups = df.groupby(["label", "sample_id"])

counter = 0

for (label_str, sample_id), group in groups:

    
    if len(group) != 50:
        print(f"skip {label_str}-{sample_id}, len={len(group)}")
        continue

    # ====== feature ======
    feat = group[["ax", "ay", "az", "gx", "gy", "gz"]].values.astype(np.float32)

    # ====== label ======
    if label_str not in label_map:
        print(f"unknown label {label_str}")
        continue

    target = np.zeros(num_classes, dtype=np.float32)
    target[label_map[label_str]] = 1.0

    feat_filename = f"feat_{label_str}_{counter}.npy"
    tar_filename = f"tar_{label_str}_{counter}.npy"

    np.save(os.path.join(feat_dir, feat_filename), feat)
    np.save(os.path.join(tar_dir, tar_filename), target)

    counter += 1

print(f"Done! total samples: {counter}")