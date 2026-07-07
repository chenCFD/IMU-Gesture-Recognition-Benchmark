import os
import shutil
import random
import argparse

def main():
    # ================= 1. Argument Parsing =================
    parser = argparse.ArgumentParser(description="Universally split IMU datasets into Train, Test, and Validation sets.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Suffix of the dataset directories to process (e.g., imu1, imu2, imu3)"
    )
    args = parser.parse_args()
    
    dataset_suffix = args.dataset

    os.makedirs("splited", exist_ok=True)

    # ================= 2. Directory Configuration =================
    # Dynamic input directories based on the --dataset argument
    feat_dir = f"not_split/feat_{dataset_suffix}"
    tar_dir = f"not_split/tar_{dataset_suffix}"

    # Dynamic output directories mapping
    out_dirs = {
        "train_feat": f"splited/train_{dataset_suffix}/feat",
        "train_tar": f"splited/train_{dataset_suffix}/tar",
        "test_feat": f"splited/test_{dataset_suffix}/feat",
        "test_tar": f"splited/test_{dataset_suffix}/tar",
        "valid_feat": f"splited/valid_{dataset_suffix}/feat",
        "valid_tar": f"splited/valid_{dataset_suffix}/tar",
    }

    # Verify if source directories exist
    if not os.path.exists(feat_dir) or not os.path.exists(tar_dir):
        print(f"[Error] Source directories '{feat_dir}' or '{tar_dir}' do not exist. Please check your --dataset input.")
        return

    # Create output directories if they don't exist
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # ================= 3. File Collection & Sorting =================
    # Collect all .npy feature files
    feat_files = [f for f in os.listdir(feat_dir) if f.endswith(".npy")]

    if not feat_files:
        print(f"[Warning] No .npy files found in '{feat_dir}'.")
        return

    # Helper function to extract index from filename for alignment
    def get_idx(name):
        try:
            return int(name.split("_")[-1].split(".")[0])
        except ValueError:
            # Fallback to string sorting if the suffix is not purely numeric
            return name

    # Sort files to guarantee feature and target alignment before shuffling
    feat_files.sort(key=get_idx)

    # ================= 4. Shuffle & Ratio Splitting =================
    # Fix the random seed for reproducibility
    random.seed(42)  
    random.shuffle(feat_files)

    total = len(feat_files)
    
    # Calculate split indices using 80% / 10% / 10% proportions
    valid_end = int(total * 0.1)
    test_end = valid_end + int(total * 0.1)

    valid_files = feat_files[:valid_end]
    test_files = feat_files[valid_end:test_end]
    train_files = feat_files[test_end:]

    print(f"  Dataset Split Summary for [{dataset_suffix}]:")
    print(f"  Total files: {total}")
    print(f"  Train set  : {len(train_files)} ({len(train_files)/total:.1%})")
    print(f"  Test set   : {len(test_files)} ({len(test_files)/total:.1%})")
    print(f"  Valid set  : {len(valid_files)} ({len(valid_files)/total:.1%})")
    print("-" * 40)

    # ================= 5. File Copy Function =================
    def copy_split_files(file_list, feat_out, tar_out):
        for f in file_list:
            # Reconstruct the corresponding target filename
            # Assumes feature filename format contains "feat_"
            feat_src = os.path.join(feat_dir, f)
            tar_name = "tar_" + f.split("feat_")[1]
            tar_src = os.path.join(tar_dir, tar_name)

            feat_dst = os.path.join(feat_out, f)
            tar_dst = os.path.join(tar_out, tar_name)

            # Safeguard to ensure matching target file exists before copying
            if os.path.exists(tar_src):
                shutil.copy(feat_src, feat_dst)
                shutil.copy(tar_src, tar_dst)
            else:
                print(f"[Warning] Target file missing for feature: {f}, skipped.")

    # ================= 6. Execution =================
    print("Copying files to split directories...")
    copy_split_files(train_files, out_dirs["train_feat"], out_dirs["train_tar"])
    copy_split_files(test_files, out_dirs["test_feat"], out_dirs["test_tar"])
    copy_split_files(valid_files, out_dirs["valid_feat"], out_dirs["valid_tar"])

    print("Dataset splitting completed successfully!")


if __name__ == "__main__":
    main()