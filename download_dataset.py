import kagglehub
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["KAGGLEHUB_CACHE"] = current_dir

# 1 IMU dataset
path = kagglehub.dataset_download("dilharajayawardhane/6-axis-motion-gesture-dataset-hand-waves-and-flicks")
print("Path to dataset 1 files:", path)

# 2 IMU dataset
path = kagglehub.dataset_download("suveenellawela/hand-gesture-classification-2-imu-glove")
print("Path to dataset 2 files:", path)

# 3 IMU dataset
path = kagglehub.dataset_download("harrisonlou/imu-glove")
print("Path to dataset 3 files:", path)