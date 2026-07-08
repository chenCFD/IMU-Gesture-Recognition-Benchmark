# IMU-Hand-Gesture-Recognition-Benchmark

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

PyTorch benchmark for **IMU-based Hand Gesture Recognition (HGR)**. 

This provides a standardized pipeline to compare classic open-source models and the latest academic State-of-the-Art (SOTA) architectures (2025/2026) across multiple public datasets.

---

## Supported Datasets

We currently support 3 public Kaggle datasets. The pipeline standardizes them into `(Batch, Timestamp, Channel)` inputs and `(Classes)` output.

| Dataset ID | Source | # IMUs | Input Shape `(T, C)` | Output Classes | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **`imu1`** | [Kaggle Link](https://www.kaggle.com/datasets/dilharajayawardhane/6-axis-motion-gesture-dataset-hand-waves-and-flicks) | 1 | `(50, 6)` | 5 | 6-Axis Motion Gesture Dataset: Hand Waves & Flicks |
| **`imu2`** | [Kaggle Link](https://www.kaggle.com/datasets/suveenellawela/hand-gesture-classification-2-imu-glove) | 2 | `(24, 12)` | 7 | Hand Gesture Classification 2-IMU Glove |
| **`imu3`** | [Kaggle Link](https://www.kaggle.com/datasets/harrisonlou/imu-glove) | 3 | `(90, 18)` | 8 | Hand Gesture Classification 3-IMU Glove |

---

## Model Zoo

This benchmark unifies unofficial implementations from diverse sources into a single `train.py` interface.

**Classic & Open-Source Architectures:**
* `CNN` (based on [Dilharajay/gesture-cmd](https://github.com/Dilharajay/gesture-cmd))
* `LSTM` (based on [IyanekiB/Smart-Glove...](https://github.com/IyanekiB/Smart-Glove-Gesture-Recognition-Using-ML))
* `CRNN` (based on [TemryL/EyeRub-Det](https://github.com/TemryL/EyeRub-Det))
* `RCNN` (based on [dhruba0/gesture_recognition_imu](https://github.com/dhruba0/gesture_recognition_imu))

**Recent Academic SOTA (unofficial implementation):**
* `TF_Fusion_2025`: Dual IMU-Based HGR With Time–Frequency Feature Fusion (IEEE Sensors 2025).
* `Huawei_Transformer_2026`: OpenWatch Multimodal Benchmark Architecture (arXiv 2026).

---

## Getting Started

Follow these steps to reproduce the benchmarks. 

### Step 0: Install Environment
```bash
pip install requirements.txt
```

### Step 1: Download Datasets
Fetch the required public datasets automatically from Kaggle.
```bash
python download_dataset.py
```

### Step 2: Build & Split Datasets
Transform the raw data into standardized .pt tensors, then split them into Training, Validation, and Testing sets.
```bash
# Example for IMU3:
python make_dataset_imu3.py
python split_dataset.py --dataset imu3

# You can repeat this for imu1 and imu2
# python make_dataset_imu1.py && python split_dataset.py --dataset imu1
```

### Step 3: Train & Benchmark
Train any model on any dataset using the unified entry point. The script will automatically handle dimension matching based on the dataset selected.
```bash
# Syntax: python train.py --model [MODEL_NAME] --dataset [DATASET_ID]

# Example: Train the 2026 Transformer on the 3-IMU dataset
python train.py --model Huawei_Transformer_2026 --dataset imu3

# Example: Compare it with a classic CRNN on the 3-IMU dataset
python train.py --model CRNN --dataset imu3
```

## References & Acknowledgements
This framework builds upon the excellent data collection and architectural designs from the open-source community. If you use this pipeline, please consider citing the original dataset authors, Github open source repo and the respective papers:

Dataset:
1 IMU dataset: [6-axis-motion-gesture-dataset-hand-waves-and-flicks](https://www.kaggle.com/datasets/dilharajayawardhane/6-axis-motion-gesture-dataset-hand-waves-and-flicks)
2 IMU dataset: [hand-gesture-classification-2-imu-glove](https://www.kaggle.com/datasets/suveenellawela/hand-gesture-classification-2-imu-glove)
3 IMU dataset: [3-imu-glove](https://www.kaggle.com/datasets/harrisonlou/imu-glove)

Academic Papers Unofficial Implementation in this Paper:
```bash
@misc{bonazzi2026openwatchmultimodalbenchmarkhand,
      title={OpenWatch: A Multimodal Benchmark for Hand Gesture Recognition on Smartwatches}, 
      author={Pietro Bonazzi and Youssef Ahmed and Daniel Eckert and Andrea Ronco and Junjie Zeng and Dengxin Dai and Michele Magno},
      year={2026},
      eprint={2605.04791},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={[https://arxiv.org/abs/2605.04791](https://arxiv.org/abs/2605.04791)}, 
}
```

```bash
@ARTICLE{11142918,
  author={Liu, Yizhen and Meng, Zhaozong and Ni, Yubo and Jia, Shikui and Gao, Nan and Zhang, Zonghua},
  journal={IEEE Sensors Journal}, 
  title={Dual IMU-Based Hand Gesture Recognition With Time–Frequency Feature Fusion}, 
  year={2025},
  volume={25},
  number={19},
  pages={37244-37254},
  doi={10.1109/JSEN.2025.3601375}
}
```
