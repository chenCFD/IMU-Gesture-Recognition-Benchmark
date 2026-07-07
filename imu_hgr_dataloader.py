import numpy as np
import torch
from torch.utils.data import Dataset
import os
from os.path import join


class imu_gesture_dataset(Dataset):
    def __init__(self, path='train'):
        # assert path in ['train', 'test', 'valid']

        self.data_path = path

        # feature path
        self.pathi = join(self.data_path, 'feat')

        # target path
        self.patho = join(self.data_path, 'tar')

        # 掃描檔案
        self.filei = []
        for f in os.listdir(self.pathi):
            if f.endswith('.npy'):
                self.filei.append(f)

        self.filei.sort()  # 保險排序
        self.file_num = len(self.filei)

    def __len__(self):
        return self.file_num

    def __getitem__(self, index):
        feat_name = self.filei[index]

        # load feature
        x = np.load(join(self.pathi, feat_name))

        # load target
        tar_name = feat_name.replace('feat', 'tar')
        y = np.load(join(self.patho, tar_name))

        # ====== 重要：轉 shape（給CNN or LSTM） ======
        # CNN1D 用 (C, T)
        x = x.T  # (50,6) → (6,50)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


if __name__ == "__main__":
    dataset = imu_gesture_dataset(stage='train')

    print("dataset size:", len(dataset))

    x, y = dataset[0]

    print("x shape:", x.shape)  # 預期 (6, 50)
    print("y shape:", y.shape)  # 預期 (5,)
    print("y:", y)

    # 測試 DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_x, batch_y in loader:
        print("batch x:", batch_x.shape)  # (B, 6, 50)
        print("batch y:", batch_y.shape)  # (B, 5)
        break