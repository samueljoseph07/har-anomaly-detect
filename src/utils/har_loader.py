import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_har_dataset(data_dir="data/HAR", normalize=True):
    signal_types = ["body_acc_x","body_acc_y","body_acc_z",
                    "body_gyro_x","body_gyro_y","body_gyro_z",
                    "total_acc_x","total_acc_y","total_acc_z"]
    signals = []
    for signal in signal_types:
        path = os.path.join(data_dir, "train", "Inertial Signals", f"{signal}_train.txt")
        data = np.loadtxt(path)
        signals.append(data)
    stacked = np.stack(signals, axis=-1)  # [samples, 128, 9]
    stream = stacked.reshape(-1, stacked.shape[2])
    if normalize:
        stream = StandardScaler().fit_transform(stream)
    return stream