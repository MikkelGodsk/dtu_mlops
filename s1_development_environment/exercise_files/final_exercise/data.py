import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    train = torch.utils.data.ConcatDataset([
        torch.utils.data.TensorDataset(
            torch.tensor(train_raw['images']),
            torch.tensor(train_raw['labels']))
        for train_raw in [np.load("../../../data/corruptmnist/train_{:d}.npz".format(i)) for i in range(5)]
    ])
    test_raw = np.load("../../../data/corruptmnist/test.npz")
    test = torch.utils.data.TensorDataset(
        torch.tensor(test_raw['images']), 
        torch.tensor(test_raw['labels'])
    )
    return train, test
