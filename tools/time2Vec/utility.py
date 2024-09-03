from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AbstractPipelineClass:
    def __init__(self, model=None):
        if model:
            self.model = model
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def preprocess(self, x):
        raise NotImplementedError

    def predict(self, x):
        preprocessed = self.preprocess(x)
        return self.decorate_output(self.model(preprocessed))

    def decorate_output(self):
        raise NotImplementedError

class NextDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]

        # print(dates)

    def __len__(self):
        return len(self.dates) - 1

    def __getitem__(self, idx):
        return np.array(self.dates[idx]).astype(np.float32), np.array(self.dates[idx + 1]).astype(np.float32)


class TimeDateDataset(Dataset):
    def __init__(self, dates):
        dates = [date.split(",") for date in dates]

        convert_int = lambda dt: list(map(int, dt))
        self.dates = [convert_int(date) for date in dates]

        # print(dates)

    def __len__(self):
        return len(self.dates) - 1

    def __getitem__(self, idx):
        x = np.array(self.dates[idx]).astype(np.float32)
        return x, x

