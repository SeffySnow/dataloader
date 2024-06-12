import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-1].values  # All columns except the last one
        self.labels = self.data.iloc[:, -1].values  # The last column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, labels

# Example usage:
# train_dataset = CustomDataset('path/to/train.csv')
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# for features, labels in train_loader:
#     print(features, labels)
