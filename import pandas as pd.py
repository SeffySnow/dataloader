import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class PSMDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.features = self.data_frame.iloc[:, :-1].values  # Assuming the last column is the label
        self.labels = self.data_frame.iloc[:, -1].values

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

def get_dataloader(csv_file, batch_size=32, shuffle=True, num_workers=2):
    dataset = PSMDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # Example usage
    train_loader = get_dataloader('path/to/train.csv', batch_size=32, shuffle=True)
    test_loader = get_dataloader('path/to/test.csv', batch_size=32, shuffle=False)

    for features, labels in train_loader:
        print(features, labels)