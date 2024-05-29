# Import necessary libraries
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np


# Define a function to normalize data
def normalize(data):
    # Calculate displacement in x and y directions
    # Normalize by substring mean and dividing by variance.

    displacement_x = []
    displacement_y = []
    for _, group in data.groupby("traj_idx"):
        x = np.asarray(group["x"])
        y = np.asarray(group["y"])
        d_x = x[1:] - x[:-1]
        d_y = y[1:] - y[:-1]
        displacement_x = displacement_x + list(d_x)
        displacement_y = displacement_y + list(d_y)

    # Calculate variance in x and y directions
    variance_x = np.sqrt(np.std(displacement_x))
    variance_y = np.sqrt(np.std(displacement_y))

    # Normalize data
    data.loc[:, "x"] = (data["x"] - data["x"].mean()) / variance_x
    data.loc[:, "y"] = (data["y"] - data["y"].mean()) / variance_y


# Define a function to list directory tree with pathlib
def list_directory_tree_with_pathlib(starting_directory):
    path_object = Path(starting_directory)
    folders = []
    for file_path in path_object.rglob("*.csv"):
        folders.append(file_path)
    return folders


# Define a custom dataset class for all data
class Dataset_all_data(Dataset):
    def __init__(self, filenames, transform=False):
        # Initialize filenames and transform flag
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        # Return the number of files
        return len(self.filenames)

    def __getitem__(self, idx):
        # Read csv file and extract data and label
        df = pd.read_csv(self.filenames[idx])
        data = df[["traj_idx", "frame", "x", "y"]]
        label = np.asarray(df[["alpha", "D"]])

        # Normalize data if transform flag is True
        if self.transform:
            normalize(data)

        data = np.asarray(data)
        # Return data and label
        return data, label


# Define a custom dataset class for separating trajectories
class Dataset_separating_trajs(Dataset):
    def __init__(self, filenames: list, transform: bool = False):
        # Initialize filenames
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        # Return the number of files
        return len(self.filenames)

    def __getitem__(self, idx):
        # Read csv file and extract data and label
        df = pd.read_csv(self.filenames[idx])
        datas = []
        labels = []
        # Group data by trajectory index
        if self.transform:
            normalize(df)

        for _, id in df.groupby("traj_idx"):
            data = np.asarray(id[["frame", "x", "y"]])
            label = np.asarray(id[["alpha", "D"]])

            # Reshape data and label
            datas.append(data.reshape((-1, *data.shape)))
            labels.append(label.reshape((-1, *label.shape)))

        # Return data and label
        return datas, labels


# Main function
if __name__ == "__main__":
    # List all csv files in the directory
    all_datasets = list_directory_tree_with_pathlib(
        "/home/m.lavaud/ANDI_2_Challenge_EMetBrown/data/data/small_batch_1/1/1/"
    )[:20]

    # Create dataset objects
    dataset_1 = Dataset_all_data(all_datasets)
    dataset_2 = Dataset_separating_trajs(all_datasets)

    # Import DataLoader from torch.utils.data
    from torch.utils.data import DataLoader

    # Import random module
    import random

    # Shuffle all datasets
    random.shuffle(all_datasets)

    # Create train and test datasets
    train_dataset = Dataset_all_data(all_datasets[: int((len(all_datasets)) * 0.80)])
    test_dataset = Dataset_all_data(all_datasets[int((len(all_datasets)) * 0.80 + 1) :])

    # Create data loaders for train and test datasets
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=1,
    )

    test_dataset = DataLoader(
        test_dataset,
        num_workers=1,
    )

    # Iterate over test dataset and print batches
    # for batch in iter(test_dataset):
    #     print(batch)

    # Create test dataset for separating trajectories
    train_dataset = Dataset_separating_trajs(
        all_datasets[: int((len(all_datasets)) * 0.80)], transform=True,
    )
    test_dataset = Dataset_separating_trajs(
        all_datasets[int((len(all_datasets)) * 0.80 + 1) :], transform=True
    )

    # Iterate over test dataset and print batches
    i=0
    for batch in iter(test_dataset):
        if i==0:
            print(batch)
        
