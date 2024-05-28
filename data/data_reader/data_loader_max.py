from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np


def list_directory_tree_with_pathlib(starting_directory):
    path_object = Path(starting_directory)
    folders = []
    for file_path in path_object.rglob("*.csv"):
        folders.append(file_path)
    return folders


class Dataset_all_data(Dataset):
    def __init__(self, filenames):
        # Filenames : List of csv files.

        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        df = pd.read_csv(self.filenames[idx])
        data = np.asarray(df[["traj_idx", "frame", "x", "y"]])
        label = np.asarray(df[["alpha", "D"]])

        if idx == self.__len__():
            return IndexError

        return data, label


class Dataset_separating_trajs(Dataset):
    def __init__(self, filenames):
        # Filenames : List of csv files.

        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        df = pd.read_csv(self.filenames[idx])

        datas = []
        labels = []
        ## Now we will create a list with all the different trajectories
        for _, id in df.groupby("traj_idx"):

            data = np.asarray(id[["frame", "x", "y"]])
            label = np.asarray(id[["alpha", "D"]])

            datas.append(data.reshape((-1, *data.shape)))
            labels.append(label.reshape((-1, *label.shape)))

        if idx == self.__len__():
            return IndexError

        return datas, labels


if __name__ == "__main__":

    all_datasets = list_directory_tree_with_pathlib("/home/m.lavaud/ANDI_2_Challenge_EMetBrown/data/data/small_batch_1/1/1/")[:20]
    dataset_1 = Dataset_all_data(all_datasets)
    dataset_2 = Dataset_separating_trajs(all_datasets)

    from torch.utils.data import DataLoader

    import random

    ## Now we can create a test and train dataloader :

    random.shuffle(all_datasets)

    train_dataset = Dataset_all_data(all_datasets[: int((len(all_datasets)) * 0.80)])
    test_dataset = Dataset_all_data(all_datasets[int((len(all_datasets)) * 0.80 + 1):])

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=1,
    )
    test_dataset = DataLoader(
        test_dataset,
        num_workers=1,
    )

    for batch in iter(test_dataset):
        print(batch)

    
    test_dataset = Dataset_separating_trajs(all_datasets[int((len(all_datasets)) * 0.80 + 1):])
    test_dataset = DataLoader(test_dataset,)
    for batch in iter(test_dataset):
        print(batch)
