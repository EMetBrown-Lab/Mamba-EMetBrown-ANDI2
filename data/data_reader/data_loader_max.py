# Import necessary libraries
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass


# Function to pad an array to a specific shape
def to_shape(a, shape):
    # Unpack the target shape
    y_, x_ = shape

    # Get the current shape of the array
    y, x = a.shape

    # Calculate the padding needed in the y and x directions
    y_pad = y_ - y
    x_pad = x_ - x

    # Pad the array using numpy's pad function
    return np.pad(
        a,
        # Calculate the padding for each dimension
        ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )


# Function to pad data and labels to a specific shape
def apply_padding(data_df, N, T_max):
    # Define the final shape of the data and labels
    final_shape = (N, 3, T_max)

    # Initialize the final data and labels with zeros
    final_data = np.zeros(final_shape)
    final_label = np.zeros((N, 2, T_max))

    # Select a random subset of trajectory indices
    selected_ids = np.random.choice(data_df["traj_idx"].unique(), size=N, replace=False)

    # Iterate over the selected trajectory indices
    for n, id in enumerate(selected_ids):
        # Filter the data for the current trajectory index
        exp = data_df[data_df["traj_idx"] == id]

        # Extract the data and labels for the current trajectory
        data = exp[["frame", "x", "y"]].to_numpy().T
        label = exp[["alpha", "D"]].to_numpy().T

        # If the data is longer than T_max, truncate it
        if data.shape[1] > T_max:
            final_data[n, :, :] = data[:, :T_max]
            final_label[n, :, :] = label[:, :T_max]
        # Otherwise, pad the data to T_max
        else:
            final_data[n, :, :] = to_shape(data, (3, T_max))
            final_label[n, :, :] = to_shape(label, (2, T_max))

    # Return the padded data and labels
    return final_data, final_label


# Define a function to normalize data
def normalize_df(data):
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


def normalize_np(data):

    displacement_x = []
    displacement_y = []
    for n in range(data.shape[0]):
        x = data[n,1,:]
        y = data[n,2,:]
        d_x = x[1:] - x[:-1]
        d_y = y[1:] - y[:-1]
        displacement_x = displacement_x + list(d_x)
        displacement_y = displacement_y + list(d_y)


    # Calculate variance in x and y directions
    variance_x = np.sqrt(np.std(displacement_x))
    variance_y = np.sqrt(np.std(displacement_y))

    # Normalize data

    data[:,1,:] = (data[:,1,:] - np.mean(data[:,1,:])) / variance_x
    data[:,2,:] = (data[:,2,:] - np.mean(data[:,2,:])) / variance_x


    return data

    


# Define a function to list directory tree with pathlib
def list_directory_tree_with_pathlib(starting_directory):
    path_object = Path(starting_directory)
    folders = []
    for file_path in path_object.rglob("*.csv"):
        folders.append(file_path)
    return folders


# Define a custom dataset class for all data
@dataclass
class Dataset_all_data(Dataset):
    # Initialize filenames and transform flag
    # Pad value should be a tuple such as (N, Tmax)
    filenames: list 
    transform: bool = False
    pad: None | tuple = None

    def __len__(self):
        # Return the number of files
        return len(self.filenames)

    def __getitem__(self, idx):
        # Read csv file and extract data and label
        df = pd.read_csv(self.filenames[idx])
        

        if self.pad is None:
            data = df[["traj_idx", "frame", "x", "y"]]
            label = np.asarray(df[["alpha", "D"]])

        else:
            if len(self.pad) !=2:
                raise ValueError("pad value should be set as (N, T_max)")
            data, label = apply_padding(df, *self.pad)

        # Normalize data if transform flag is True
        if self.transform:
            if self.pad is None:
                normalize_df(data)
                data = np.asarray(data)
            else:
                data = normalize_np(data)

        
        # Return data and label
        return data, label


# Define a custom dataset class for separating trajectories
@dataclass
class Dataset_separating_trajs(Dataset):
    # Initialize filenames
    # Pad value should be a tuple such as (N, Tmax)
    filenames: list
    transform: bool = False
    pad: None | tuple = None

    def __len__(self):
        # Return the number of files
        return len(self.filenames)

    def __getitem__(self, idx):
        # Read csv file and extract data and label
        df = pd.read_csv(self.filenames[idx])
        datas = []
        labels = []
        # Group data by trajectory index

        if self.pad is None:
            if self.transform:
                normalize_df(df)

            for _, id in df.groupby("traj_idx"):
                data = np.asarray(id[["frame", "x", "y"]])
                label = np.asarray(id[["alpha", "D"]])

                # Reshape data and label
                datas.append(data.reshape((-1, *data.shape)))
                labels.append(label.reshape((-1, *label.shape)))

            # Return data and label
            return datas, labels
        else:
            data, label = apply_padding(df)
            
            if self.transform:
                data = normalize_np(data)
            


            for i in self.pad[0]:
                datas.append(data[i,:,:])
                labels.append(label[i,:,:])
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
        all_datasets[: int((len(all_datasets)) * 0.80)],
        transform=True,
    )
    test_dataset = Dataset_separating_trajs(
        all_datasets[int((len(all_datasets)) * 0.80 + 1) :], transform=True
    )

    # Iterate over test dataset and print batches
    i = 0
    for batch in iter(test_dataset):
        if i == 0:
            print(batch)
