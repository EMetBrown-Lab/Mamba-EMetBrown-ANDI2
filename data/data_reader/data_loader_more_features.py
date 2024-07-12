import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Function to pad an array to a specific shape
def to_shape(a, shape):
    # Unpack the target shape
    y_, x_ = shape

    # Get the current shape of the array
    y, x = a.shape

    # Calculate the padding needed in the y and x directions
    y_pad = y_ - y
    x_pad = x_ - x
    output = np.zeros()
    # Pad the array using numpy's pad function
    return np.pad(
        a,
        [(0, 1), (0, 1)],
        # Calculate the padding for each dimension
        # ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    angle = (ang1 - ang2) % (2 * np.pi)
    if angle > np.pi: angle = angle - 2*np.pi
    return angle



# Function to pad data and labels to a specific shape
def apply_padding(data_df, N, T_max):
    # Define the final shape of the data and labels
    final_shape = (N, T_max-1, 6)

    # Initialize the final data and labels with zeros
    final_data = np.zeros(final_shape)
    final_label = np.zeros((N, T_max-1, 3))

    # Select a random subset of trajectory indices
    if len(data_df["traj_idx"].unique()) < N:
        selected_ids = np.random.choice(
            data_df["traj_idx"].unique(), size=N, replace=True
        )
    else:
        selected_ids = np.random.choice(
            data_df["traj_idx"].unique(), size=N, replace=False
        )

    # Iterate over the selected trajectory indices
    for n, id in enumerate(selected_ids):
        # Filter the data for the current trajectory index
        exp = data_df[data_df["traj_idx"] == id]

        # Extract the data and labels for the current trajectory
        data = exp[["frame", "x", "y"]].to_numpy()  
        data[:, 0] = data[:, 0] - data[0, 0] + 1  # putting first frame rate to 1
        data[:, 1] = data[:, 1] - data[0, 1]  # putting initial position to 0
        data[:, 2] = (
            data[:, 2] - data[0, 2]
        )  # putting initital position to 0        # print(exp["frame"])
        # Displacement
        Dx = data[1:,1] - data[:-1,1]
        Dy = data[1:,2] - data[:-1,2]

        Dx2 = data[2:,1] - data[:-2,1]
        Dy2 = data[2:,1] - data[:-2,1]

        dist_1 = np.sqrt(Dx**2 + Dy**2)
        dist_2 = np.sqrt(Dx2**2 + Dy2**2)
        
        
        MDx = np.zeros(len(Dx))
        MDy = np.zeros(len(Dx))
        MSD = np.zeros(len(Dx))
        mean_dist_1 = np.zeros(len(Dx))
        mean_dist_2 = np.zeros(len(Dx))
        
        angles = np.zeros(len(Dx))
        distance_displacement = np.sqrt(np.power(Dx,2) + np.power(Dy,2))
        #Displacement average

        for i in range(1, len(Dx)+1):
            # MDx[i-1] = np.mean(data[i:,1] - data[:-i,1])
            # MDy[i-1] = np.mean(data[i:,2] - data[:-i,2])
            MSD[i-1] = np.mean((data[i:,2] - data[:-i,2])**2)
            A = (data[(i-1),1], data[(i-1),2])
            B = (data[i,1], data[i,2])
        
            # Computation of angles

            angles[i-1] = angle_between(A,B)

            start= i - 1
            start = max(start, 0)
            end_1 = i + 1 
            end_2 = i + 1
            if end_1 > len(Dx):
                end_1 = len(Dx)
            if end_2 >len(Dx) - 1:
                end_2 = len(Dx) - 1

            mean_dist_1[i-1] = np.mean(mean_dist_1[start:end_1+1])
            mean_dist_2[i-1] = np.mean(mean_dist_2[start:end_2+1])
            
        label = exp[["alpha", "D", "state"]].to_numpy()
        ## adding one to the states
        label[:, 2] = label[:, 2] + 1
        # If the data is longer than T_max, truncate it
        if data.shape[0] > T_max:
            # final_data[n, :, :] = data[:T_max, :]
            final_data[n,:,0] = Dx[:(T_max-1)]
            final_data[n,:,1] = Dy[:(T_max-1)]
            final_data[n,:,2] = mean_dist_1[:(T_max-1)]
            final_data[n,:,3] = mean_dist_2[:(T_max-1)]
            final_data[n,:,4] = distance_displacement[:(T_max-1)]
            final_data[n,:,5] = angles[:(T_max-1)]

            final_label[n, :, :] = label[:T_max-1, :]

        # Otherwise, pad the data to T_max
        else:
            # print((label.shape, T_max))
            final_data[n, : (data.shape[0] -1), 0] = Dx
            final_data[n, : (data.shape[0] -1), 1] = Dy
            final_data[n, : (data.shape[0] -1), 2] = mean_dist_1
            final_data[n, : (data.shape[0] -1), 3] = mean_dist_2
            final_data[n, : (data.shape[0] -1), 4] = distance_displacement
            final_data[n, : (data.shape[0] -1), 5] = angles

            final_label[n, : data.shape[0] -1, :] = label[:-1, :]

    # Return the padded data and label
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
        x = data[n, :, 1]
        y = data[n, :, 2]
        d_x = x[1:] - x[:-1]
        d_y = y[1:] - y[:-1]
        displacement_x = displacement_x + list(d_x)
        displacement_y = displacement_y + list(d_y)

    # Calculate variance in x and y directions
    variance_x = np.sqrt(np.std(displacement_x))
    variance_y = np.sqrt(np.std(displacement_y))

    # Normalize data

    data[:, :, 1] = (data[:, :, 1] - np.mean(data[:, :, 1])) / variance_x
    data[:, :, 2] = (data[:, :, 2] - np.mean(data[:, :, 2])) / variance_x

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
    noise: bool = False

    def __len__(self):
        # Return the number of files
        return len(self.filenames)

    def __getitem__(self, idx):
        # Read csv file and extract data and label
        df = pd.read_csv(self.filenames[idx])

        if self.pad is None:
            data = df[["traj_idx", "frame", "x", "y"]]
            label = np.asarray(df[["alpha", "D"]])
            label_2 = np.asarray(df["state"])

        else:
            if len(self.pad) != 2:
                raise ValueError("pad value should be set as (N, T_max)")
            data, label = apply_padding(df, *self.pad)
            data = data[:, :, :]  ## Removing the frame column
            label_2 = label[:, :, -1]
            label_2[label_2[:, :] > 0] = label_2[label_2[:, :] > 0]
            label = label[:, :, :-1]

        # Normalize data if transform flag is True
        if self.transform:
            if self.pad is None:
                normalize_df(data)
                data = np.asarray(data)
            else:
                data = normalize_np(data)

        if self.noise:
            data = add_noise(data)

        # Normalize D between 0 and 1

        # label[:,:,1][label[:,:,1] != 0] = np.log(label[:,:,1][label[:,:,1] != 0]) #- np.log(1e-6)) #/   (np.log(1e12) - np.log(1e-6))
        # label = label[:,:,1]
        label_K = np.zeros((label.shape[0], 2))

        # print(np.unique(label_2))

        for i in range(label.shape[0]):
            K = np.unique(label[i, :, 1][label[i, :, 1] != 0])
            if len(K) == 2:
                label_K[i, :] = K

                if label[i, 0, 1] != label_K[i, 0]:
                    label_K[i, :] = label_K[i, ::-1]

            elif len(K) == 1:
                states = label_2[i, :]
                if 1 in states:
                    # print(np.unique(states))
                    if states[0] == 1:
                        label_K[i, :] = [0, K[0]]
                    else:
                        label_K[i, :] = [K[0], 0]

                    # print(label_regression[i,:])

                else:
                    label_K[i, :] = [K[0], K[0]]

            else:
                if np.unique(label[i, :, 1]) == 0:
                    label_K[i, :] = [0, 0]
                else:

                    # print(np.unique(label[i,:,1]))

                    # print(Ds)
                    raise Exception("more than 2 diffusions")

        # print(np.unique(label_2))
        label_alpha = np.zeros((label.shape[0], 2))

        for i in range(label.shape[0]):
            alpha = np.unique(label[i, :, 0][label[i, :, 0] != 0])
            if len(alpha) == 2:
                label_alpha[i, :] = alpha
                if label[i, 0, 0] != label_alpha[i, 0]:
                    label_alpha[i, :] = label_alpha[i, ::-1]

            elif len(alpha) == 1:
                states = label_2[i, :]
                if 1 in states:
                    # print(np.unique(states))
                    if states[0] == 1:
                        label_alpha[i, :] = [0, alpha[0]]
                    else:
                        label_alpha[i, :] = [alpha[0], 0]

                    # print(label_regression[i,:])

                else:
                    label_alpha[i, :] = [alpha[0], alpha[0]]

            else:
                if np.unique(label[i, :, 1]) == 0:
                    label_alpha[i, :] = [0, 0]
                else:

                    # print(np.unique(label[i,:,1]))

                    # print(Ds)
                    raise Exception("more than 2 diffusions")

        label_segmentation = np.zeros((label_2.shape[0], label_2.shape[1]))

        for i in range(label.shape[0]):
            if label_K[i, 0] == label_K[i, 1]:
                position = label[i, :, 1] == label_K[i, 0]
                label_segmentation[i, position] = 1
            else:

                position_1 = label[i, :, 1] == label_K[i, 0]
                position_2 = label[i, :, 1] == label_K[i, 1]

                label_segmentation[i, position_1] = 1
                label_segmentation[i, position_2] = 2

        return (
            torch.from_numpy(data.astype(np.float32)),
            torch.from_numpy(label_segmentation.astype(np.float32)),
            torch.from_numpy(label_K),
            torch.from_numpy(label_alpha),
        )
        # torch.from_numpy(label_2.astype(np.float32)),


def add_noise(data):
    noise_amplitude = np.random.choice(
        [
            0,
            0.01,
            0.1,
        ]
    )
    noise = np.random.normal(0, noise_amplitude, data[:, :, :].shape)
    data[:, :, :][data[:, :, 1:] != 0] = (
        data[:, :, :][data[:, :, 1:] != 0] + data[:, :, :][data[:, :, 1:] != 0] * noise
    )
    return data