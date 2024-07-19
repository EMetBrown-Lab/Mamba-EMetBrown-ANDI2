# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from einops import rearrange


# Function to calculate the angle between two points
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)


# Function to pad data and labels to a specific shape
def apply_padding(data_df, N, T_max, train=True):
    # Define the final shape of the data and labels
    final_shape = (N, T_max - 1, 6)

    # Initialize the final data and labels with zeros
    final_data = np.zeros(final_shape)
    final_label = np.zeros((N, T_max - 1, 3)) if train else None

    # Select a random subset of trajectory indices
    selected_ids = np.random.choice(
        data_df["traj_idx"].unique(),
        size=N,
        replace=len(data_df["traj_idx"].unique()) < N,
    )

    # Iterate over the selected trajectory indices
    for n, id in enumerate(selected_ids):
        # Filter the data for the current trajectory index
        exp = data_df[data_df["traj_idx"] == id]

        # Extract the data and labels for the current trajectory
        data = exp[["frame", "x", "y"]].to_numpy()
        data[:, 0] -= data[0, 0] - 1  # Shift frame rate to start from 1
        data[:, 1] -= data[0, 1]  # Shift initial position to 0
        data[:, 2] -= data[0, 2]  # Shift initial position to 0

        # Calculate displacement, mean displacement, and angles
        Dx = data[1:, 1] - data[:-1, 1]
        Dy = data[1:, 2] - data[:-1, 2]
        MDx = np.zeros(len(Dx))
        MDy = np.zeros(len(Dx))
        angles = np.zeros(len(Dx))
        distance_displacement = np.sqrt(np.power(Dx, 2) + np.power(Dy, 2))

        for i in range(1, len(Dx) + 1):
            MDx[i - 1] = np.mean(data[i:, 1] - data[:-i, 1])
            MDy[i - 1] = np.mean(data[i:, 2] - data[:-i, 2])
            angles[i - 1] = angle_between(
                (data[(i - 1), 1], data[(i - 1), 2]), (data[i, 1], data[i, 2])
            )

        # If training, extract labels
        if train:
            label = exp[["alpha", "D", "state"]].to_numpy()
            label[:, 2] += 1  # Shift states to start from 1

        # Pad or truncate data and labels to the final shape
        if data.shape[0] > T_max:
            final_data[n, :, :] = np.column_stack(
                (
                    Dx[: (T_max - 1)],
                    Dy[: (T_max - 1)],
                    MDx[: (T_max - 1)],
                    MDy[: (T_max - 1)],
                    distance_displacement[: (T_max - 1)],
                    angles[: (T_max - 1)],
                )
            )
            if train:
                final_label[n, :, :] = label[: T_max - 1, :]
        else:
            final_data[n, : (data.shape[0] - 1), :] = np.column_stack(
                (Dx, Dy, MDx, MDy, distance_displacement, angles)
            )
            if train:
                final_label[n, : data.shape[0] - 1, :] = label[:-1, :]

    # Return the padded data and label
    return final_data, final_label


# Function to normalize data
def normalize_data(data):
    displacement_x = np.diff(data[:, :, 1], axis=1).flatten()
    displacement_y = np.diff(data[:, :, 2], axis=1).flatten()

    variance_x = np.sqrt(np.std(displacement_x))
    variance_y = np.sqrt(np.std(displacement_y))

    data[:, :, 1] = (data[:, :, 1] - np.mean(data[:, :, 1])) / variance_x
    data[:, :, 2] = (data[:, :, 2] - np.mean(data[:, :, 2])) / variance_y

    return data


# Function to list directory tree with pathlib
def list_directory_tree(starting_directory):
    return [file_path for file_path in Path(starting_directory).rglob("*.csv")]


# Custom dataset class for all data
@dataclass
class DatasetAllData(Dataset):
    filenames: list
    transform: bool = False
    pad: None | tuple = None
    noise: bool = False
    train: bool = True

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        df = pd.read_csv(self.filenames[idx])

        if self.pad is None:
            data = df[["traj_idx", "frame", "x", "y"]]
            label = np.asarray(df[["alpha", "D"]]) if self.train else None
            label_2 = np.asarray(df["state"]) if self.train else None
        else:
            if len(self.pad) != 2:
                raise ValueError("pad value should be set as (N, T_max)")
            data, label = apply_padding(df, *self.pad, train=self.train)
            label_2 = label[:, :, -1]
            label_2[label_2[:, :] > 0] = label_2[label_2[:, :] > 0]
            label = label[:, :, :-1]

        if self.transform:
            data = normalize_data(data)

        if self.noise:
            data = add_noise(data)

        if self.train:
            label_K = process_label_K(label, label_2)
            label_alpha = process_label_alpha(label, label_2)
            label_segmentation = process_label_segmentation(label, label_2, label_K)
            return (
                torch.from_numpy(data.astype(np.float32)),
                torch.from_numpy(label_segmentation.astype(np.float32)),
                torch.from_numpy(label_K),
                torch.from_numpy(label_alpha),
            )
        else:
            return torch.from_numpy(data.astype(np.float32))


# Function to add noise to data
def add_noise(data):
    noise_amplitude = np.random.choice([0.01, 0.1])
    noise = np.random.normal(0, noise_amplitude, data.shape)
    data[data != 0] += data[data != 0] * noise
    return data


def process_label_K(label, label_2):
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

    return label_K


def process_label_alpha(label, label_2):

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
    return label_alpha


def process_label_segmentation(label, label_2, label_K):

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

    return label_segmentation