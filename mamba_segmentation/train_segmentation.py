from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from EmetMamba_segmentation import EmetConfig, EmetMamba
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
        #((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )


# Function to pad data and labels to a specific shape
def apply_padding(data_df, N, T_max):
    # Define the final shape of the data and labels
    final_shape = (N, T_max, 3)

    # Initialize the final data and labels with zeros
    final_data = np.zeros(final_shape)
    final_label = np.zeros((N, T_max, 3))

    # Select a random subset of trajectory indices
    if len(data_df["traj_idx"].unique()) < N:
        selected_ids = data_df["traj_idx"].unique()
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
        # print(exp["frame"])
        label = exp[["alpha", "D", "state"]].to_numpy()
        ## adding one to the states
        label[:,2] = label[:,2] + 1
        # If the data is longer than T_max, truncate it
        if data.shape[0] > T_max:
            final_data[n, :, :] = data[:T_max, :]
            final_label[n, :, :] = label[:T_max, :]

        # Otherwise, pad the data to T_max
        else:
            # print((label.shape, T_max))
            final_data[n, :data.shape[0], :] = data
            final_label[n, :data.shape[0], :] = label

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
        label_regression = np.zeros((label.shape[0], 2))
        # print(np.unique(label_2))
        ## Make the _ only 2 _ output
        for i in range(label.shape[0]):
            Ds = np.unique(label[i,:,1][label[i,:,1] != 0])
            if  len(Ds) == 2:
                label_regression[i,:] = Ds

      
            elif len(Ds) == 1:
                states = label_2[i,:]
                if 1 in states:
                    # print(np.unique(states))
                    if states[0] == 1:
                        label_regression[i,:] = [0, Ds[0]]
                    else:
                        label_regression[i,:] = [Ds[0], 0]
                    
                    # print(label_regression[i,:])

                else:
                    label_regression[i,:] = [Ds[0],Ds[0]] 

            else:
                if  np.unique(label[i,:,1]) == 0:
                    label_regression[i,:] = [0,0]
                else :

                    # print(np.unique(label[i,:,1]))

                    # print(Ds)
                    raise Exception("more than 2 diffusions")

        ## Making the segmentation output

        label_segmentation = np.zeros((label_2.shape[0], label_2.shape[1]))



        for i in range(label.shape[0]):
            if label_regression[i,0] == label_regression[i,1]:
                position = label[i,:,1] == label_regression[i,0]
                label_segmentation[i,position] = 1
            else:
                position_1 = label[i,:,1] == label_regression[i,0]
                position_2 = label[i,:,1] == label_regression[i,1]
                label_segmentation[i,position_1] = 1
                label_segmentation[i,position_2] = 2

        np.unique(label_segmentation[i,:])

        




        # Normaliza alpha between 0 and 1
        # label[:,:,0] = label[:,:,0] / 2

        #return only D
        # label = label[:,:,1]
        # Return data and label


        return torch.from_numpy(data.astype(np.float32)), (
            torch.from_numpy(label_regression.astype(np.float32)),
            torch.from_numpy(label_segmentation.astype(np.float32)),
        )
    
def add_noise(data):
    noise_amplitude = np.random.choice([0.01, 0.1, 1])
    noise = np.random.normal(0, noise_amplitude, data[:,:,1:].shape)
    data[:,:,1:] = data[:,:,1:] + data[:,:,1:]*noise
    return  data

def train(a):

    all_data_set = list_directory_tree_with_pathlib(
    r"/home/m.lavaud/Documents/Zeus/I2/T_200_const_100_to_150/batch_T_Const_1",)

    bi_mamba_stacks, dropout, learning_rate, n_layer = a

    learning_rate = learning_rate
    max_epochs = 6
    max_particles = 20
    max_traj_len = 200
    
    
    training_dataset = Dataset_all_data(
        all_data_set[:2000], transform=False, pad=(max_particles, max_traj_len)
    )
    test_dataset = Dataset_all_data(
        all_data_set[-100:], transform=False, pad=(max_particles, max_traj_len)
    )
    dataloader = DataLoader(training_dataset, shuffle=True, batch_size=10, num_workers=0)
    dataloader_test = DataLoader(test_dataset, shuffle=True, batch_size=10, num_workers=0)
    
    config = EmetConfig(
        d_model=3,
        n_layers=16,
        dt_rank="auto",
        d_state=16,
        expand_factor=2,
        d_conv=4,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        rms_norm_eps=1e-5,
        bias=False,
        conv_bias=True,
        inner_layernorms=False,
        pscan=True,
        use_cuda=True,
        bi_mamba_stacks=bi_mamba_stacks,
        conv_stack=n_layer,
        dropout=dropout,
    )
    model = EmetMamba(config=config)
    model.train()
    

    classification_criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Define optimizer
    running_total_loss = []
    running_classification_total_loss = []
    running_regression_total_loss = []
    model = EmetMamba(config=config)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    running_test_loss = []
  
    
    
    
    for epoch in range(max_epochs):
        with tqdm(dataloader, unit="batch", disable=False) as tepoch:
            
            running_classification_loss = []
            running_regression_loss = []
            model.train()
            for inputs, (regression_targets, classification_targets) in tepoch:
            # for inputs, classification_targets in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # print(torch.unique(classification_targets))
                inputs = inputs.to("cuda")
                # print(classification_targets)
                classification_targets = torch.flatten(
                    classification_targets, start_dim=1, end_dim=2
                ).type(torch.LongTensor)
                classification_targets = classification_targets.to("cuda")
    

                optimizer.zero_grad()

                classification_output = model(inputs)
    
                # print(classification_output.size())
                # print(classification_targets)

                classification_loss = classification_criterion(
                    classification_output.view(-1, 3).to("cpu"),
                    classification_targets.view(-1).to("cpu"),
                )

                classification_loss = classification_loss.to("cuda")
                classification_loss.backward()

                optimizer.step()
    
                tepoch.set_postfix(
                    loss=classification_loss.item(),
                )
                running_classification_loss.append(classification_loss.item())
    
            running_classification_total_loss.append(np.mean(running_classification_loss))
            

            test_loss = evaluate_model(model,dataloader_test,classification_criterion)
            running_test_loss.append(test_loss)

    result = {"bi_mamba_stacks":bi_mamba_stacks,
              "n_layers":n_layer,
              "dropout":dropout,
              "learning_rate":learning_rate, 
              "running_classification_total_loss":running_classification_total_loss,
              "running_test_loss":running_test_loss
             }
    return result, model


def evaluate_model(model, dataloader, criterion, device = "cuda"):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, (_, targets) = batch
            inputs = inputs.to(device)
            targets = torch.flatten(
                targets, start_dim=1, end_dim=2
            ).type(torch.LongTensor)            

            outputs = model(inputs)
            loss = criterion(
                outputs.view(-1, 3).to("cpu"),
                targets.view(-1).to("cpu"),
            )
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        return total_loss / total_samples

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return (self.mse(torch.log(pred + 1), torch.log(actual + 1)))
