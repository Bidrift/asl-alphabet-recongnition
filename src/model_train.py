import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Function to load the entire dataset from a CSV file and create a DataLoader
def load_dataloader_from_csv(csv_file, batch_size=128, img_shape=(1, 32, 32)):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Separate the features (image data) and labels
    labels = df['label'].values
    features = df.drop(columns=['label']).values

    # Reshape the features back into the original image shape
    features = features.reshape(-1, *img_shape)

    # Convert to PyTorch tensors
    data_tensors = torch.tensor(features, dtype=torch.float32)
    label_tensors = torch.tensor(labels, dtype=torch.long)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(data_tensors, label_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader

# Path may differ from where you
train_loader = load_dataloader_from_csv("dataset/preprocessed/asl_alphabet/train.csv", batch_size=128)
val_loader = load_dataloader_from_csv("dataset/preprocessed/asl_alphabet/val.csv", batch_size=128)
test_loader = load_dataloader_from_csv("dataset/preprocessed/asl_alphabet/test.csv", batch_size=128)

# Example: Iterate over the train_loader
for batch_idx, (data, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
