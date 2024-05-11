# Author: Taylor Seghesio
# Organization: UNR CSE
# Course: CS 687
# date_Updated: 20APR2024

# Acknowledgements: (see final project report deliverable for documented citations in formal iEEE format)
# [10]  https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html ### resnet pre built model
# [16]  https://www.programiz.com/python-programming/methods/built-in/iter  ###used for iter
# [17]  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html  ###data_loader/pytorch
# [18]  https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/ ###imshow in python
# [19]  https://pytorch.org/vision/stable/transforms.html ### image augmentation
# [20]  https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib ### using plt.pause instead of block=False
# [21]  https://medium.com/@harshit4084/track-your-loop-using-tqdm-7-ways-progress-bars-in-python-make-things-easier-fcbbb9233f24 ### prog bar
# [22]  https://blog.paperspace.com/how-to-maximize-gpu-utilization-by-finding-the-right-batch-size/ ### helped with accuracy and algorithm performance
# [23]  https://www.linkedin.com/advice/3/how-can-you-improve-neural-network-performance-xkrxe#:~:text=Selecting%20the%20number%20of%20epochs,it%20based%20on%20validation%20performance. ###help with accuracy and curve performance


# About this code: This Python script uses PyTorch and torchvision libraries to train, validate, and test a ResNet18,
# residual network model, for classifying chest x-ray images into COVID-19 and Normal classes. The script involves data
# handling, data extraction, dataset creation, and transformations/augmentations on the data. A detailed view of the
# architecture is provided through torchsummary, and matplotlib provides a detailed output of training/validation
# metrics. During runtime, tqdm, offers a visualisation in the form of a progress bar for each epoch run. The script
# also utilizes CUDAs for GPU acceleration. Built modular, this script offers the ability to adjust and tune
# hyperparameters and choose different optimization algorithms for experimentation.


# imports for neural network
import torch
from torch import nn, optim
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR

# imports for vision tasks
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

# imports for preparing datasets
import os
import zipfile

# imports for visualizations
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# GLOBALS
BATCH_SIZE = 128
NUM_EPOCHS = 30

# Used for debugging CUDA execution and confirming correct initialization with correct CUDA device
print(torch.cuda.is_available())

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


def extract_dataset(dataset_path, extract_to):
    print(f"Verifying extraction in: {extract_to}")
    if not os.path.isdir(extract_to):
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")


def visualize_dataset(train_dir, val_dir, test_dir):
    def get_data(data_loader):
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        filenames = []
        random_data_sample = data_loader.dataset.indices[:10]
        for index in random_data_sample:
            full_path = data_loader.dataset.dataset.samples[index][0]
            filename = full_path.split(os.sep)[-1]
            filenames.append(filename)

        total_covid = 0
        total_normal = 0

        subset_indices = data_loader.dataset.indices
        for idx in subset_indices:
            path, label = data_loader.dataset.dataset.samples[idx]
            if label == data_loader.dataset.dataset.class_to_idx['COVID-19']:
                total_covid += 1
            elif label == data_loader.dataset.dataset.class_to_idx['Normal']:
                total_normal += 1

        return filenames, total_covid, total_normal

    train_files, train_covid, train_normal = get_data(train_dir)
    val_files, val_covid, val_normal = get_data(val_dir)
    test_files, test_covid, test_normal = get_data(test_dir)

    print('Train files:', train_files)
    print('Validation files:', val_files)
    print('Test files:', test_files)

    print('\nTotal training images:', len(train_dir.dataset))
    print('Total training COVID images:', train_covid)
    print('Total training Normal images:', train_normal)

    print('\nTotal validation images:', len(val_dir.dataset))
    print('Total validation COVID images:', val_covid)
    print('Total validation Normal images:', val_normal)

    print('\nTotal test images:', len(test_dir.dataset))
    print('Total test COVID images:', test_covid)
    print('Total test Normal images:', test_normal)


def prepare_datasets(extract_to, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transform,
    ])

    data_root = os.path.join(extract_to, "Curated X-Ray Dataset")

    full_dataset = ImageFolder(data_root, train_transform)
    train_size = int(0.7 * len(full_dataset))
    test_size = int(0.15 * len(full_dataset))
    val_size = len(full_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size])
    val_dataset.dataset.transform = test_dataset.dataset.transform = transform
    train_dir = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dir = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dir = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dir, val_dir, test_dir


def show_batch(data_dir):
    for images, labels in data_dir:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break

    plt.pause(0.001)
    input("Press [enter] to continue.")


def load_architecture(model, device):
    model.to(device)
    summary(model, (3, 256, 256))
    return model


def train_model(model, train_dir, val_dir, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001  # used when not using scheduler

    # optimizer choices for experimenting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = optim.Adam(model.parameters(), weight_decay=1e-6)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=1e-6)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=1e-6, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=2, gamma=0.1) # only used for some experiments

    early_stopping_patience = 10
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0

    epoch_train_loss_values = []
    epoch_val_loss_values = []
    epoch_train_acc_values = []
    epoch_val_acc_values = []

    for epoch in range(num_epochs):
        print(f'\nRunning epoch {epoch + 1}/{num_epochs}')
        model.train()
        train_losses, train_accuracies = [], []

        with tqdm(total=len(train_dir), desc=f'Epoch {epoch + 1}', unit='batch', leave=False) as pbar:
            for images, labels in train_dir:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                train_losses.append(loss.item())
                train_accuracies.append(acc)

                pbar.update(1)

        scheduler.step()  # only used if step scheduler is enabled

        epoch_train_loss = sum(train_losses) / len(train_losses)
        epoch_train_acc = sum(train_accuracies) / len(train_accuracies)
        epoch_train_loss_values.append(epoch_train_loss)
        epoch_train_acc_values.append(epoch_train_acc)

        model.eval()
        val_losses, val_accuracies = [], []
        with torch.no_grad():
            for images, labels in val_dir:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss.item())

                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                val_accuracies.append(acc)

        epoch_val_loss = sum(val_losses) / len(val_losses)
        epoch_val_acc = sum(val_accuracies) / len(val_accuracies)
        epoch_val_loss_values.append(epoch_val_loss)
        epoch_val_acc_values.append(epoch_val_acc)

        print(f'Epoch: {epoch + 1}\n'
              f'Train Acc: {epoch_train_acc:.3f}, Val Acc: {epoch_val_acc:.3f} '
              f'Train Loss: {epoch_train_loss:.3f}, Val Loss: {epoch_val_loss:.3f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
            print(f'Best Metric Updated: {best_val_loss:.3f} at epoch {best_epoch}')
        else:
            early_stopping_counter += 1
            print(f'Best Metric: {best_val_loss:.3f} at epoch: {best_epoch}\n')

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping after {early_stopping_patience} epochs of no improvement.")
            break

    print(f'Finished Training. Best Validation Loss: {best_val_loss:.3f} achieved at Epoch {best_epoch}')
    return epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values


def plot_results(epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values):
    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    x_train = [i + 1 for i in range(len(epoch_train_loss_values))]
    y_train = epoch_train_loss_values
    x_val = [i + 1 for i in range(len(epoch_val_loss_values))]
    y_val = epoch_val_loss_values
    plt.plot(x_train, y_train, label='Train Loss')
    plt.plot(x_val, y_val, label='Val Loss', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    x_train_acc = [i + 1 for i in range(len(epoch_train_acc_values))]
    y_train_acc = epoch_train_acc_values
    x_val_acc = [i + 1 for i in range(len(epoch_val_acc_values))]
    y_val_acc = epoch_val_acc_values
    plt.plot(x_train_acc, y_train_acc, label='Train Acc')
    plt.plot(x_val_acc, y_val_acc, label='Val Acc', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.pause(0.001)
    input("Press [enter] to continue.")


def test_model(model, test_dir, device, criterion):
    print("Testing model now...")
    model.eval()
    total_accuracy, total_test_loss = 0.0, 0.0
    num_batches = len(test_dir)

    with torch.no_grad():
        with tqdm(total=len(test_dir), desc='Testing Model: ', unit='batch', leave=False) as pbar:
            for data, labels in test_dir:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)
                accuracy = (outputs.argmax(dim=1) == labels).float().mean()

                total_test_loss += loss.item()
                total_accuracy += accuracy.item()

                pbar.update(1)

    avg_loss = total_test_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    print(f'Test Accuracy: {avg_accuracy:.3f}, Test Loss: {avg_loss:.3f}')


def main():
    # Handles our Dataset
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(root_dir, 'DATASETS', 'Curated X-Ray Dataset.zip')
    extract_to = os.path.join(root_dir, 'DATASETS')
    extract_dataset(dataset_path, extract_to)
    train_dir, val_dir, test_dir = prepare_datasets(extract_to, BATCH_SIZE)

    # Visualizes our Dataset - For debugging/confirmation
    visualize_dataset(train_dir, val_dir, test_dir)
    show_batch(train_dir)

    # Instantiates our ResNet18 Model and performs model tasks
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = load_architecture(model, DEVICE)
    results = train_model(model, train_dir, val_dir, NUM_EPOCHS, DEVICE)
    (epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values) = results
    plot_results(epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values)

    # Testing the model
    model.load_state_dict(torch.load('best_model.pth'))
    loss = nn.CrossEntropyLoss()
    test_model(model, test_dir, DEVICE, loss)

    input("Press Enter to close program and close plotted data/images...")


if __name__ == '__main__':
    main()
