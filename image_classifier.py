import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report

class CNN(nn.Module):
    """ A convolutional neural network (CNN) for image classification

    Attributes:
    -----------
        conv1, conv2, conv3 (nn.Conv2d): Convolutional layers.
        pool (nn.MaxPool2d): Pooling layer.
        fc1, fc2 (nn.Linear): Fully connected layers.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 100)
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data():
    """ Loads and returns the CIFAR-100 dataset.

    Returns:
    --------
        dataset: The CIFAR-100 dataset.
    """
    dataset = load_dataset("cifar100")
    return dataset

def transform_features(examples:dict):
    """ Applies transformations to the dataset.

    Parameters:
    -----------
        examples: A batch of examples from the dataset.

    Returns:
    --------
        examples: The transformed batch of examples (as tensors).
    """
    examples['pixel_values'] = [transforms.ToTensor()(image) for image in examples['img']]
    return examples

def show_images(
    images:list,
    labels:list,
    class_names:list,
    num_images=5
) -> None:
    """ Displays a grid of images with their labels.

    Parameters:
    -----------
        images: List of images to display.
        labels: Corresponding labels of the images.
        class_names: List of class names for labels.
        num_images: Number of images to display (default is 5).
    """
    # Set the number of rows and columns for the subplot grid
    num_rows = num_images // 5 + int(num_images % 5 != 0)
    num_cols = min(num_images, 5)
    
    plt.figure(figsize=(2.5 * num_cols, 2.5 * num_rows))
    
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

    plt.show()

def train_model(
    model:CNN,
    train_data:DataLoader,
    optimizer,
    criterion,
    num_epochs:int=10,
    label_type:str='fine_label'
) -> None:
    """ Trains the CNN model.

    Parameters:
    -----------
        model (CNN): The CNN model to train. 
        train_data (DataLoader): The data loader containing training data.
        optimizer: The optimization algorithm to use for training.
        criterion: The loss function to calculate loss.
        num_epochs: The number of epochs to train the model for (default is 10).
        label_type: Label type to classify. Either 'fine_label' or 'coarse_label'.
    """
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        for batch in tqdm(train_data, desc='Training Batches', leave=False):
            inputs, labels = batch['pixel_values'], batch[label_type]
            optimizer.zero_grad()

            # Forward pass
            output = model(inputs)
            
            # Calculate loss
            loss = criterion(output, labels)
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()

def evaluate_model(
    model:CNN,
    test_data:DataLoader,
    class_names:list,
    label_type:str='fine_label'
) -> None:
    """ Evaluates the CNN model.

    Parameters:
    -----------
        model (CNN): The CNN model.
        test_data (DataLoader): The data loader containing test data.
        class_names: List of class names corresponding to the labels.
        label_type: Label type to classify. Either 'fine_label' or 'coarse_label'.
    """
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_data:
            images, labels = batch['pixel_values'], batch[label_type]
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())

    print(classification_report(true_labels, predicted_labels, target_names=class_names))
    print(f'Accuracy of the network on the 10000 test images for {label_type}: {100 * correct / total}%')

def main(plot_data:bool=False):
    """
    Main function to run the CNN model training and evaluation.

    Parameters:
    -----------
    plot_data: Whether to plot data images (default is False).
    """
    # Define hyperparameters
    lr, num_epochs = 0.001, 10

    # Define label type, either 'fine_label' or 'coarse_label'
    label_type = 'coarse_label'

    # Load the CIFAR-100 dataset
    dataset = load_data()

    class_names = dataset['train'].features[label_type].names

    if plot_data:
        num_images_to_show = 10
        example_batch = dataset['train'].select(range(num_images_to_show))
        example_images = example_batch['img']
        example_labels = example_batch[label_type]
        
        show_images(example_images, example_labels, class_names, num_images=num_images_to_show)

    dataset = dataset.map(transform_features, batched=True)
    dataset.set_format(type='torch', columns=['pixel_values', 'fine_label', 'coarse_label'])

    train_loader = DataLoader(dataset['train'], batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset['test'], batch_size=64, shuffle=False)

    # Model, criterion and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(
        model=model,
        train_data=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        label_type=label_type
    )

    evaluate_model(
        model=model,
        test_data=test_loader,
        class_names=class_names,
        label_type=label_type
    )

if __name__ == "__main__":
    main(plot_data=False)