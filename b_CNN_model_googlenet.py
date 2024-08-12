# Adapted from the original Pytorch tutorial by Sasank Chilamkurthy

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader

from sklearn import metrics 

cudnn.benchmark = True
plt.ion()   # interactive mode

# Set 'random' seed
torch.manual_seed(220029955)

# Welcome message
print("Welcome! We will train the last layer of a pre-trained CNN model.\n")


# Define the transforms needed 
data_transforms = transforms.Compose([
        transforms.Resize([224,224]), # Minimum size needed for googlenet
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Required normalisation for Densenet
    ])

# Get the dataset from the images created from the wav files
dataset = datasets.ImageFolder(os.path.join("data", "images"), transform=data_transforms)

# Define the classes (Insular and Pelagic)
classes = dataset.classes

# Split the data into train, val and test sets
train_size = int(0.6 * len(dataset))
val_size = int((len(dataset) - train_size) / 2)
test_size = val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
print(f"The dataset consists of {train_size + val_size + test_size} datapoints, split as follows:")
print(f"Train set: {train_size} \nValidation set: {val_size} \nTest size: {test_size}\n")

# Define the device to be used for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'The device being used is: {device}\n')

# Define the batch size and number of epochs based on the device
if str(device) == "cuda:0":
    batch_size = 64
    num_epochs = 24
else:
    batch_size = 20
    num_epochs = 3

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

dataloaders = {"train": train_dataloader,
               "val": val_dataloader}

dataset_sizes = {"train": len (train_dataset),
                 "val": len(test_dataset)}


# below code is taken from 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
    
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n\n')
        print(f'The best val accuracy score is: {best_acc:4f}\n\n')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

# Loading googlenet
model_conv = torchvision.models.googlenet(weights='IMAGENET1K_V1')

# This part does the training on the final layer only
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# Now train the model, and view the loss and accuracy scores
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=num_epochs)

# This part creates the classes and scores from the validation data for the metrics
y_score = []
true_classes = []
predicted_classes = []

for inputs, labels in val_dataloader:
    inputs = inputs.to(device) 
    labels = labels.to(device)

    # This part 'flattens' the tensor into a list
    labels = labels.cpu().numpy().tolist()
    true_classes.extend(labels)

    with torch.no_grad():
        model_conv.eval()  
        output = model_conv(inputs)

        for each_output in output:
            predicted_class = each_output.cpu().data.numpy().argmax() # Numpify each output
            predicted_classes.append(predicted_class) # Conactenate

            p = torch.nn.functional.softmax(output, dim=1) # Get probabilities
            top_proba = p.cpu().numpy()[0][0] # Get probabilities of the positive class ('Insular') only
            y_score.append(top_proba)# predicted_classes

# Let's find the precision, recall and confusion matrix as well
print(f"Precision: {round(metrics.precision_score (true_classes, predicted_classes),5)}")
print(f"Recall: {round(metrics.recall_score(true_classes, predicted_classes),5)}")
print("Confusion matrix:\n", metrics.confusion_matrix(true_classes, predicted_classes))