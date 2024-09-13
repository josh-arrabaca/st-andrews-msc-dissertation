# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import WeightedRandomSampler

cudnn.benchmark = True
plt.ion()   # interactive mode

# Set 'random' seed
torch.manual_seed(220029955)

# Welcome message
print("Welcome! We will train the last layer of a pre-trained CNN model.\n")


# Define the transforms needed 
data_transforms = transforms.Compose([
        transforms.Resize([224,224]), # Minimum size needed for Densenet
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
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Define a function to visualise the class distributions per batch
def plot_classes(dataloader):
        batch_num_counts = []
        cur_batch = 0
        class_0_counts = []
        class_1_counts = []
        for i in range(len(train_dataloader)):
                my_inputs, my_labels = next(iter(train_dataloader))
                this_class_0_count = 0
                this_class_1_count = 0
                for label in my_labels:
                        if label.cpu().numpy() == 0:
                                this_class_0_count+=1
                        elif label.cpu().numpy() == 1:
                                this_class_1_count+=1
                class_0_counts.append(this_class_0_count)
                class_1_counts.append(this_class_1_count)               
                batch_num_counts.append(cur_batch)
                cur_batch += 1

        df = pd.DataFrame({'Insular': class_0_counts,
                        'Pelagic': class_1_counts})

        df.plot(kind="bar", 
                title="Weighed Distribution of Classes per Batch",
                xlabel="Batch Number",
                ylabel="Images per Batch")
        
        plt.savefig("weighed_distribution_pre_optim.png", bbox_inches='tight')
        plt.close()
        print(df)

# # Visualise the distribution
# plot_classes(train_dataloader, "Imbalanced_Distribution_of_Classes_per_Batch")

# Define function to create a balanced sampler
# The following code was adapted from Vivek Maskara
# https://www.maskaravivek.com/post/pytorch-weighted-random-sampler/
def balanced_sampler(full_dataset, train_dataset):
    # Find number of samples per class
    y_train_indices = train_dataset.indices
    y_train = [full_dataset.targets[i] for i in y_train_indices]
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    # Find weights per class
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)

    # Define sampler
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    return sampler

# Create a balanced sampler
sampler = balanced_sampler(dataset, train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)


# Visualise the balanced dataset per batch
plot_classes(train_dataloader)

dataloaders = {"train": train_dataloader,
               "val": val_dataloader}

dataset_sizes = {"train": len (train_dataset),
                 "val": len(test_dataset)}


# Adapted from the original Pytorch tutorial by Sasank Chilamkurthy
train_losses = []
val_losses = []
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

                if phase == 'train':
                     train_losses.append(epoch_loss)
                else:
                     val_losses.append(epoch_loss)
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


# We load ResNet since this had the best base accuracy compared to the other CNNs
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

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

# Now we train the model
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=num_epochs)


# This part creates the classes and scores from the training data for the plots
plt.plot(train_losses, 'b', label='Training Loss')
plt.plot(val_losses, 'r--', label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Model Loss over Epochs")
plt.savefig("losses.png", bbox_inches='tight')
plt.close()

# # This is for validation
# val_sampler = balanced_sampler(dataset, val_dataset)
# val_dataloader2 = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
y_score = []
true_classes = []
predicted_classes = []

# for inputs, labels in val_dataloader2:
for inputs, labels in train_dataloader:
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
            
# Let's also check the Precision-Recall Curve, and the Threshold Plots after using the weighed sampler
# These plots are saved locally as image files
display = PrecisionRecallDisplay.from_predictions(true_classes,
                                                    y_score,
                                                    name="CNN",
                                                    plot_chance_level=True)
_ = display.ax_.set_title("2-class Precision-Recall curve (Weighed Distribution)")
plt.ylim([0,1])
plt.savefig("precision-recall-curve-weighed.png", bbox_inches='tight')
plt.close()

precision, recall, thresholds = precision_recall_curve(true_classes, y_score)
plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(thresholds, recall[:-1], 'r--', label='Recall')
plt.xlabel('Threshold')
plt.legend(loc='lower left')
plt.ylim([0,1])
plt.title("Precision-Recall-Threshold Plot (Weighed Distribution)")
plt.savefig("precision-recall-threshold-weighed.png", bbox_inches='tight')
plt.close()

# Let's find the precision, recall and confusion matrix as well
print(f"Precision: {round(metrics.precision_score (true_classes, predicted_classes),5)}")
print(f"Recall: {round(metrics.recall_score(true_classes, predicted_classes),5)}")
print("Confusion matrix:\n", metrics.confusion_matrix(true_classes, predicted_classes))