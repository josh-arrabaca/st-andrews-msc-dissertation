# Import Libraries
import torch
import os.path
from scipy import signal
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'The device being used is: {device}\n')

# Load the best model
if str(device) == "cuda:0":
    model = torch.load('best-model-resnet.pth')
else:
    model = torch.load('best-model-resnet.pth', map_location=torch.device('cpu'))
model.eval()

# Define function for converting single wav to png file 
def new_audio_to_png (wavfile):
    sound = AudioSegment.from_wav(wavfile)
    sound = sound.set_channels(1)

    # Convert the result to ndarry, and find the sample_rate
    samples = sound.get_array_of_samples()
    samples = np.array(samples).astype(np.int16)
    sample_rate = sound.frame_rate

    # Convert to spectrogram data
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # Plot the result without any axes or labels
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.tick_params(axis='both', which='both', bottom=False, left=False, top=False,
                    labelbottom=False, labelleft=False)
    plt.rcParams["figure.figsize"] = (12,8)
    plt.savefig("spectrogram_to_be_predicted.png", bbox_inches='tight')
    plt.close()

file_name = sys.argv[1]
new_audio_to_png(os.path.join(file_name))

# Define the transforms needed 
data_transforms = transforms.Compose([
        transforms.Resize([224,224]), # Minimum size needed for Resnet
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Required normalisation for Resnet
    ])

# Define classes
classes = ['Insular', 'Pelagic']

# Convert the image to a tensor
img = Image.open("spectrogram_to_be_predicted.png").convert('RGB')
new_tensor = data_transforms(img)
new_tensor = new_tensor.unsqueeze(0)

# Move the data to the GPU if using gpu
if str(device) == "cuda:0":
     new_tensor = new_tensor.to(device)

# Get prediction
with torch.no_grad():
      output = model(new_tensor)
      index = output.data.cpu().numpy().argmax()
      class_name = classes[index]

# Get probabilities of the prediction
p = torch.nn.functional.softmax(output, dim=1)
top_proba = p.cpu().numpy()[0][index]

# Print the results
print("output: ", output)
print("index:", index)
print("class name:", class_name)
print("probabilities:", p)
print(f"\nThere's a {round((top_proba*100),2)} % chance that this audio file belongs to the {class_name} group.")
