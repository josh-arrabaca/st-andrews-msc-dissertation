# Import Libraries
import os.path
import numpy as np
from pydub import AudioSegment
from scipy import signal
import matplotlib.pyplot as plt

# Define the function that will convert the files
def batch_wav_to_png (folder):
    """ Goes through all subfolders in a chosen folder, and converts wav files to spectrogram png."""
    print("Finding files and converting wav to png...\n")

    # Create folders for the images
    if not os.path.exists(os.path.join("data", "images", "Insular")):
        os.makedirs(os.path.join("data", "images", "Insular"))
    if not os.path.exists(os.path.join("data", "images", "Pelagic")):
        os.makedirs(os.path.join("data", "images", "Pelagic"))

    # Initialise counters
    wavfile_counter = 0
    otherfile_counter = 0
    folder_counter = 0

    # Walkthrough all subfolders
    for root, dirs, files in os.walk(folder):
        
        for file in files:

            # Process files if they are wav
            if file.endswith((".wav")) and any(dolphin_type in str(root) for dolphin_type in ["Pelagic", "Insular"]):
                fileloc =  os.path.join(root, file)

                # Extract raw audio data from the wav, and "flatten" channels to only one channel
                sound = AudioSegment.from_wav(fileloc)
                sound = sound.set_channels(1)

                # Convert the result to ndarry, and find the sample_rate
                samples = sound.get_array_of_samples()
                samples = np.array(samples).astype(np.int16)
                sample_rate = sound.frame_rate

                # Convert to spectrogram data
                frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

                # Plot the result without any axes or labels
                # Part of this code was provided by Dr Chrissy Fell, School of Mathematics and Statistics, University of St Andrews
                plt.pcolormesh(times, frequencies, np.log(spectrogram))
                plt.tick_params(axis='both', which='both', bottom=False, left=False, top=False,
                                labelbottom=False, labelleft=False)
                plt.rcParams["figure.figsize"] = (12,8)
                if "Insular" in str(root): 
                    my_file = file + ".png"
                    plt.savefig(os.path.join(folder, "images", "Insular", my_file))  
                elif "Pelagic" in str(root):
                    my_file = file + ".png"
                    plt.savefig(os.path.join(folder, "images", "Pelagic", my_file))
                plt.close()
                
                #Increment the counters
                wavfile_counter += 1
                if wavfile_counter % 50 == 0:
                    print (f"Processing {wavfile_counter}th wave file...")

            else:
                otherfile_counter += 1

        folder_counter += 1

    print (f"Processing done!\nConverted {wavfile_counter} .wav files to .png in {folder_counter} folders.\nThe files are located in '{folder}\images' folder.\n\nIgnored {otherfile_counter} other files.")


# Name the folder with wav files here
folder = "data"

# Run the function
batch_wav_to_png (folder)
