# standrewsdissertation
This code is for my MSc Data-Intensive Analysis dissertation entitled _"Convolutional Neural Networks for the Classification of Pseudorca crassidens Passive Acoustic Data"_. The latest version can be found in https://github.com/josh-arrabaca/standrewsdissertation/releases/tag/Submission2.

To run these, please ensure the code is in the top folder, and the related dataset is in the subfolder "data", and that the audio files are in the appropriate sub-subfolders "Insular" and "Pelagic".

To make predictions, you can run at the command line: "python c_make_prediction.py filename.wav".

The flow of experiments is as follows:
1. For converting the audio files to spectrograms:
    * a_convert_wav_to_png.py
2. For applying Transfer Learning to different pretrainined CNNs. These mostly use the same code, except a different model was loaded for each experiment:
    * b_CNN_model_DenseNet.py
    * b_CNN_model_googlenet.py
    * b_CNN_model_efficientnet.py
    * b_CNN_model_ResNet.py
3. For applying the \textit{Weighed Random Sampler} to the imbalanced dataset. Much of the code is similar to the above, except for adding the sampler library, and plotting the new distibution per batch:
    * b_CNN_model_ResNet_wrs.py
4. For hyperparameter tuning. Tuning was done first with the learning rate, then the momentum value. The code is also very similar to the above, except for the for-loop code added to train and test each of the hyperparameter values:
    * b_CNN_model_ResNet_wrs_lr.py
    * b_CNN_model_ResNet_wrs_momentum.py
5. For testing the best model with the unseen test set. As with the above, most of the code is the same, except for the metrics: 
    * b_CNN_model_ResNet_best.py
6. For making predictions using the saved best model: 
    * c_make_prediction.py
