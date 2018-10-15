# Multi Scale Music Tagger
A Multi Scale Convolutional Neural Network Architecture for Music Auto-Tagging

A PyTorch implementation of the model developed for the Study Oriented Project Course (CS F266) at BITS Pilani - Hyderabad Campus, Jan 2017 - April 2017. 

Project by: Amala Deshmukh and [Tanmaya Dabral](https://github.com/many-facedgod)

## Requirements
1. Python 3.6 
2. PyTorch 0.4.1 
3. Librosa 0.6.2 (Required for generating spectrograms) 

Check "requirements.txt".

## Downloading and Processing the Data:
We used the MagnaTagATune dataset for our experiments. The dataset can be downloaded from [here](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) and should be stored in the "data" subfolder of the repository. After downloading the mp3.zip.* files, execute the following commands to generate the raw data: `cat mp3.zip.* > mp3_all.zip`, followed by `unzip mp3_all.zip`

To generate spectrograms from the raw audio clips, run the "generate_python.py" script. This will create the following files in the data subfolder: "Spectrogram.data" and "annotations.pickle".

## Training the model:
To train the model, run the following command: `python train_music_tagger.py`.

The training parameters can be modified in "config.py".

## Results:
We evaluated our model using the average ROC AUC score. The data was split randomly in the ratio of 13:1:3 (train:validation:test) over the entire dataset. The model parameters giving the best AUC score on the validation set for 100 epochs were saved to "data/TrainedModel.pickle" and the test score was calculated on these parameters. The scores for one of our runs are reported below:

Best validation AUC score: 0.9057

Corresponding training cost: 0.1068

Corresponding test AUC score: 0.9045

Note: Since we're splitting the dataset randomly in our "train_music_tagger.py" script, these scores may vary on each run. For most of our runs, the best validation score and corresponding test AUC score were between 0.90 and 0.91.

## Contact:
For any questions about the model or the pretrained weights and training logs for the scores reported above, feel free to ping me.
