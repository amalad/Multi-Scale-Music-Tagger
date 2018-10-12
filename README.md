# Multi Scale Music Tagger
A Multi Scale Convolutional Neural Network Architecture for Music Auto-Tagging

Model developed and implemented for the Study Oriented Project Course (CS F266), Jan 2017 - April 2017.
Project by: Amala Deshmukh and [Tanmaya Dabral](https://github.com/many-facedgod)

# Downloading and Preparing the Data:
We used the MagnaTagATune dataset for our experiments. The data can be downloaded from [here](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) and should be stored in the "data" subfolder of the repository. After downloading the mp3.zip.* files, execute the following commands to generate the raw data: `cat mp3.zip.* > mp3_all.zip`

To generate spectrograms for the raw audio clips, run the generate_python.py script. This will create a file titled "Spectrogram.data" in the data subfolder.

# Training the model:
To train the model, run the following command: `python music_tagger.py train`

# Testing the model:
To test the model, run the following command: `python music_tagger.py test`
