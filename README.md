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

To generate spectrograms from the raw audio clips, run the "generate_python.py" script. This will create the following files in the data subfolder: "Spectrogram.data" and "annotations_cleaned.pickle".

We consider only the top 50 tags (can be changed in "config.py") for classification as tag frequencies are extremely skewed (See [this](https://github.com/keunwoochoi/magnatagatune-list#histogram-of-tags)).

Furthermore, we merge the following synonym tags, as proposed [here](https://github.com/keunwoochoi/magnatagatune-list#proposed-tag-preprocessing):
```
synonyms = [['beat', 'beats'],
			['chant', 'chanting'],
			['choir', 'choral'],
			['classical', 'clasical', 'classic'],
			['drum', 'drums'],
			['electro', 'electronic', 'electronica', 'electric'],
			['fast', 'fast beat', 'quick'],
			['female', 'female singer', 'female singing', 'female vocals', 'female voice', 'woman', 'woman singing', 'women'],
			['flute', 'flutes'],
			['guitar', 'guitars'],
			['hard', 'hard rock'],
			['harpsichord', 'harpsicord'],
			['heavy', 'heavy metal', 'metal'],
			['horn', 'horns'],
			['india', 'indian'],
			['jazz', 'jazzy'],
			['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
			['no beat', 'no drums'],
			['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
			['opera', 'operatic'],
			['orchestra', 'orchestral'],
			['quiet', 'silence'],
			['singer', 'singing'],
			['space', 'spacey'],
			['string', 'strings'],
			['synth', 'synthesizer'],
			['violin', 'violins'],
			['vocal', 'vocals', 'voice', 'voices'],
			['strange', 'weird']]
```

## Training the model:
To train the model, run the following command: `python train_music_tagger.py`.

The training parameters can be modified in "config.py".

## Results:
We evaluated our model using the average ROC AUC score. The data was split randomly in the ratio of 13:1:3 (train:validation:test) over the entire dataset. The model weights giving the best AUC score on the validation set for 100 epochs were saved to "data/trained_model_state_dict.pth" and the test score was calculated on these weights. The scores for one of our runs are reported below:

Best validation AUC score: 0.91590

Corresponding training cost: 0.11139

Corresponding test AUC score: 0.90936

Note: Since we're splitting the dataset randomly in our "train_music_tagger.py" script, these scores may vary on each run. For most of our runs, the best validation score and corresponding test score were both between 0.90 and 0.915.

## To Do:
We intend to modify our model a little bit and test it on the Million Songs Dataset. Will post results soon.

## Contact:
Feel free to ping me if you need the pretrained weights, the training logs for the reported scores or if you have any questions about the model.
