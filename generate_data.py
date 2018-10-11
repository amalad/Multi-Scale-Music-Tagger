import numpy as np
import librosa
import gzip
import pickle
import os
import csv
import config

def filter_tags(n_tags, annotations_file_path, write_path):
	"""
	Prunes tags by frequency
	:param n_tags: Number of tags to be retained
	:param annotations_file_path: Path of original annotations file
	:param write_path: Path of file to dump the filtered annotations
	"""
	print("Filtering top " + str(n_tags) + " tags...")
	ids = []
	tags = np.empty((0, 188), dtype="<U1")
	paths = []
	with open(annotations_file_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter="\t")
		next(csv_reader) # Skip the header
		for row in csv_reader:
			ids.append(int(row[0]))
			tags = np.append(tags, [row[1:189]], axis=0)
			paths.append(row[189])
	tags = tags.astype(int)
	freq = np.sum(tags, axis=0)
	tags = tags[:, freq.argsort()[-1 * n_tags:][::-1].tolist()]
	pickle.dump((ids, tags, paths), open(write_path, "wb"))
	print("Done")

def generate_spectrograms(file_paths, print_every=5, use_numpy=True, compress=False):
	"""
	Generates spectrograms for audio data
	:param file_paths: List of paths of audio files
	:param print_every: Determines frequency of progress reports
	:param use_numpy: Use numpy or not
	:param compress: Compress or not
	"""
	print("Generating spectrograms....")
	specs = []
	for index, path in enumerate(file_paths):
	    if index in config.DELETE:
	        specs.append(np.zeros((128, 628), dtype=_type))
	        continue
	    signal, rate = librosa.load("data/" + path, sr=config.SR)
	    specs.append(np.log(np.clip(librosa.feature.melspectrogram(signal, rate), config.EPS, np.inf)).astype(np.float32))
	    if (index + 1) % print_every == 0:
	        print("{} spectrograms calculated".format(index + 1))
	print("Done")
	specs = np.array(specs)
	print("Dumping spectrograms....")
	if not use_numpy:
	    pickle.dump(specs, gzip.open("data/Spectrograms.data", "wb") if compress else open("data/Spectrograms.data", "wb"))
	else:
	    if compress:
	        np.savez_compressed(open("data/Spectrograms.data", "wb"), specs)
	    else:
	        np.save(open("data/Spectrograms.data", "wb"), specs)
	print("Done")

if __name__ == "__main__":

	# Load annotations
	filtered_annotations_file = config.DATA_PATH + "annotations_final_" + str(config.N_TAGS) + ".pickle"
	if not os.path.exists(filtered_annotations_file):
		filter_tags(config.N_TAGS, config.ANNOTATIONS_PATH, filtered_annotations_file)
	print("Loading annotations....")
	ids, tags, paths = pickle.load(open(filtered_annotations_file, "rb"))
	print("Done")

	# Generate Spectrograms
	generate_spectrograms(paths)