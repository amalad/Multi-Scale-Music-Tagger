import numpy as np
import librosa
import gzip
import pickle
import os
import csv
import config

def read_annotations(annotations_file_path):
	"""
	Prunes tags by frequency
	:param annotations_file_path: Path of original annotations file
	:return: IDs of data instances, Names of tags, frequencies of tags, tags annotations for data, paths of audio
	clips (all in decreasing order of tag frequencies)
	"""
	print("Reading raw annotations...")
	ids = []
	names = []
	tags = np.empty((0, 188), dtype="<U1")
	paths = []
	with open(annotations_file_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter="\t")
		names =  np.array(next(csv_reader)[1:189])
		for row in csv_reader:
			ids.append(int(row[0]))
			tags = np.append(tags, [row[1:189]], axis=0)
			paths.append(row[189])
	tags = tags.astype(int)
	freq = np.sum(tags, axis=0)
	print(len(ids), names.shape, freq.shape, tags.shape, len(paths))
	print("Done")
	return ids, names, freq, tags, paths

def generate_annotations(names, freq, tags, write_path):
	"""
	Generates a custom annotations file
	:param names: Names of tags
	:param freq: Frequencies of tags
	:param write_path: File to write custom annotations to
	"""
	print("Generating processed annotations file...")
	sorted_indices = freq.argsort()[::-1]
	pickle.dump((freq[sorted_indices], names[sorted_indices], tags[:, sorted_indices]), open(write_path, "wb"))
	print("Done")

def generate_spectrograms(file_paths, spec_path, print_every=100, use_numpy=True, compress=False):
	"""
	Generates spectrograms for audio data
	:param file_paths: List of paths of audio files
	:param print_every: Determines frequency of progress reports
	:param use_numpy: Use numpy or not
	:param compress: Compress or not
	"""
	print("Generating spectrograms....")
	specs = []
	count = 0
	for index, path in enumerate(file_paths):
	    count += 1
	    if (index) in config.DELETE:
	        specs.append(np.zeros((128, 628), dtype=np.float32))
	        continue
	    signal, rate = librosa.load("data/" + path, sr=config.SR)
	    specs.append(np.log(np.clip(librosa.feature.melspectrogram(signal, rate), config.EPS, np.inf)).astype(np.float32))
	    if (index + 1) % print_every == 0:
	        print("{} spectrograms calculated".format(index + 1))
	print("Done: " + str(count))
	specs = np.array(specs)
	print("Dumping spectrograms....")
	if not use_numpy:
	    pickle.dump(specs, gzip.open(spec_path, "wb") if compress else open(spec_path, "wb"))
	else:
	    if compress:
	        np.savez_compressed(open(spec_path, "wb"), specs)
	    else:
	        np.save(open(spec_path, "wb"), specs)
	print("Done")

if __name__ == "__main__":

	# Load annotations
	_, names, freq, tags, paths = read_annotations(config.ANNOTATIONS_PATH)

	# Generate custom annotations file
	generate_annotations(names, freq, tags, config.DATA_PATH + "annotations.pickle")

	# Generate Spectrograms
	generate_spectrograms(paths, config.DATA_PATH + "Spectrograms.data")