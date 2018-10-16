import numpy as np
import librosa
import gzip
import pickle
import os
import csv
import config

def clean_annotations(annotations_file_path):
	"""
	Reads and cleans the raw annotations file
	:param annotations_file_path: Path of raw annotations file
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
	for syn in synonyms:
		ind = [np.where(names==x)[0][0] for x in syn]
		ind.sort()
		comb = np.amax(tags[:, ind], axis=1)
		tags[:, ind[0]] = comb
		tags = np.delete(tags, ind[1:], axis=1)
		names = np.delete(names, ind[1:])
	freq = np.sum(tags, axis=0)
	print(len(ids), names.shape, freq.shape, tags.shape, len(paths))
	print("Done")
	return ids, names, freq, tags, paths

def generate_annotations_file(names, freq, tags, write_path):
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
	_, names, freq, tags, paths = clean_annotations(config.ANNOTATIONS_PATH)

	# Generate custom annotations file
	generate_annotations_file(names, freq, tags, config.DATA_PATH + "annotations_cleaned.pickle")

	# Generate Spectrograms
	generate_spectrograms(paths, config.DATA_PATH + "Spectrograms.data")
