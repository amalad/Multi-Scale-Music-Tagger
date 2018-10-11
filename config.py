DELETE = [16250, 24867, 25546]
N_TAGS = 50 # Number of output tags

# For Spectrogram generator
DATA_PATH = "data/"
ANNOTATIONS_PATH = DATA_PATH + "annotations_final.csv"
SR = 11025 # Sampling Rate
EPS = 1e-20 # For log clipping

# For MSCNN model
LR = 0.0001 # Learning rate
BATCH_SIZE = 20
TRAIN_SIZES = [5000, 5000, 5000, 5000] # List of super batches
VALIDATION_SIZES = 5800
EPOCHS = 100
VALIDATON_BATCH_SIZES = 20
L2 = 0.0001