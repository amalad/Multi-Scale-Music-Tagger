DELETE = [16250, 24867, 25546] # Problematic files
N_TAGS = 50 # Number of output tags

# For Spectrogram generator
DATA_PATH = "data/"
ANNOTATIONS_PATH = DATA_PATH + "annotations_final.csv"
SR = 11025 # Sampling Rate
EPS = 1e-20 # For log clipping

# Training parameters
USE_CUDA = True
LR = 0.0001 # Learning rate
BATCH_SIZE = 20
TRAIN_SIZE = 20000
VALIDATION_SIZE = 5800 # Size of validation set
TEST_SIZE = 0 # Size of test set
EPOCHS = 100
VALIDATON_BATCH_SIZE = 20
L2 = 0.0001