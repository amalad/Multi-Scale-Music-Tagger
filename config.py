DELETE = [16250, 24867, 25546] # Problematic files

# For Spectrogram generator
DATA_PATH = "data/"
ANNOTATIONS_PATH = DATA_PATH + "annotations_final.csv" # Raw annotations file
SR = 11025 # Sampling Rate
EPS = 1e-20 # For log clipping

# Training parameters
N_TAGS = 50 # Number of output tags
USE_CUDA = True
EPOCHS = 100
L2 = 0.0001
LR = 0.0001 # Learning rate

TRAIN_SIZE = 19775
BATCH_SIZE = 20
VALIDATION_SIZE = 1520
VALIDATION_BATCH_SIZE = 20
TEST_SIZE = 4565
TEST_BATCH_SIZE = 20