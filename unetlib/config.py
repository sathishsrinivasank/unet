# import the necessary packages
import os

# define the dataset name and dataset path
DATASET_NAME = "oxford_iiit_pet:3.*.*"
DATASET_PATH = "dataset"

# define the batch and buffer size for the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 1000

# define the number of epochs to train the model
NUM_EPOCHS = 20

# define the output paths
OUTPUT_PATH = "output"
MODEL_PATH = os.path.join(OUTPUT_PATH, "unet")
SAMPLE_PATH = os.path.join(OUTPUT_PATH, "sample_test_visual.png")
LEARNING_CURVE_PATH = os.path.join(OUTPUT_PATH, "learning_curve.png")