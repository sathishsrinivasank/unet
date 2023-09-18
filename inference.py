# USAGE:
# python inference.py

# import the necessary packages
from unetlib.data import load_image_test
from unetlib.utils import show_predictions
from unetlib import config
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf

# load the dataset
print("[INFO] loading the dataset...")
(dataset, info) = tfds.load(config.DATASET_NAME,
	data_dir=config.DATASET_PATH, with_info=True)

# process the testing dataset
testDataset = (dataset["test"]
	.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
)

# build an efficient test data input pipeline
print("[INFO] buidling tf.data test pipeline...")
testBatches = (testDataset
	.skip(3000)
	.take(669)
	.batch(config.BATCH_SIZE)
)

# load the UNet model
print("[INFO] loading the U-Net model...")
unetModel = load_model(filepath=config.MODEL_PATH, compile=False)

# infer on the model and save the predicted images to disk
print("[INFO] running inference and saving the predictions to disk...")
show_predictions(unetModel=unetModel, dataset=testBatches.skip(5),
	outputPath=config.OUTPUT_PATH, num=3)