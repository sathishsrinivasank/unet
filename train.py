# USAGE
# python train.py

# import the necessary packages
from unetlib.data import get_steps_for_training
from unetlib.model import build_unet_model
from unetlib.data import load_image_train
from unetlib.data import load_image_test
from unetlib.utils import display_learning_curves
from unetlib.utils import display
from unetlib import config
from tensorflow.keras.models import save_model
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os

# load the dataset
print("[INFO] loading the dataset...")
(dataset, info) = tfds.load(config.DATASET_NAME,
	data_dir=config.DATASET_PATH, with_info=True)

# process the training and testing dataset
print("[INFO] processing the train and test dataset...")
trainDataset = (dataset["train"]
	.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
)
testDataset = (dataset["test"]
	.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
)

# build an efficient train, val, and test data input pipeline
print("[INFO] buidling tf.data train, val, and test pipeline...")
trainBatches = (trainDataset
	.cache()
	.shuffle(config.BUFFER_SIZE)
	.batch(config.BATCH_SIZE)
	.repeat()
	.prefetch(buffer_size=tf.data.AUTOTUNE)
)
validationBatches = (testDataset
	.take(3000)
	.batch(config.BATCH_SIZE)
)
testBatches = (testDataset
	.skip(3000)
	.take(669)
	.batch(config.BATCH_SIZE)
)

# check whether the output folder exists, if it does not, create the
# output folder
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)

# visualize a random sample from the test dataset
print(f"[INFO] saving the sample datapoint to {config.SAMPLE_PATH}")
sampleBatch = next(iter(testBatches))
randomIndex = np.random.choice(sampleBatch[0].shape[0])
(sampleImage, sampleMask) = (sampleBatch[0][randomIndex],
	sampleBatch[1][randomIndex])
display(displayList=[sampleImage, sampleMask],
	imagePath=config.SAMPLE_PATH)

# build the U-Net model
print("[INFO] building the U-Net model...")
unetModel = build_unet_model()

# compile the model
print("[INFO] compiling the model...")
unetModel.compile(optimizer=tf.keras.optimizers.Adam(),
	loss="sparse_categorical_crossentropy", metrics="accuracy")

# calculate the training length and steps per epoch
(stepsPerEpoch, validationSteps) = get_steps_for_training(info=info,
	batchSize=config.BATCH_SIZE)

# train the model
print("[INFO] training the model...")
modelHistory = unetModel.fit(
	trainBatches,
	epochs=config.NUM_EPOCHS,
	steps_per_epoch=stepsPerEpoch,
	validation_steps=validationSteps,
	validation_data=validationBatches
)

# plot and save the learning curve to disk
print(f"[INFO] plot and save the learning curve to\
{config.LEARNING_CURVE_PATH}")
display_learning_curves(history=unetModel.history,
	numEpochs=config.NUM_EPOCHS, learningPath=config.LEARNING_CURVE_PATH)

# save the trained model to disk
print("[INFO] saving the trained model to disk...")
save_model(model=unetModel, filepath=config.MODEL_PATH,
	include_optimizer=False)