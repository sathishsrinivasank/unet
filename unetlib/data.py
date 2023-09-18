# import the necessary packages
import tensorflow as tf

def resize(inputImage, inputMask):
	# resize the image and mask to a predefined dimension
	# and return them
	inputImage = tf.image.resize(inputImage, (128, 128),
		method="nearest")
	inputMask = tf.image.resize(inputMask, (128, 128),
		method="nearest")
	return (inputImage, inputMask)

def augment(inputImage, inputMask):
	# randomly flip the image and mask with a probability of 50%
	if tf.random.uniform(()) > 0.5:
		inputImage = tf.image.flip_left_right(inputImage)
		inputMask = tf.image.flip_left_right(inputMask)

	# return the augmented image and mask
	return (inputImage, inputMask)

def normalize(inputImage, inputMask):
	# normalize the input image and mask
	inputImage = tf.cast(inputImage, tf.float32) / 255.0
	inputMask -= 1
	
	# return the input image and the mask
	return (inputImage, inputMask)

def load_image_train(datapoint):
	# get the input image and mask from the datapoint
	inputImage = datapoint["image"]
	inputMask = datapoint["segmentation_mask"]

	# resize, augment and normalize the image and mask
	(inputImage, inputMask) = resize(inputImage, inputMask)
	(inputImage, inputMask) = augment(inputImage, inputMask)
	(inputImage, inputMask) = normalize(inputImage, inputMask)

	# return the processed image and mask
	return (inputImage, inputMask)

def load_image_test(datapoint):
	# get the input image and mask from the datapoint
	inputImage = datapoint["image"]
	inputMask = datapoint["segmentation_mask"]

	# resize and normalize the image and mask
	(inputImage, inputMask) = resize(inputImage, inputMask)
	(inputImage, inputMask) = normalize(inputImage, inputMask)

	# return the processed image and mask
	return (inputImage, inputMask)

def get_steps_for_training(info, batchSize):
	# calculate the training length and steps per epoch
	trainLength = info.splits["train"].num_examples
	stepsPerEpoch = trainLength // batchSize
	valSubsplits = 5
	testLength = info.splits["test"].num_examples
	validationSteps= testLength // batchSize // valSubsplits
	
	# return the calculated steps per epoch for the train and validation
	# dataset
	return (stepsPerEpoch, validationSteps)