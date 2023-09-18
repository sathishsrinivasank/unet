# import the necessary packages
from matplotlib import pyplot as plt
import tensorflow as tf
import os

def display(displayList, imagePath):
	# build a pyplot figure of a predefined size
	plt.figure(figsize=(15, 15))

	# initialize the title of the figure
	title = ["Input Image", "True Mask", "Predicted Mask"]

	# iterate over the list of image and dispay them over the pyplot
	# figure
	for (idx, image) in enumerate(displayList):
		plt.subplot(1, len(displayList), idx+1)
		plt.imshow(tf.keras.utils.array_to_img(image))
		plt.title(title[idx])
		plt.axis("off")
	
	# save the image to disk
	plt.savefig(imagePath)

def display_learning_curves(history, numEpochs, learningPath):
	# get the accuracy and the validation accuracy from the training
	# history
	acc = history.history["accuracy"]
	valAcc = history.history["val_accuracy"]

	# get the loss and the validation loss from the training history
	loss = history.history["loss"]
	valLoss = history.history["val_loss"]

	# get the range of epoch and build a pyplot figure
	epochsRange = range(numEpochs)
	fig = plt.figure(figsize=(12, 6))

	# using subplots plot the accuracy and loss plots
	plt.subplot(1, 2, 1)
	plt.plot(epochsRange, acc, label="train accuracy")
	plt.plot(epochsRange, valAcc, label="validataion accuracy")
	plt.title("Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend(loc="lower right")
	plt.subplot(1, 2, 2)
	plt.plot(epochsRange, loss, label="train loss")
	plt.plot(epochsRange, valLoss, label="validataion loss")
	plt.title("Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend(loc="upper right")
	fig.tight_layout()
	
	# save the learning curve plot to disk
	plt.savefig(learningPath)

def create_mask(predMask):
	# create the mask from prediction
	predMask = tf.argmax(predMask, axis=-1)
	predMask = predMask[..., tf.newaxis]

	# return the mask
	return predMask[0]

def show_predictions(unetModel, dataset, outputPath, num):
	# plot the prediction and save them to disk
	count = 0
	for (image, mask) in dataset.take(num):
		predMask = unetModel.predict(image)
		display(imagePath=os.path.join(outputPath, f"inference{count}.png"),
			displayList=[image[0], mask[0], create_mask(predMask)])
		count += 1
		