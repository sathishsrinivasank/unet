# import the necessary packages
from tensorflow.keras import layers
import tensorflow as tf

def double_conv_block(x, nFilters):
	# apply twich conv and relu activation to the input tensor
	x = layers.Conv2D(
		nFilters,
		3,
		padding="same",
		activation="relu",
		kernel_initializer="he_normal")(x)
	x = layers.Conv2D(
		nFilters,
		3,
		padding="same",
		activation="relu",
		kernel_initializer="he_normal")(x)

	# return the processed tensor
	return x

def downsample_block(x, nFilters):
	# pass the tensor through the double conv block
	f = double_conv_block(x, nFilters)

	# pass the processed tensor through MaxPool and Dropout layer
	p = layers.MaxPool2D(2)(f)
	p = layers.Dropout(0.3)(p)

	# return the conv output and the dropout output
	return (f, p)

def upsample_block(x, convFeatures, nFilters):
	# upsample the input tensor and concatenate it with the
	# conv features
	x = layers.Conv2DTranspose(nFilters, 3, 2, padding="same")(x) 
	x = layers.concatenate([x, convFeatures])
	
	# apply dropout to the concatenated tensor and pass it through the
	# double conv block
	x = layers.Dropout(0.3)(x)
	x = double_conv_block(x, nFilters)

	# return the processed tensor
	return x

def build_unet_model():
	# create the input layer of the u-net architecture
	inputs = layers.Input(shape=(128,128,3))

	# encoder: contracting path - downsample
	# 1 - downsample
	(f1, p1) = downsample_block(inputs, 64)
	# 2 - downsample
	(f2, p2) = downsample_block(p1, 128)
	# 3 - downsample
	(f3, p3) = downsample_block(p2, 256)
	# 4 - downsample
	(f4, p4) = downsample_block(p3, 512)

	# 5 - bottleneck
	bottleneck = double_conv_block(p4, 1024)

	# decoder: expanding path - upsample
	# 6 - upsample
	u6 = upsample_block(bottleneck, f4, 512)
	# 7 - upsample
	u7 = upsample_block(u6, f3, 256)
	# 8 - upsample
	u8 = upsample_block(u7, f2, 128)
	# 9 - upsample
	u9 = upsample_block(u8, f1, 64)

	# build the output layer
	outputs = layers.Conv2D(
		3,
		1,
		padding="same",
		activation="softmax")(u9)

	# unet model with Keras Functional API
	unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

	# return the unet model
	return unet_model