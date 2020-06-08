# organize imports
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.applications import imagenet_utils, vgg16, mobilenet
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.mobilenet import preprocess_input
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import tensorflowjs as tfjs

# process an image to be mobilenet friendly
def process_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	pImg = image.img_to_array(img)
	#pImg = cv2.cvtColor(np.float32(pImg), cv2.COLOR_BGR2RGB)
	
	#pImg =preprocess_input(img)
	pImg = np.expand_dims(pImg, axis=0)
	
	return pImg

# main function
if __name__ == '__main__':

	# path to test image
	test_img_path = "/home/hoangntbn/Desktop/demo_deeplearning/c3.jpg"

	# process the test image
	pImg = process_image(test_img_path)
	#cv2.imshow('a', (pImg).reshape(224, 224, 3))
	# cv2.waitKey(0)

	# model = vgg16.VGG16()
	model = load_model('/home/hoangntbn/Desktop/demo_deeplearning/model_keras/custom.h5')
	#model=load_model('/home/hoangntbn/Desktop/demo_deeplearning/model_keras/mobilenet.h5')
	#model=load_model('/home/hoangntbn/Desktop/demo_deeplearning/model_keras/resnet50.h5')
	# make predictions on test image using mobilenet
	prediction = model.predict(pImg)

	prediction=prediction.reshape(10)

	# obtain the top-5 predictions
	# results = imagenet_utils.decode_predictions(prediction)
	# print(prediction.shape)


	# # a=np.array

	# classes1=np.argsort(prediction)
	# print(classes1)
	# print(prediction)

	
	# classes = np.argmax(prediction)
	# print(classes)

	# # # convert the mobilenet model into tf.js model
	save_path = "/home/hoangntbn/Desktop/demo_deeplearning/models1/vgg_v2"
	tfjs.converters.save_keras_model(model, save_path)
	print("done")