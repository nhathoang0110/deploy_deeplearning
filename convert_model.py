from keras.applications import VGG16
import argparse
from tensorflow.keras.models import load_model
from tensorflowjs import tfjs

model = load_model('/home/hoangntbn/Desktop/20192/DL/test/Bai30-tensorflowJS/DL_vgg16_extra.h5')
# model.save('mobile.h5')
tfjs.converters.save_keras_model(model, '/home/hoangntbn/Desktop/20192/DL/test/Bai30-tensorflowJS/model1/')	