import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import ast
import json
from model_functions import *

IMAGES_ROOT = 'probe_dataset/probe_images'  # this is where your pictures are saved
MODEL = 'mobilenet'                               # this is the model you want to run, either VGG or mobilenet

if MODEL == 'VGG':
    run = 39
else:
    run = 14

pixel_max = 255              # max brightness in the input images, used for normalisation
input_shape = (224, 224, 3)  # Standard input size for MobileNetV2 and VGG16.
image_height_width = 224     # height and width used for MobileNetV2 and VGG16.

images = []
images_tensor = []
file_names = os.listdir(IMAGES_ROOT)
for file in file_names:
    image  = cv2.imread(f'{IMAGES_ROOT}/{file}', cv2.IMREAD_GRAYSCALE)
    images.append(image)
    resized_image = cv2.resize(image, (image_height_width, image_height_width))
    three_channel_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    images_tensor.append(three_channel_image)

inputs_tensor = tf.convert_to_tensor(np.asarray(images_tensor).astype(np.float32), dtype=tf.float32)/pixel_max

hyperparams = pd.read_csv('hyperparams.csv', index_col=0)


base_model_type = str(hyperparams.loc[run, 'base_model_type'])
dense_units_list = ast.literal_eval(hyperparams.loc[run, 'dense_units_list'])
dropout_percent = float(hyperparams.loc[run, 'dropout_percent'])
output_activation = str(hyperparams.loc[run, 'output_activation'])
learning_rate = float(hyperparams.loc[run, 'learning_rate'])
batch_size = int(hyperparams.loc[run, 'batch_size'])

# Build the model with the specified hyperparameters
model = build_model(input_shape, dropout_percent, base_model_type, dense_units_list, output_activation, learning_rate)
model.load_weights(f'weights/{run}.h5')

outputs = convert_prediction(compute_outputs(model, inputs_tensor)).astype(int)

annotations = [{"bbox": output.tolist(), "image_id": id} for id, output in enumerate(outputs)]
images = [{"file_name": file_name, "height": 400, "id": id, "width": 640} for id, file_name in enumerate(file_names)]
data = {'annotations':annotations, 'images':images}
with open('my_predictions.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)