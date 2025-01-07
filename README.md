The goal is to create a deep learning system to correctly identify and locate a bounding box around an ultrasonic thickness measurement probe attached to a drone. We are provided with 308 labelled images for training, validation, and testing. Each label contains an x-coordinate, y-coordinate, width, and height of the bounding box. There are no images that do not contain the probe.

 I used VGG16 and MobileNetV2 as feature extractors for transfer learning, with an additional prediction head.


run file 'run_model.py' to generate predictions using trained model

need to set on lines 11 and 12
IMAGES_ROOT = 'probe_dataset/probe_images'  # this is where your pictures are saved
MODEL = 'VGG'                               # this is the model you want to run, either "VGG" or "mobilenet"

It will save a file 'my_predictions.json' which contains all the predictions in the same format as they were given.


code.ipynb is the development environment and shows all the steps including hyper parameter tuning.
metrics.py is where MAP and IoU are calculated. 
model_functions.py contains the model definition and code to make predictions
