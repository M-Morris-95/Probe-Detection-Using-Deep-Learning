The goal is to create a deep learning system to correctly identify and locate a bounding box around an ultrasonic thickness measurement probe attached to a drone. We are provided with 308 labelled images for training, validation, and testing. Each label contains an x-coordinate, y-coordinate, width, and height of the bounding box. There are no images that do not contain the probe.

 I used VGG16 and MobileNetV2 as feature extractors for transfer learning, with an additional prediction head.
