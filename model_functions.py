import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape, dropout_percent, base_model_type, dense_units_list, output_activation, learning_rate):
    """
    Function to build the model based on hyperparameters.
    
    Parameters:
    - input_shape: Shape of the input images (height, width, channels).
    - dropout_percent: Dropout percentage.
    - base_model_type: 'mobilenetv2' or 'vgg16' for the base model.
    - dense_units_list: List of integers representing the number of units in the dense layers.
    - output_activation: Activation function for output ('sigmoid' or 'linear').
    - learningrate: Learning rate.
    
    Returns:
    - A compiled Keras model.
    """
    
    # Load the base model (MobileNetV2 or VGG16) based on hyperparameter selection
    if base_model_type == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False,
                                                       weights='imagenet')
    elif base_model_type == 'vgg16':
        base_model = tf.keras.applications.VGG16(input_shape=input_shape,
                                                 include_top=False,
                                                 weights='imagenet')
    else:
        raise ValueError("Invalid base model type. Choose 'mobilenetv2' or 'vgg16'.")

    # Freeze the base model to avoid training its weights
    base_model.trainable = False
    
    # Build the model
    input_layer = tf.keras.Input(shape=input_shape)  # Input layer
    x = base_model(input_layer, training=False)  # Base model
    x = layers.Dropout(dropout_percent)(x)  # Dropout after base model
    x = layers.Flatten()(x)  # Flatten the output
    
    # Add dense layers with units from dense_units_list
    for units in dense_units_list:
        x = layers.Dropout(dropout_percent)(x)
        x = layers.Dense(units, activation="relu")(x)
    
    # Output layer with the specified number of output units and activation function
    outputs = layers.Dense(4, activation=output_activation)(x)
    
    # Define the model
    model = models.Model(input_layer, outputs)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
                ) 

    return model

def compute_outputs(model, inputs, image_height_width=224):
    '''
    Compute the model predictions for a 224 x 224 image.
    '''
    output = np.asarray(model(inputs))
    output = output*image_height_width
    output[output>image_height_width] = image_height_width
    output[output<0] = 0
    return output

def convert_prediction(prediction, AR_X = 640/224 , AR_Y = 400/224):
    ''' 
    Adjust the predictions for a 660x400 image
    '''
    prediction[..., 0]*= AR_X
    prediction[..., 2]*= AR_X
    prediction[..., 1]*= AR_Y
    prediction[..., 3]*= AR_Y

    return prediction
