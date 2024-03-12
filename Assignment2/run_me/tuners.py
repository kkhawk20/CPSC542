from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import keras_tuner as kt
import keras
import numpy as np
import os
import time

save_dir = os.path.join(os.path.dirname(__file__), 'outputs')
model_checkpoint_path = os.path.join(save_dir, 'best_model.h5')
# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

def dice_coeff(y_true, y_pred, smooth=1e-6):
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

# Define a combined loss
def bce_dice_loss(y_true, y_pred):
    bce = BinaryCrossentropy(from_logits=False)
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

def build_model(hp):
    input_size=(256,256,3)

    # Utilizing a pre-trained VGG16 model as the encoder part of U-NET++
    vgg16 = VGG16(include_top = False, weights = 'imagenet', 
            input_shape = input_size)
    
    for layer in vgg16.layers: # Freeze layers
        layer.trainable = False

    # Encoder - VGG16
    vgg_outputs = [vgg16.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
    c1, c2, c3, c4, c5 = vgg_outputs
    
    # Bottleneck
    bottleneck = c5

    # Decoder
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(bottleneck)
    merge6 = concatenate([c4, up6], axis=3)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    merge7 = concatenate([c3, up7], axis=3)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    merge8 = concatenate([c2, up8], axis=3)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    merge9 = concatenate([c1, up9], axis=3)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, 1, activation = 'sigmoid')(c9)

    model = Model(inputs=vgg16.input, outputs=outputs)

    lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4, 1e-5])

    # model.compile(optimizer = Adam(learning_rate = lr), 
    #             loss = bce_dice_loss, 
    #             metrics = ['accuracy', dice_coeff])
    # Tuner found the best learning rate: 0.01
    # Average IoU: 0.18527531623840332, Average Dice: 0.18527531623840332     
    
    # Testing out different optimizers...
    model.compile(optimizer = RMSprop(learning_rate = lr), 
                loss = bce_dice_loss, 
                metrics = ['accuracy', dice_coeff])

    return model

def unet(train_gen, val_gen, test_gen):

    tuner = kt.RandomSearch(build_model,
                            objective = 'val_accuracy', 
                            max_trials = 10,
                            overwrite = True, # Needed to overwrite previous saves due to issues
                            directory = save_dir, 
                            project_name = 'Assignment2',
                            )

    tuner.search(train_gen, epochs = 10, 
                validation_data = val_gen, 
                verbose = 1)

     # Correctly get the best model/hp and evaluate it
    best_hp = tuner.get_best_hyperparameters()[0]
    
    best_model = tuner.get_best_models(num_models = 1)[0]  # Select the first model from the list of best models

    # Utilizing checkpoint for saving model and early stopping to minmize loss 
    checkpoint = ModelCheckpoint(filepath = model_checkpoint_path, 
                                monitor='val_accuracy', 
                                verbose=1, save_best_only=True, 
                                save_weights_only=False, mode='auto', 
                                save_freq='epoch')

    early = EarlyStopping(monitor='val_accuracy', patience = 20, 
                        verbose=1, mode='auto')

    history = best_model.fit(train_gen, validation_data = val_gen, 
                        callbacks = [checkpoint, early], 
                        epochs = 500,
                        verbose = 2)

    # Save best model's summary to file
    output_file_path = os.path.join(save_dir, 'best_model_summary.txt')
    with open(output_file_path, 'w') as f:
        def print_to_file(text):
            print(text, file=f)
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))
        print_to_file(f"\nTuner found the best learning rate: {best_hp.get('learning_rate'):.2f}")
 
    return best_model, history
