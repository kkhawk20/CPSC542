from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from keras_tuner import Objective
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
    print("y_true shape:", tf.shape(y_true))
    print("y_pred shape:", tf.shape(y_pred))
    bce = BinaryCrossentropy(from_logits=False)
    # return bce(y_true, y_pred)
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

# Testing out a different pre-trained encoder architecture
def build_unet_resnet50(input_shape=(256, 256, 3)):
    # Load ResNet50 as the encoder with pretrained weights and without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Encoder: Capture skip connections (outputs of interest) from ResNet50
    # These layer names are specific to ResNet50 architecture
    skip_layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
    encoder_outputs = [base_model.get_layer(name).output for name in skip_layer_names]

    # Decoder: Construct the decoder layers with skip connections and upsampling
    decoder_filters = [256, 128, 64, 32]
    x = base_model.output
    for i, f in enumerate(decoder_filters):
        # Upsampling (you can also experiment with Conv2DTranspose)
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        # Skip connection
        if i < len(encoder_outputs):  # Safety check
            x = concatenate([x, encoder_outputs[-(i + 1)]])

        # Convolutional layers
        x = Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(f, (3, 3), activation='relu', padding='same')(x)

    # Output layer for binary segmentation
    output = Conv2D(1, (1, 1), activation='sigmoid')(x)

    # Final model
    model = Model(inputs=base_model.input, outputs=output)

    return model

# This is a vgg16 model as the encoder for the u-net
# Kinda sucks, trying other models now...
# Tuner found the best learning rate: 0.00001
# Average IoU: 0.18527531623840332, Average Dice: 0.18527531623840332
def build_model(hp):
    input_size=(256,256,3)

    # Utilizing a pre-trained VGG16 model as the encoder part of U-NET++
    vgg16 = VGG16(include_top = False, weights = 'imagenet', 
            input_shape = input_size)
    
    for layer in vgg16.layers[:15]: # Freeze top 15 layers (leave 4)
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
    c6 = Dropout(0.5)(c6) #Adding dropout to see if this helps
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
    
    # Done with model, compile now
    lr = hp.Choice('learning_rate', values = [1e-3, 1e-4, 1e-5])

    model.compile(optimizer = Adam(learning_rate = lr), 
                    # loss = bce_dice_loss, # Removing this for debugging stuff
                  loss = tf.keras.losses.BinaryCrossentropy()
                    metrics = [dice_coeff])
    
    return model

def unet(train_gen, val_gen, test_gen):

    tuner = kt.RandomSearch(build_model,
                            objective = Objective('val_dice_coeff', direction = "max"), 
                            max_trials = 10,
                            overwrite = True, # Needed to overwrite previous saves due to issues
                            directory = save_dir, 
                            project_name = 'Assignment2',
                            )

    tuner.search(train_gen, epochs = 5, 
                validation_data = val_gen, 
                verbose = 2)

     # Correctly get the best model/hp and evaluate it
    best_hp = tuner.get_best_hyperparameters()[0]
    
    best_model = tuner.get_best_models(num_models = 1)[0]  # Select the first model from the list of best models

    # Utilizing checkpoint for saving model and early stopping to minmize loss 
    checkpoint = ModelCheckpoint(filepath = model_checkpoint_path, 
                                monitor='val_dice_coeff', 
                                verbose=1, save_best_only=True, 
                                save_weights_only=False, mode='max', 
                                save_freq='epoch')

    early = EarlyStopping(monitor='val_dice_coeff', patience = 15, 
                        verbose=1, mode='max')
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                     patience=10,
                                     verbose = 1,
                                     min_lr = 1e-6)

    history = best_model.fit(train_gen, validation_data = val_gen, 
                        callbacks = [checkpoint, early, lr_scheduler], 
                        epochs = 500,
                        verbose = 2)

    # Save best model's summary to file
    output_file_path = os.path.join(save_dir, 'best_model_summary.txt')
    with open(output_file_path, 'w') as f:
        def print_to_file(text):
            print(text, file=f)
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))
        print_to_file(f"\nTuner found the best learning rate: {best_hp.get('learning_rate'):.4f}")
 
    return best_model, history
