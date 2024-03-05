from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import keras
import numpy as np
import os
import time

def unet(train_gen, val_gen, test_gen):

    input_size=(256,256,3)
    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    model_checkpoint_path = os.path.join(save_dir, 'unet_KH.h5')

    # Utilizing a pre-trained VGG16 model as the encoder part of U-NET++
    vgg16 = VGG16(include_top = False, weights = 'imagenet', 
            input_shape = input_size)
    for layer in vgg16.layers: # Freeze layers
        layer.trainable = False

    layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])

    inputs = vgg16.input

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

    model.compile(optimizer = Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * np.exp(-0.1)

    callbacks = [
        EarlyStopping(patience = 10, verbose = 1),
        ModelCheckpoint(filepath = os.path.join(save_dir,'unet_KH.h5'), 
        verbose = 1, save_best_only = True),
        LearningRateScheduler(scheduler, verbose = 1)
    ]
    # Tracking training time for LOLs
    start_time = time.time()

    history = model.fit(train_gen, validation_data = val_gen, 
                        callbacks = callbacks, epochs = 500)

    end_time = time.time()

    # Calculate and print the training duration
    training_duration = end_time - start_time
    hours, rem = divmod(training_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")

    # Load the best model
    best_model = load_model(model_checkpoint_path)

    # Save best model's summary to file
    output_file_path = os.path.join(save_dir, 'best_model_summary.txt')
    with open(output_file_path, 'w') as f:
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"Training took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")

    return best_model, history
