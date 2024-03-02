from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import keras
import numpy as np
import os

def unet(train_gen, val_gen, test_gen):

    input_size=(256,256,3)
    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * np.exp(-0.1)

    callbacks = [
        EarlyStopping(patience = 10, verbose = 1),
        ModelCheckpoint(filepath = os.path.join(save_dir,'unet_KH.h5'), verbose = 1, save_best_only = True),
        LearningRateScheduler(scheduler, verbose = 1)
    ]

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    outputs = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # ################ Simple CNN, not used
    # inputs = Input(input_size)
    # # Contracting Path
    # c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    # p1 = MaxPooling2D((2, 2))(c1)
    # c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    # p2 = MaxPooling2D((2, 2))(c2)

    # # Bottleneck
    # bn = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    # bn = Dropout(0.5)(bn)

    # # Expansive Path
    # u1 = UpSampling2D((2, 2))(bn)
    # merge1 = concatenate([c2, u1], axis=3)
    # c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
    # u2 = UpSampling2D((2, 2))(c3)
    # merge2 = concatenate([c1, u2], axis=3)
    # c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge2)
    
    # outputs = Conv2D(1, (1, 1), activation='sigmoid')(c4)
    #################

    model = Model(inputs = [inputs], outputs = [outputs])

    model.compile(optimizer = Adam(learning_rate = 1e-4), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])

    history = model.fit(train_gen, validation_data = val_gen, 
                        callbacks = callbacks, epochs = 500)

      # Load the best model
    best_model = load_model(model_checkpoint_path)

    # Save best model's summary to file
    output_file_path = os.path.join(save_dir, 'best_model_summary.txt')
    with open(output_file_path, 'w') as f:
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))

    return best_model, history