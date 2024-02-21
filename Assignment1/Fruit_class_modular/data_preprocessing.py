# data_preprocessing.py
# Purpose is to load in TTV data, augment training set

import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

def load_data():
    # Ensuring the data is being accessed! 
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'Data')

    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')
    val_dir = os.path.join(data_dir, 'Val')

    # Pre-setting the known image size and set batch size
    image_size = (300,300)
    batch_size = 32

    # Adding preprocessing for augmentation!! 
    # Increasing diversity, etc. 
    trData = ImageDataGenerator(
        rotation_range = 180,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        zoom_range = 0.2,
        fill_mode = 'nearest'
    )
    # Reading in training DS with categorical classes
    train_ds = trData.flow_from_directory(directory = train_dir, 
                                        target_size = image_size, 
                                        class_mode = 'categorical', 
                                        batch_size = batch_size)

    # Reading in testing DS with categorical classes
    teData = ImageDataGenerator()
    test_ds = teData.flow_from_directory(directory = test_dir, 
                                        target_size = image_size, 
                                        class_mode = 'categorical', 
                                        batch_size = batch_size)

    # Reading in validation DS with categorical classes
    valData = ImageDataGenerator()
    val_ds = valData.flow_from_directory(directory = val_dir, 
                                        target_size = image_size, 
                                        class_mode = 'categorical', 
                                        batch_size = batch_size)

    return train_ds, test_ds, val_ds