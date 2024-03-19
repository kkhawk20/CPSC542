# data_preprocessing.py
# Purpose is to load in TTV data

import os
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

def load_data():
    # Ensuring the data is being accessed!
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'Data_new')

    # Define the directories for images and masks
    train_image_dir = os.path.join(data_dir, 'Train', 'Images')
    train_mask_dir = os.path.join(data_dir, 'Train', 'Masks')
    val_image_dir = os.path.join(data_dir, 'Val', 'Images')
    val_mask_dir = os.path.join(data_dir, 'Val', 'Masks')
    test_image_dir = os.path.join(data_dir, 'Test', 'Images')
    test_mask_dir = os.path.join(data_dir, 'Test', 'Masks')

    # Pre-setting the known image size and set batch size
    image_size = (256, 256)
    batch_size = 32

    # Create instances of the SegmentationDataGenerator for each dataset
    train_gen = SegmentationDataGenerator(train_image_dir, train_mask_dir, batch_size, image_size, augment = True)
    val_gen = SegmentationDataGenerator(val_image_dir, val_mask_dir, batch_size, image_size)
    test_gen = SegmentationDataGenerator(test_image_dir, test_mask_dir, batch_size, image_size)

    return train_gen, val_gen, test_gen

class SegmentationDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, image_size, augment = False):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.mask_paths = [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)]
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        if self.augment:
            self.image_data_gen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip = True,
                fill_mode='nearest'
            )
            self.mask_data_gen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip = True,
                fill_mode='nearest'
            )
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size

    # def grayscale_to_color(self, mask):
    #     # Assuming the mask is a single-channel image with pixel values 0 or 255
    #     color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)  # create a 3-channel RGB image
    #     color_mask[mask == 0] = [255, 0, 0]  # red color for class 0
    #     color_mask[mask == 255] = [0, 255, 0]  # green color for class 1
    #     return color_mask

    def __getitem__(self, idx):

        def augment_masks(batch_masks, mask_data_gen):
            # Assuming batch_masks is a batch of your grayscale masks
            # Temporarily expand the mask to 3D to satisfy ImageDataGenerator requirements
            batch_masks_expanded = np.expand_dims(batch_masks, axis=-1)
            augmented_masks = np.array([mask_data_gen.random_transform(mask) for mask in batch_masks_expanded])
            # Squeeze the masks back to 2D
            augmented_masks_squeezed = np.squeeze(augmented_masks, axis=-1)
            return augmented_masks_squeezed

        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = np.array([cv2.resize(cv2.imread(file_path), self.image_size) for file_path in batch_image_paths])
        batch_masks = np.array([cv2.resize(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), self.image_size) for file_path in batch_mask_paths])

        if self.augment:
            # Augment images
            batch_images = np.array([self.image_data_gen.random_transform(img) for img in batch_images])
            # Augment masks - Ensure this method is defined to handle grayscale masks correctly
            batch_masks = augment_masks(batch_masks, self.mask_data_gen)
        
        batch_masks = (batch_masks > 127).astype(np.float32)
        batch_masks = np.expand_dims(batch_masks, axis=-1)  # Ensure masks are properly shaped for the model

        # Normalization
        batch_images = batch_images.astype(np.float32) / 255.0 # Color
        # print(batch_images.shape)
        batch_masks = batch_masks.astype(np.float32) # Greyscale
        # print(batch_masks.shape)

        # print(f"Batch images shape: {batch_images.shape}, dtype: {batch_images.dtype}, min: {batch_images.min()}, max: {batch_images.max()}")
        # print(f"Batch masks shape: {batch_masks.shape}, dtype: {batch_masks.dtype}, min: {batch_masks.min()}, max: {batch_masks.max()}")

        return batch_images, batch_masks


