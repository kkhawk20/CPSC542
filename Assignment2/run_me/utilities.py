import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import keras.backend as K
import warnings
warnings.filterwarnings('ignore')

def calculate_iou(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def overlay_segmentation(image_path, model):
    original_image = load_img(image_path, target_size=(256, 256))
    numpy_image = img_to_array(original_image)
    input_image = np.expand_dims(numpy_image, axis=0)
    predictions = model.predict(input_image)

    predicted_mask = np.argmax(predictions, axis=-1)
    predicted_mask = np.squeeze(predicted_mask, axis=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title("Segmentation")
    plt.imshow(original_image)
    plt.imshow(predicted_mask, alpha=0.5, cmap='jet')  # Adjust transparency with alpha
    plt.show()

def model_eval(model_path, test_images_dir):
    model = load_model(model_path)
    image_paths = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir)]
    
    # Assume test_images_dir contains pairs of image and mask files, or adjust accordingly
    ious = []
    dices = []
    
    for image_path in image_paths:
        original_image = load_img(image_path, target_size=(256, 256))
        numpy_image = img_to_array(original_image)
        input_image = np.expand_dims(numpy_image, axis=0) / 255.0  # Normalize if your model expects this

        # Load corresponding mask
        # This part depends on how your dataset is structured
        mask_path = image_path.replace('images', 'masks')  # Adjust based on your directory structure
        true_mask = load_img(mask_path, target_size=(256, 256), color_mode="grayscale")
        true_mask = img_to_array(true_mask)
        true_mask = true_mask / 255.0  # Normalize mask to [0, 1] if needed

        predictions = model.predict(input_image)
        predicted_mask = np.argmax(predictions, axis=-1)
        predicted_mask = np.squeeze(predicted_mask, axis=0)

        iou = calculate_iou(true_mask, predicted_mask)
        dice = dice_coefficient(true_mask, predicted_mask)
        ious.append(iou)
        dices.append(dice)

        # Optionally, visualize the first few images
        if len(ious) <= 5:  # Adjust the number of images to visualize
            overlay_segmentation(image_path, model)
    
    print(f"Average IoU: {np.mean(ious)}, Average Dice: {np.mean(dices)}")

# Example usage
# model_eval('path_to_your_model.h5', 'path_to_your_test_images')
