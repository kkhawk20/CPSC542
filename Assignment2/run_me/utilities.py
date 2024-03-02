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

def overlay_segmentation(image_path, model, save_dir):
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
    
    # Save the figure to the specified directory
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"overlay_{image_name}")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory

def model_eval():
    
    test_images_dir = os.path.join(os.path.dirname(__file__), 'Data_new/Test/Images')
    model = load_model(os.path.join(os.path.dirname(__file__), 'outputs/unet_KH.h5'))
    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')  # Specify where to save overlay images
    os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist

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
        mask_path = image_path.replace('Images', 'Masks')  # Adjust based on your directory structure
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
            overlay_segmentation(image_path, model, save_dir)
    
    print(f"Average IoU: {np.mean(ious)}, Average Dice: {np.mean(dices)}")
