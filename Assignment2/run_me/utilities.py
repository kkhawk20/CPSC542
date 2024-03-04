import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import keras.backend as K
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

def calculate_iou(y_true, y_pred, smooth=1e-6):
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2])
    print("Intersection shape:", intersection.shape)

    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) - intersection
    print("Union calculation passed")    
    iou = (intersection + smooth) / (union + smooth)
    
    return tf.reduce_mean(iou)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

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

def plot_and_save_metrics(history, save_dir):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_metrics.jpg'))
    plt.close()  # Close the plot after saving to free up memory


def model_eval(history, model):
    
    # This block of code is going to load in the trained model, and overlay
    # The predicted image segmentation
    test_images_dir = os.path.join(os.path.dirname(__file__), 'Data_new/Test/Images')
    #model = load_model(os.path.join(os.path.dirname(__file__), 'outputs/unet_KH.h5'))
    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')  # Specify where to save overlay images
    os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist

    image_paths = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir)]
    
    # Assume test_images_dir contains pairs of image and mask files, or adjust accordingly
    ious = []
    dices = []
    
    for image_path in image_paths:
        original_image = load_img(image_path, target_size=(256, 256))
        numpy_image = img_to_array(original_image)
        input_image = np.expand_dims(numpy_image, axis=0) / 255.0

        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        mask_filename = name_without_ext + "_mask.jpg"  # Ensure this matches your mask file extensions
        mask_path = os.path.join(os.path.dirname(image_path).replace('Images', 'Masks'), mask_filename)

        if not os.path.exists(mask_path):
            print(f"Mask file does not exist: {mask_path}")
            continue  # Skip this image-mask pair and move to the next

        true_mask = load_img(mask_path, target_size=(256, 256), color_mode="grayscale")
        true_mask = img_to_array(true_mask)
        true_mask = true_mask / 255.0

        predictions = model.predict(input_image)
        predicted_mask = tf.argmax(predictions, axis=-1)
        predicted_mask = tf.squeeze(predicted_mask)  # This should already give you a shape of (256, 256)


        iou = calculate_iou(true_mask, predicted_mask)
        dice = dice_coefficient(true_mask, predicted_mask)
        ious.append(iou)
        dices.append(dice)

        # Visualize the first few images
        if len(ious) <= 5:  # number of images to visualize
            overlay_segmentation(image_path, model, save_dir)
    
    print(f"Average IoU: {np.mean(ious)}, Average Dice: {np.mean(dices)}")

    # This is going to save a graph that will show val&test acc&loss
    plot_and_save_metrics(history, save_dir)
