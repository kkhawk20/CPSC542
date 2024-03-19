import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import keras.backend as K
import warnings
import tensorflow as tf
from matplotlib.colors import ListedColormap
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
warnings.filterwarnings('ignore')

def calculate_iou(y_true, y_pred, smooth=1e-6):
    #print("y_true shape:", y_true.shape)
    #print("y_pred shape:", y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    
    # Expand the dimensions of y_pred to match y_true
    y_pred = tf.expand_dims(y_pred, -1)
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2])
    #print("Intersection shape:", intersection.shape)

    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2]) - intersection
    #print("Union calculation passed")    
    iou = (intersection + smooth) / (union + smooth)
    
    return tf.reduce_mean(iou)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    # Ensure y_pred has the same dimensions as y_true
    y_pred = tf.expand_dims(y_pred, -1)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def model_output_loss(output):
    return tf.reduce_mean(output)
 
def make_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    # Create a Gradcam object
    gradcam = Gradcam(model, model_modifier=None, clone=True)
    
    # If pred_index is provided, use it to create a loss function that isolates the output at the pred_index
    if pred_index is not None:
        loss = lambda output: output[..., pred_index]
    else:
        # If not provided, we assume binary segmentation and use the mean of the output
        loss = model_output_loss

    # Generate heatmap with GradCAM
    heatmap = gradcam(loss,
                      seed_input=image,
                      penultimate_layer=-1)  # -1 automatically infers the last convolutional layer
    heatmap = normalize(heatmap)
    
    return heatmap

def make_occlusion_map(model, image, patch_size=15):
    # Create Saliency object
    saliency = Saliency(model,
                        model_modifier=ReplaceToLinear(),
                        clone=False)
    
    # Define loss function for occlusion
    loss = lambda output: K.mean(output)
    
    # Generate occlusion sensitivity map
    occlusion_map = saliency(loss,
                             seed_input=image,
                             keepdims=True)
    
    occlusion_map = normalize(occlusion_map)
    return occlusion_map

# def overlay_segmentation(image_path, true_mask_path, model, save_dir):
#     custom_cmap = ListedColormap(['black', 'white'])

#     original_image = load_img(image_path, target_size=(256, 256))
#     numpy_image = img_to_array(original_image) / 255.0
#     predictions = model.predict(np.expand_dims(numpy_image, axis=0))
#     predicted_mask = predictions.squeeze() 

#     predicted_mask_binary = (predicted_mask > 0.5).astype(np.float32)
    
#     # Load the actual mask
#     true_mask = img_to_array(load_img(true_mask_path, target_size=(256, 256), color_mode="grayscale")) / 255.0

#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 3, 1)
#     plt.title("Original Image")
#     plt.imshow(original_image)
#     plt.axis('off')
    
#     plt.subplot(1, 3, 2)
#     plt.title("Actual Mask")
#     plt.imshow(true_mask, cmap='gray')  # Assuming true mask is in grayscale
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title("Predicted Mask")
#     plt.imshow(original_image)
#     plt.imshow(predicted_mask_binary, alpha=1, cmap=custom_cmap)  # Overlay predicted mask
#     plt.axis('off')
    
#     # Save the figure to the specified directory
#     image_name = os.path.basename(image_path)
#     save_path = os.path.join(save_dir, f"overlay_{image_name}")
#     plt.savefig(save_path)
#     plt.close()  # Close the plot to free memory

#     # return original_image, true_mask, predicted_mask_binary

def plot_and_save_metrics(history, save_dir):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label = 'Training Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coeff'], label = 'training Dice coeff')
    plt.plot(history.history['val_dice_coeff'], label = "Validation Dice Coeff")
    plt.title('Model Dice Coeff')
    plt.ylabel('Dice Coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_metrics.jpg'))
    plt.close()  # Close the plot after saving to free up memory

# Chat code
def plot_and_save_combined_images(original_image, true_mask, predicted_mask_binary, gradcam_heatmap, occlusion_map, save_path):
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title('Actual Mask')
    ax[1].axis('off')

    ax[2].imshow(predicted_mask_binary, cmap='gray')
    ax[2].set_title('Predicted Mask')
    ax[2].axis('off')

    ax[3].imshow(original_image)
    ax[3].imshow(gradcam_heatmap, cmap='jet', alpha=0.5)
    ax[3].set_title('Grad-CAM')
    ax[3].axis('off')

    ax[4].imshow(original_image)
    ax[4].imshow(occlusion_map, cmap='jet', alpha=0.5)
    ax[4].set_title('Occlusion Map')
    ax[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# def model_eval(history, model):
    
#     last_conv_layer_name = 'conv2d_7'
#     # This block of code is going to load in the trained model, and overlay
#     # The predicted image segmentation

#     test_images_dir = os.path.join(os.path.dirname(__file__), 'Data_new/Test/Images')
#     save_dir = os.path.join(os.path.dirname(__file__), 'outputs')  # where to save overlay images
#     os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist

#     image_paths = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir)]
    
#     ious = []
#     dices = []
    
#     # This is going to calc the IoU and Dice metrics
#     for image_path in image_paths:
#         original_image = load_img(image_path, target_size=(256, 256))
#         numpy_image = img_to_array(original_image)  / 255.0
#         input_image = np.expand_dims(numpy_image, axis=0)

#         base_name = os.path.basename(image_path)
#         name_without_ext = os.path.splitext(base_name)[0]
#         mask_filename = name_without_ext + "_mask.jpg"  # Ensure this matches your mask file extensions
#         mask_path = os.path.join(os.path.dirname(image_path).replace('Images', 'Masks'), mask_filename)

#         if not os.path.exists(mask_path):
#             print(f"Mask file does not exist: {mask_path}")
#             continue  # Skip this image-mask pair and move to the next

#         true_mask = load_img(mask_path, target_size=(256, 256), color_mode="grayscale")
#         true_mask = img_to_array(true_mask)
#         true_mask = true_mask / 255.0
#         print("True Mask Shape :", true_mask.shape)
#         # true_mask = np.squeeze(true_mask)
#         # print("True Mask Shape after squeeze:", true_mask.shape)

#         predictions = model.predict(input_image)
#         predicted_mask = tf.squeeze(predictions)  # This should already give you a shape of (256, 256)
#         predicted_mask_binary = tf.cast(predicted_mask > 0.5, dtype=tf.float32)
        
#         iou = calculate_iou(true_mask, predicted_mask_binary)
#         dice = dice_coefficient(true_mask, predicted_mask_binary)
#         ious.append(iou)
#         dices.append(dice)


#         # Assume we already have `true_mask_path` and `predicted_mask_binary` from the model
#         original_image, true_mask, predicted_mask_binary = overlay_segmentation(image_path, mask_path, predicted_mask_binary)
        
#         # Generate Grad-CAM and occlusion maps
#         gradcam_heatmap = make_gradcam_heatmap(model, numpy_image, last_conv_layer_name)
#         occlusion_map = make_occlusion_map(model, numpy_image)

#         # Now combine all the images horizontally
#         combined_image = np.hstack([
#             np.array(original_image),
#             np.array(true_mask, cmap='gray'),
#             np.array(predicted_mask_binary, cmap='gray'),
#             gradcam_heatmap,
#             occlusion_map
#         ])

#         # Convert combined image to uint8 if necessary, rescale to 0-255 if it's in float format
#         if combined_image.dtype != np.uint8:
#             combined_image = (255 * combined_image).astype(np.uint8)

#         # Save combined image
#         plt.figure(figsize=(20, 4))
#         plt.imshow(combined_image)
#         plt.title(f"Combined Visualizations for {os.path.basename(image_path)}")
#         plt.axis('off')
#         save_path = os.path.join(save_dir, f"combined_visualization_{i}.png")
#         plt.savefig(save_path)
#         plt.close()

#         # for i, image_path in enumerate(image_paths[:5]):
#         #     overlay_segmentation(image_path, mask_path, model, save_dir)
#         #     gradcam_heatmap = make_gradcam_heatmap(image_path, save_dir, model, numpy_image, last_conv_layer_name)
#         #     occlusion_map = make_occlusion_map(model, numpy_image)

#         #     plt.figure(figsize=(20, 4))

#         #     # Original Image
#         #     plt.subplot(1, 5, 1)
#         #     plt.title("Original Image")
#         #     plt.imshow(original_image)
#         #     plt.axis('off')
            
#         #     # True Mask
#         #     plt.subplot(1, 5, 2)
#         #     plt.title("True Mask")
#         #     plt.imshow(true_mask.squeeze(), cmap='gray')
#         #     plt.axis('off')

#         #     # Predicted Mask
#         #     plt.subplot(1, 5, 3)
#         #     plt.title("Predicted Mask")
#         #     plt.imshow(predicted_mask_binary, cmap='gray')
#         #     plt.axis('off')

#         #     # Grad-CAM Heatmap
#         #     plt.subplot(1, 5, 4)
#         #     plt.title("Grad-CAM")
#         #     plt.imshow(original_image)
#         #     plt.imshow(gradcam_heatmap[0], cmap='jet', alpha=0.5)
#         #     plt.axis('off')

#         #     # Occlusion Map
#         #     plt.subplot(1, 5, 5)
#         #     plt.title("Occlusion Map")
#         #     plt.imshow(original_image)
#         #     plt.imshow(occlusion_map[0], cmap='jet', alpha=0.5)
#         #     plt.axis('off')

#         #     # Save the figure to the specified directory
#         #     plt.tight_layout()
#         #     plt.savefig(os.path.join(save_dir, f"visualizations_{i}.png"))
#         #     plt.close()

#     avg_iou = np.mean(ious)
#     avg_dice = np.mean(dices)
#     output_file_path = os.path.join(save_dir, 'best_model_summary.txt')
#     with open(output_file_path, 'a') as f:
#         def print_to_file(text):
#             print(text, file=f)
#         print_to_file(f"\nAverage IoU: {avg_iou}, Average Dice: {avg_dice}")

#     # This is going to save a graph that will show val&test acc&loss
#     plot_and_save_metrics(history, save_dir)

def model_eval(history, model):
    test_images_dir = os.path.join(os.path.dirname(__file__), 'Data_new/Test/Images')
    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')  # where to save overlay images
 
    last_conv_layer_name = 'conv2d_7'
    num_images=5
    image_paths = sorted([os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir)])

    ious = []
    dices = []

    for i, image_path in enumerate(image_paths):
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        true_mask_path = os.path.join(os.path.dirname(image_path).replace('Images', 'Masks'), name_without_ext + "_mask.jpg")

        original_image = load_img(image_path, target_size=(256, 256))
        numpy_image = img_to_array(original_image) / 255.0
        numpy_image = np.expand_dims(numpy_image, axis=0)

        predictions = model.predict(numpy_image)
        predicted_mask = (predictions.squeeze() > 0.5).astype(np.uint8)

        true_mask = img_to_array(load_img(true_mask_path, target_size=(256, 256), color_mode='grayscale')) / 255.0
        true_mask = true_mask.squeeze()

        # Calculate IoU and Dice coefficients
        iou = calculate_iou(true_mask, predicted_mask)
        dice = dice_coefficient(true_mask, predicted_mask)
        ious.append(iou.numpy())  # .numpy() converts from tf.Tensor to numpy array
        dices.append(dice.numpy())

        if i < num_images:
            gradcam_heatmap = make_gradcam_heatmap(model, numpy_image, last_conv_layer_name)
            occlusion_map = make_occlusion_map(model, numpy_image)

            save_path = os.path.join(save_dir, f'combined_visualization_{i}.jpg')
            plot_and_save_combined_images(original_image, true_mask, predicted_mask, gradcam_heatmap, occlusion_map, save_path)

    # Save average IoU and Dice
    avg_iou = np.mean(iou)
    avg_dice = np.mean(dice)
    with open(os.path.join(save_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f'Average IoU: {avg_iou}\n')
        f.write(f'Average Dice: {avg_dice}\n')

    # Plot and save training/validation loss and Dice coefficient graphs
    plot_and_save_metrics(history, save_dir)
