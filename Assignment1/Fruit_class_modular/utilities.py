import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import load_img
from keras.models import load_model
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm

import warnings
warnings.filterwarnings('ignore')

def model_eval(hist):

    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'Data')
    test_dir = os.path.join(data_dir, 'Test')
    image_size = (244,244)
    
    if not os.path.exists(save_dir):
        print("Unable to save output images/files")

    # Creating visuals to analyze performance of model
    # Starting with accuracy vs loss for test/validation
    # Moving to visualizing 5 images from test set and predictions

    # Plotting the validation accuracy & loss vs training accuracy & loss
    plt.figure()
    # plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    # plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Validation Accuracy","Validation Loss"])
    plt.savefig(os.path.join(save_dir, 'Accuracy_Loss1.jpg'))
    plt.close()

    # Plotting v2 to see deeper analysis...
    plt.figure(figsize=(10, 5))

    # Plot with different line styles and markers
    plt.plot(hist.history['accuracy'], 'b-o', label="Accuracy", linewidth=2, markersize=5)
    plt.plot(hist.history['val_accuracy'], 'r-s', label="Validation Accuracy", linewidth=2, markersize=5)
    plt.plot(hist.history['loss'], 'g--d', label="Loss", linewidth=2, markersize=5)
    plt.plot(hist.history['val_loss'], 'k-.*', label="Validation Loss", linewidth=2, markersize=5)
    plt.title("Model Accuracy and Loss (Test vs Val)")
    plt.ylabel("Value")
    plt.xlabel("Epoch")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Moves the legend outside of the plot
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(os.path.join(save_dir, 'Accuracy_Loss2.jpg'))
    plt.close()

    # Testing with 5 images from the testing set... using saved model
    saved_model = load_model(os.path.join(save_dir, 'vgg16_KH.h5'))

    subclasses = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    selected_images = []

    # Continue to sample until we have 5 images
    while len(selected_images) < 5:
        # Randomly pick a subclass directory
        subclass = random.choice(subclasses)
        subclass_path = os.path.join(test_dir, subclass)
        
        # Get a list of image filenames in the subclass directory
        subclass_images = [f for f in os.listdir(subclass_path) if os.path.isfile(os.path.join(subclass_path, f))]
        
        # Make sure there's at least one image in the subclass directory
        if subclass_images:
            # Randomly pick an image from the subclass
            filename = random.choice(subclass_images)
            selected_images.append(os.path.join(subclass, filename))  # Save the relative path from test_dir

    grid_size = (2,3)
    grid_image = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]), (255, 255, 255))
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=25)

    for index, filename in enumerate(selected_images):
        # Load and process each image
        img_path = os.path.join(test_dir, filename)
        img = load_img(img_path, target_size=image_size)
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        output = saved_model.predict(img_array)
        prediction = "Fresh" if output[0][0] > output[0][1] else "Rotten"
        # predicted_class_index = np.argmax(output, axis=1)[0]
        # last_conv_layer_name = 'block5_conv3'
        # gradcam_img = apply_gradcam(img_array, saved_model, last_conv_layer_name)

        # Overlay the prediction on the image
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        text_color = (255, 255, 255)  # White text color
        text_background = (0, 0, 0)  # Black background for text
        margin = 10
        text_width, text_height = draw.textsize(prediction, font=font)
        draw.rectangle([(0, img.height - text_height - 2 * margin), 
                        (text_width + 2 * margin, img.height)], 
                    fill=text_background)
        draw.text((margin, img.height - text_height - margin), prediction, font=font, fill=text_color)

        # Calculate the position of the image in the grid
        x = index % grid_size[0] * image_size[0]
        y = index // grid_size[0] * image_size[1]

        # Paste the image onto the grid canvas
        grid_image.paste(img, (x, y))

    # Save the grid image! s
    grid_image.save(os.path.join(save_dir, 'test_predictions_grid.jpg'))

    # GRAD-CAM IMPLEMENTATION
    base_dir = test_dir  # Update this path
    img_path = get_img_path(base_dir)

    # Assuming `model` is your full model with VGG16 as the base and custom dense layers on top
    last_conv_layer_name = 'block5_conv3'  # Last conv layer in the VGG16 part
    classifier_layer_names = ['global_average_pooling2d', 'dense_1024', 'dropout_0.25', 'output_softmax']
    img_array = load_and_preprocess_image(img_path)  # Assuming you've already defined this function
    heatmap = make_gradcam_heatmap(img_array, saved_model, last_conv_layer_name, classifier_layer_names)
    save_and_display_gradcam(img_path, heatmap, cam_path=os.path.join(save_dir, 'gradCAM.jpg'))  # Update save path

## Implementing GradCAM

def get_img_path(base_dir):
    category = random.choice(['Fresh', 'Rotten'])  # Choose between 'fresh' or 'rotten'
    img_dir = os.path.join(base_dir, category)
    img_name = random.choice(os.listdir(img_dir))
    return os.path.join(img_dir, img_name)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(244,244))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    
    output_file_path = os.path.join(os.path.join(os.path.dirname(__file__), 'outputs'), 'VGGmodel_summary.txt')
    with open(output_file_path, 'w') as f:
        def print_to_file(text):
            print(text, file=f)

        last_conv_layer.summary(print_fn=print_to_file)

    # Now, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the predictions
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by 'how important this channel is'
    # with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
