import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import load_img
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

def model_eval(hist):

    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'Data')
    test_dir = os.path.join(data_dir, 'Test')
    image_size = (300,300)
    last_conv_layer_name = 'block5_conv3'

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

        # Make predictions
        output = saved_model.predict(img_array)
        prediction = "Fresh" if output[0][0] > output[0][1] else "Rotten"

        gradcam_img = apply_gradcam(img_array, saved_model, last_conv_layer_name, prediction)

        # Overlay the prediction on the image
        img = img.convert("RGB")
        draw = ImageDraw.Draw(gradcam_img)
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


# Done!


# Gradcam??

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    # Create a model with direct access to the outputs of our last conv layer
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    # Get gradients of loss wrt the conv outputs
    grads = tape.gradient(loss, conv_outputs)

    # Each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weigh the output feature map channels by the importance to the prediction
    conv_outputs = conv_outputs[0]
    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    # For visualization, normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam(img_array, model, last_conv_layer_name, pred_index):
    # Generate the Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use cv2 to apply the heatmap to the original image
    img_original = np.array(img)  # Convert PIL image to numpy array (ensure 'img' is your PIL image before resizing)
    heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_original
    superimposed_img = np.uint8(superimposed_img)
    
    # Convert back to PIL image to keep the rest of your processing consistent
    return Image.fromarray(superimposed_img)

