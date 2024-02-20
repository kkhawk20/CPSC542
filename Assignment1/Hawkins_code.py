import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import random
import keras_tuner as kt
import tensorflow.keras as kb
import keras
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

# ----------- Reading in Data / Data Engineering ------------
print("----------- DATA ENGINEERING / PREPROCESSING ------------")

# Ensuring the data is being accessed! 
train_dir = '/app/rundir/Fruit_Classification/Train'
test_dir = '/app/rundir/Fruit_Classification/Test'
val_dir = '/app/rundir/Fruit_Classification/Val'
save_dir = '/app/rundir/Fruit_Classification/'

# Pre-setting the known image size and set batch size
image_size = (300,300)
batch_size = 32

# Defining an easy function to grab images and labels
def preprocess_image(image, label):
    return preprocess_input(image), label

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


# ----------- VGG16 (CNN) ------------
print("----------- CNN - VGG16 ------------")

def build_model(hp):
    # The base structure is a VGG16 with no top layer, 
    # weights set to pre-trained from imagenet
    # Input size is the 300x300x3 for color image
    base = VGG16(include_top = False, weights = 'imagenet', 
                 input_shape = (300, 300, 3))
    
    # Beginning the model with the VGG16 and flatten layer
    model = Sequential([base, GlobalAveragePooling2D()]) 

    # Dense layer, tuner chooses activation function
    model.add(Dense(1024, activation = hp.Choice('dense_activation', 
                                                 values = ['relu', 'leaky_relu', 'sigmoid'])))
    
    # Conditionally add dropout layer based on tuner options
    if hp.Boolean("dropout"):
        model.add(Dropout(rate = 0.25))

    # Adding softmax last dense layer
    model.add(Dense(2, activation = 'softmax')) # Softmax for predicted probabilities of classification
    
    # Tuner chooses learning rate between .0001 and .001, samples with LOG intervals
    learning_rate = hp.Float('lr', min_value = 1e-4, max_value = 1e-2, sampling = 'log')

    # Compiles utilizing ADAM, tuned learning rate, loss function of binary_crossentropy for 
    # Binary predictions (Fresh / Rotten), validation accuracy is metric to track
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), 
                  loss = 'binary_crossentropy', metrics = ['accuracy'],
                  )
    
    return model

# Needed to clear session due to lingering things
from tensorflow.keras.backend import clear_session
clear_session()

tuner = kt.RandomSearch(build_model,
                        objective = 'val_accuracy', 
                        max_trials = 10,
                        overwrite = True, # Needed to overwrite previous saves due to issues
                        directory = '/nfshome/kehawkins/test_rundir/Fruit_Classification/', 
                        project_name = 'Assignment1',
                        )

tuner.search(train_ds, epochs = 25, 
             validation_data = val_ds, 
             verbose = 2)

best_hp = tuner.get_best_hyperparameters(num_trials = 1)[0]

print("Tuner found the best activation function:", best_hp.get('dense_activation'))
print("Tuner found the best learning rate:", best_hp.get('lr'))

# Correctly get the best model and evaluate it
best_model = tuner.get_best_models(num_models = 1)[0]  # Select the first model from the list of best models

loss, accuracy = best_model.evaluate(test_ds)

# Utilizing checkpoint for saving model and early stopping to minmize loss 
checkpoint = ModelCheckpoint("vgg16_KH.h5", monitor='accuracy', 
                             verbose=2, save_best_only=True, 
                             save_weights_only=False, mode='auto', 
                             save_freq='epoch')
early = EarlyStopping(monitor='accuracy', patience=10, 
                      verbose=2, mode='auto')

# Fitting the best model found given the checkpoint and early stopping callbacks
hist = best_model.fit(train_ds, validation_data=val_ds, 
                    epochs=100, 
                    callbacks=[checkpoint, early], verbose=2)

# Figuring out the keys given by the model fitting history value
print("Accuracy:", hist.history["accuracy"][:5])  # Print first 5 accuracy values for visability

# Define a custom print function that writes to the file
# Utilizing this for further reading of the ouptut, documentation
with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
  def print_to_file(text):
      print(text, file = f)
  best_model.summary(print_fn=print_to_file)
  print_to_file(f"Tuner found the best activation function: {best_hp.get('dense_activation')}")
  print_to_file(f"Tuner found the best learning rate: {best_hp.get('lr')*100:.2f}")
  print_to_file(f"Best Model Test accuracy: {hist.history['accuracy'][0]*100:.2f}%")
  print_to_file(f"Best Model Test val_accuracy: {hist.history['val_accuracy'][0]*100:.2f}%")
  print_to_file(f"Best Model Test loss: {hist.history['loss'][0]*100:.2f}%")
  print_to_file(f"Best Model Test val_loss: {hist.history['val_loss'][0]*100:.2f}%")


# Creating visuals to analyze performance of model
# Starting with accuracy vs loss for test/validation
# Moving to visualizing 5 images from test set and predictions

# Plotting the validation accuracy & loss vs training accuracy & loss
plt.figure()
# plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
# plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.savefig(os.path.join(save_dir, 'Accuracy_Loss.jpg'))
plt.close()

# Plotting v2 to see if its better
plt.figure(figsize=(10, 5))  # You can adjust the size to fit your needs

# Plot with different line styles and markers
plt.plot(hist.history['accuracy'], 'b-o', label="Accuracy", linewidth=2, markersize=5)
plt.plot(hist.history['val_accuracy'], 'r-s', label="Validation Accuracy", linewidth=2, markersize=5)
plt.plot(hist.history['loss'], 'g--d', label="Loss", linewidth=2, markersize=5)
plt.plot(hist.history['val_loss'], 'k-.*', label="Validation Loss", linewidth=2, markersize=5)

plt.title("Model Accuracy and Loss")
plt.ylabel("Value")
plt.xlabel("Epoch")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Moves the legend outside of the plot

# Ensure the plot is saved without cutting off the legend
plt.tight_layout(rect=[0, 0, 0.75, 1])

plt.savefig(os.path.join(save_dir, 'Accuracy_Loss2.jpg'))
plt.close()



# Testing it out with an image from the internet for fun
img = load_img("/app/rundir/Fruit_Classification/TEST_PEACH.jpeg",target_size=(300,300))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)

saved_model = load_model("vgg16_KH.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("Fresh")
else:
    print('Rotten')


# Testing with 5 images from the testing set
saved_model = load_model('vgg16_KH.h5')

# Assuming 'test_dir' is the main directory containing the subclass directories
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

# Save the grid image
grid_image.save('test_predictions_grid.jpg')