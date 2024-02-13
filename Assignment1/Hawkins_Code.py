pip install tensorflow
pip install numpy
pip install pandas

# Get rid of annoying warnings
import warnings
warnings.filterwarnings('ignore')

# Basic imports
import numpy as np
import pandas as pd

# Imports for CNN
import tensorflow as tf
import tensorflow.keras as kb
import keras_tuner as kt

# Reading in Data / Data Engineering

train_dir = './Fruit_Classification/Train'
test_dir = './Fruit_Classification/Test'
val_dir = './Fruit_Classification/Val'

batch_size = 32
image_width = 300
image_height = 300

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir, # file path
  seed=123, # seed
  image_size= (image_width, image_height), # size of image
  batch_size=batch_size) # number of images per batch

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir, # file path
  seed=123, # seed
  image_size= (image_width, image_height), # size of image
  batch_size=batch_size) # number of images per batch

val_df = tf.keras.utils.image_dataset_from_directory(
  val_dir, # file path
  seed=123, # seed
  image_size= (image_width, image_height), # size of image
  batch_size=batch_size) # number of images per batch

# EDA
num_train_batches = tf.data.experimental.cardinality(train_ds).numpy()
num_test_batches = tf.data.experimental.cardinality(test_ds).numpy()

approx_train_size = num_train_batches * batch_size
approx_test_size = num_test_batches * batch_size

print("Train Dataset Size: ", approx_train_size)
print("Test Dataset Size: ", approx_test_size)
print("TOTAL Dataset Size: ", approx_train_size+approx_test_size)

for images, labels in train_ds.take(1):
    image_shape = images[0].shape

print(f"Image Shape: {image_shape}")

# Random Forest



# CNN - Basic from Dr.Parlett

model = kb.Sequential()

model.add(kb.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(kb.layers.MaxPooling2D((2, 2)))

model.add(kb.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(kb.layers.MaxPooling2D((2, 2)))

model.add(kb.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(kb.layers.MaxPooling2D((2, 2)))

model.add(kb.layers.Flatten())

model.add(kb.layers.Dense(128, activation='relu'))
model.add(kb.layers.Dropout(0.5))

model.add(kb.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=kb.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

history = model.fit(
    train_ds,
    epochs=2,
    validation_data= test_ds,
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

