import os
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def train_model(data):

    train_ds, test_ds, val_ds = data

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

    # Saving images/outputs to outputs folder
    save_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(save_dir):
        print("Unable to save output images/files")
    
    tuner = kt.RandomSearch(build_model,
                            objective = 'val_accuracy', 
                            max_trials = 10,
                            overwrite = True, # Needed to overwrite previous saves due to issues
                            directory = save_dir, 
                            project_name = 'Assignment1',
                            )

    tuner.search(train_ds, epochs = 50, 
                validation_data = val_ds, 
                verbose = 0)

    best_hp = tuner.get_best_hyperparameters(num_trials = 1)[0]

    # Correctly get the best model and evaluate it
    best_model = tuner.get_best_models(num_models = 1)[0]  # Select the first model from the list of best models

    # Utilizing checkpoint for saving model and early stopping to minmize loss 
    checkpoint = ModelCheckpoint(filepath = os.path.join(save_dir, "vgg16_KH.h5"), 
                                monitor='accuracy', 
                                verbose=0, save_best_only=True, 
                                save_weights_only=False, mode='auto', 
                                save_freq='epoch')
    early = EarlyStopping(monitor='accuracy', patience=10, 
                        verbose=0, mode='auto')

    # Fitting the best model found given the checkpoint and early stopping callbacks
    hist = best_model.fit(train_ds, validation_data=val_ds, 
                        epochs=100, 
                        callbacks=[checkpoint, early], verbose=2)

    # Define a custom print function that writes to the file
    # Utilizing this for further reading of the ouptut, documentation into the model_summary
    output_file_path = os.path.join(save_dir, 'model_summary.txt')
    with open(output_file_path, 'w') as f:
        def print_to_file(text):
            print(text, file=f)

        best_model.summary(print_fn=print_to_file)
        print_to_file(f"Tuner found the best activation function: {best_hp.get('dense_activation')}")
        print_to_file(f"Tuner found the best learning rate: {best_hp.get('lr') * 100:.2f}")
        print_to_file(f"Best Model Test accuracy: {hist.history['accuracy'][0] * 100:.2f}%")
        print_to_file(f"Best Model Test val_accuracy: {hist.history['val_accuracy'][0] * 100:.2f}%")
        print_to_file(f"Best Model Test loss: {hist.history['loss'][0] * 100:.2f}%")
        print_to_file(f"Best Model Test val_loss: {hist.history['val_loss'][0] * 100:.2f}%")

    return hist
