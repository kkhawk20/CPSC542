# This will run my code!
# in terminal, type in: python3 -m run_me


from . import *
import tensorflow as tf

def main():
    print("Running Image Segmentation Model!! ...")
    # Load the data using the custom SegmentationDataGenerator
    train_gen, val_gen, test_gen = load_data()
    
    # Train the U-Net model with the training and validation generators
    model, history = unet(train_gen, val_gen, test_gen)
    
    # Evaluate the model using the test generator and calculate metrics
    model_eval(history, model)

if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    main()

    


