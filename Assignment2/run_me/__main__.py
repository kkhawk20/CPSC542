# This will run my code!
# in terminal, type in: python3 -m Fruit_class_modular

from . import *
def main():
    print("Running Image Segmentation Model!! ...")
    # Load the data using the custom SegmentationDataGenerator
    train_gen, val_gen, test_gen = load_data()
    
    # Train the U-Net model with the training and validation generators
    model, history = unet(data = (train_gen, val_gen, test_gen))
    
    # Evaluate the model using the test generator and calculate metrics
    model_eval(model=history, test_gen=test_gen)

if __name__ == '__main__':
    main()

    


