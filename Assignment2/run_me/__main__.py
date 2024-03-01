# This will run my code!
# in terminal, type in: python3 -m Fruit_class_modular

from . import *
def main():
    print("Running Image Segmentation Model!! ...")
    data = load_data()
    train = unet(data)
    util = model_eval(train)

if __name__ == '__main__':
    main()
    


