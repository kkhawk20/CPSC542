# ----- DO NOT RUN THIS AGAINNNN!!!! -------#
# WARNING
# WARNING


# import cv2
# import numpy as np
# import os
# import shutil

# # Define your current dataset structure
# base_dir = '/nfshome/kehawkins/test_rundir/CPSC542/Assignment2/run_me/Data'  # Change this to your current dataset path
# categories = ['Train', 'Val', 'Test']

# # Define the new directory structure for segmentation
# new_base_dir = '/nfshome/kehawkins/test_rundir/CPSC542/Assignment2/run_me/Data_new'  # Change this to your desired new dataset path

# # Function to create masks from images
# def create_mask_for_fruit(image_path, mask_dir):
#     # Read the image
#     image = cv2.imread(image_path)
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Threshold the image to create a mask
#     _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#     # Save the mask
#     mask_filename = os.path.basename(image_path).replace('.jpg', '_mask.jpg')
#     mask_path = os.path.join(mask_dir, mask_filename)
#     cv2.imwrite(mask_path, mask)
#     return mask_path

# # Function to create directories if they don't exist
# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# # Create the new directory structure and process each image
# for category in categories:
#     # Create directories for images and masks
#     new_images_dir = os.path.join(new_base_dir, category, 'Images')
#     new_masks_dir = os.path.join(new_base_dir, category, 'Masks')
#     create_dir(new_images_dir)
#     create_dir(new_masks_dir)

#     # Process each subcategory (e.g., 'Fresh', 'Rotten')
#     for subcategory in ['Fresh', 'Rotten']:
#         sub_dir = os.path.join(base_dir, category, subcategory)
#         for filename in os.listdir(sub_dir):
#             if filename.endswith('.jpg'):
#                 # Full path for current image
#                 image_path = os.path.join(sub_dir, filename)
#                 # Create mask for current image
#                 mask_path = create_mask_for_fruit(image_path, new_masks_dir)
#                 print(f"Mask created for {filename} at {mask_path}")
                
#                 # Move the original image to the new directory
#                 new_image_path = os.path.join(new_images_dir, filename)
#                 shutil.move(image_path, new_image_path)
#                 print(f"Image moved to {new_image_path}")

# print("Dataset reorganization and mask creation complete!")
