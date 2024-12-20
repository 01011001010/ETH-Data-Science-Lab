import os
from PIL import Image

# Define the dataset path
dataset_path = '/home/bearda/PycharmProjects/winclipbad/VisA_20220922'

# Initialize variable to store the shape of the first image found
image_shape = None

# Traverse through the dataset directory to find the first image and get its shape
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(root, file)
            with Image.open(image_path) as img:
                image_shape = img.size  # Get the (width, height)
            break
    if image_shape:
        break

print("Image shape:", image_shape)
