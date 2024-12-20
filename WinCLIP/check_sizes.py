import os
from PIL import Image

def check_unique_image_shapes(dataset_path):
    """
    Traverses through the dataset and prints each unique image shape found once.
    
    Args:
    - dataset_path (str): Path to the root dataset directory.
    
    Returns:
    - None: Prints out each unique shape found in the dataset.
    """
    found_shapes = set()

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        shape = img.size  # (width, height)
                        if shape not in found_shapes:
                            print(f"Found new shape: {shape}")
                            found_shapes.add(shape)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    dataset_path = "/home/bearda/PycharmProjects/winclipbad/WinCLIP/AeBAD"
    check_unique_image_shapes(dataset_path)
