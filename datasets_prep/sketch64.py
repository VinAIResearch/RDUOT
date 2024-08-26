import os

from PIL import Image


data_name = "clipart"
resized_size = 64
input_dir = f"./data/{data_name}"
output_dir = f"./data/{data_name}_{resized_size}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    output_class_path = os.path.join(output_dir, class_folder)

    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)

    # Loop through images in the class folder
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        output_image_path = os.path.join(output_class_path, image_file)

        # Open the image using PIL
        image = Image.open(image_path)

        # Resize the image to 64x64
        resized_image = image.resize((64, 64))

        # Save the resized image to the output folder
        resized_image.save(output_image_path)
