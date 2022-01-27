# coding:utf-8

# Import the required libraries
import argparse
import os

import pandas as pd
import numpy as np

from tensorflow import keras

# Create the arguments

valid_image_format = ["jpeg", "jpg"]
disease_labels = [
    "chlorosis",
    "downy_mildew",
    "powdery_mildew",
    "rust"
]

def is_valid_image(parser, file):
    if not os.path.isfile(file):
        parser.error(f"The image {file} does not exist!")
    else:
        extension = file.split(".")[-1]
        if extension in valid_image_format:
            return file
        else:
            parser.error(f"The image must be in {valid_image_format} format")


def create_parser():
    parser = argparse.ArgumentParser(description = "Predict the plant disease based on a picture")

    parser.add_argument(
        "-i", 
        dest = "image_path",
        required = True,
        help = "Path to the image file",
        metavar = "FILE",
        type = lambda x: is_valid_image(parser, x)
)
    return parser
    
def get_image_path(parser):
    return parser.parse_args().image_path


if __name__ == "__main__":
    parser = create_parser()
    image_path = get_image_path(parser)
    
    # Import the image
    image = keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    
    # Load the model
    model = keras.models.load_model("plant_disease_classifier.h5")
    
    # Prediction
    prediction = model.predict(image)[0]
    
    # Output
    output = pd.DataFrame()
    output["disease"] = disease_labels
    output["prob"] = prediction
    output = output.sort_values("prob", ascending = False)
    print(output.to_string(index = False))