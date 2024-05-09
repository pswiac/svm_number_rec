# Python script for importing jpg images into a csv file as binary values for image size 28x28
# and classifying them according to imported classifier

import os
import csv
import joblib as joblib
import numpy as np
import pandas as pd
from PIL import Image


# Get the list of image files in the directory
image_files = [file for file in os.listdir("split_images") if file.endswith('.jpg')]

# Open the CSV file in write mode
with open("digits.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row
    header = ['Image File']
    for i in range(784):
        header.append(str(i))
    writer.writerow(header)

    # Write each image data as a new row
    for image_file in image_files:
        image_path = os.path.join("split_images", image_file)

        # Load the image using PIL
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Resize for classifier
        image = image.resize((28, 28))

        # Extract the grayscale values of each pixel
        pixel_values = list(image.getdata())

        # Write the image file name and pixel values as a new row
        row = [image_file] + pixel_values
        writer.writerow(row)

image = Image.new('L', (28, 28))
image.putdata(pixel_values)
image.show()

# Import classifier
svm_classifier = joblib.load(open("svm_classifier.pkl", 'rb'))

df = pd.read_csv("digits.csv")

test = df.values

test_data = test[:, 1:]

test_data = test_data / 255

predicted = svm_classifier.predict(test_data)

print(predicted)
np.savetxt("predicted_digits.txt", predicted, fmt='%d')

