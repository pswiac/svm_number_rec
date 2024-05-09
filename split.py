# Python script for splitting an image of handwritten digits into jpg images of single charachters
# and preprocessing them for image recognition

import os
import shutil
import cv2


# Load the image
image_path = "handwritten.jpg" # input path to image to be split and classified
image = cv2.imread(image_path)

# Rotate the image if needed
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform thresholding to convert to binary image
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours from left to right
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# Draw contours on the image
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

image_with_contours = cv2.resize(image_with_contours, (500, 500))
# Display the image with contours
cv2.imshow('', image_with_contours)
cv2.waitKey(0)

# Clear directory for saving images
if os.path.exists("split_images"):
    shutil.rmtree("split_images")
os.makedirs("split_images")

# Split the image into individual characters
for i, contour in enumerate(contours):
    # Get the bounding box coordinates for the character
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the character image
    char_image = gray_image[y:y + h, x:x + w]

    # Apply histogram equalization
    blurred_image = cv2.GaussianBlur(char_image, (3, 3), 3)

    # Enhance the image
    enhanced_image = cv2.adaptiveThreshold(blurred_image, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 9, 4)

    # Resize the character image to 28x28
    resized_image = cv2.resize(enhanced_image, (28, 28))

    # Compute the histogram and contrast as its standardt deviation
    hist = cv2.calcHist([resized_image], [0], None, [256], [0, 256])
    contrast = cv2.meanStdDev(hist)[1][0][0]

    # Set threshold
    threshold = 48

    # Save the character image if the contrast is high enough

    if contrast <= threshold:
        cv2.imwrite("split_images/char" + str(i) + ".jpg", resized_image)

    print(contrast)
