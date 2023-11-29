# Imports
import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread(r'C:\Lab9\ATU1.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Perform Harris corner detection
blockSize = 2
aperture_size = 3
k = 0.04
dst = cv2.cornerHarris(imgGray, blockSize, aperture_size, k)

# Create a deep copy of the original image
imgHarris = copy.deepcopy(imgOrig)

# Set the threshold for corner detection
threshold = 0.01

# Loop through every element in the dst matrix
for i in range(len(dst)):
    for j in range(len(dst[i])):
        # Check if the element is greater than the threshold
        if dst[i][j] > (threshold * dst.max()):
            # Draw a circle on the imgHarris image
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)  # Red color for circles

# Plot the image with Harris corners
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.show()

# Plot the result
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection with Circles')
plt.show()
