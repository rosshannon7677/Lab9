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

# Mark the corners on the deep copy
imgHarris[dst > 0.01 * dst.max()] = [0, 0, 255]  # Red color for corners

# Plot the result
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.show()
