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

# Create a deep copy of the original image for drawing Harris circles
imgHarris = copy.deepcopy(imgOrig)

# Set the threshold for Harris corner detection
threshold_harris = 0.01

# Loop through every element in the dst matrix
for i in range(len(dst)):
    for j in range(len(dst[i])):
        # Check if the element is greater than the threshold
        if dst[i][j] > (threshold_harris * dst.max()):
            # Draw a circle on the imgHarris image for Harris corners
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)  # Red color for Harris circles

# Plot the image with Harris corners
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.show()

# Commented out the Harris corner detection code
"""
# Set the threshold for Harris corner detection
threshold_harris = 0.01

# Loop through every element in the dst matrix
for i in range(len(dst)):
    for j in range(len(dst[i])):
        # Check if the element is greater than the threshold
        if dst[i][j] > (threshold_harris * dst.max()):
            # Draw a circle on the imgHarris image for Harris corners
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)  # Red color for Harris circles
"""

# Perform corner detection using Shi-Tomasi algorithm
maxCorners = 100  # Experiment with different values
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(imgGray, maxCorners, qualityLevel, minDistance)

# Create a deep copy of the original image for drawing GFTT circles
imgGFTT = copy.deepcopy(imgOrig)

# Convert corners to integers
corners = np.int0(corners)

# Draw circles on the image for each detected corner using Shi-Tomasi
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(imgGFTT, (x, y), 3, (0, 255, 0), -1)  # Green color for GFTT circles

# Create a deep copy of the original image for Shi-Tomasi corners
imgShiTomasi = copy.deepcopy(imgOrig)

# Draw circles on the image for each detected corner using Shi-Tomasi
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(imgShiTomasi, (x, y), 3, (255, 0, 0), -1)  # Blue color for Shi-Tomasi circles

# Plot the image with Shi-Tomasi corners
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corners')
plt.show()
