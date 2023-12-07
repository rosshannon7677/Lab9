import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv.imread(r'C:\Lab9\ATU1.jpg', cv.IMREAD_GRAYSCALE)

# ORB (Oriented FAST and Rotated BRIEF)
# Initiate ORB detector
orb = cv.ORB_create()

# Find the keypoints with ORB
kp = orb.detect(img, None)

# Compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# Draw only keypoints location, not size and orientation
img_with_keypoints = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

# Plot the image with keypoints
plt.imshow(img_with_keypoints, cmap='gray')
plt.title('ORB Keypoints')
plt.show()
