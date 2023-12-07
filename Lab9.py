import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the images
img1 = cv2.imread(r'C:\Lab9\House.jpg')  # Update with the correct path if needed
img2 = cv2.imread(r'C:\Lab9\ATU2.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_bf = bf.match(des1, des2)

# Sort them in ascending order of distance
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

# Draw matches
img_matches_bf = cv2.drawMatches(img1, kp1, img2, kp2, matches_bf[:10], None, flags=2)

# Display the result
plt.imshow(img_matches_bf)
plt.title('BruteForceMatcher')
plt.show()
