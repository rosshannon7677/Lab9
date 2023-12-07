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

# Display the result for the first pair of images
plt.figure(figsize=(8, 4))
plt.subplot(121), plt.imshow(img_matches_bf), plt.title('BruteForceMatcher - House vs ATU2')

# Load new images
img1_new = cv2.imread(r'C:\Lab9\Dail.jpg')
img2_new = cv2.imread(r'C:\Lab9\Kitchen.jpg')

# Convert the new images to grayscale
gray1_new = cv2.cvtColor(img1_new, cv2.COLOR_BGR2GRAY)
gray2_new = cv2.cvtColor(img2_new, cv2.COLOR_BGR2GRAY)

# Find the keypoints and descriptors with ORB for the new images
kp1_new, des1_new = orb.detectAndCompute(gray1_new, None)
kp2_new, des2_new = orb.detectAndCompute(gray2_new, None)

# Brute-Force Matcher for the new images
matches_bf_new = bf.match(des1_new, des2_new)

# Sort them in ascending order of distance
matches_bf_new = sorted(matches_bf_new, key=lambda x: x.distance)

# Draw matches for the new images
img_matches_bf_new = cv2.drawMatches(img1_new, kp1_new, img2_new, kp2_new, matches_bf_new[:10], None, flags=2)

# Display the result for the second pair of images
plt.subplot(122), plt.imshow(img_matches_bf_new), plt.title('BruteForceMatcher - Dail vs Kitchen')

# Adjust layout for better display
plt.tight_layout()

# Show the plots
plt.show()
