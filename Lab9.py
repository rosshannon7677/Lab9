import cv2
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread(r'C:\Lab9\House.jpg')  # Update with the correct path to your image

# Split the image into channels
b, g, r = cv2.split(img)

# Create a Matplotlib subplot
plt.figure(figsize=(10, 4))

# Display the original image
plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.axis('off')

# Display the Red channel
plt.subplot(1, 4, 2), plt.imshow(r, cmap='gray'), plt.title('Red Channel')
plt.axis('off')

# Display the Green channel
plt.subplot(1, 4, 3), plt.imshow(g, cmap='gray'), plt.title('Green Channel')
plt.axis('off')

# Display the Blue channel
plt.subplot(1, 4, 4), plt.imshow(b, cmap='gray'), plt.title('Blue Channel')
plt.axis('off')

# Show the plot
plt.show()
