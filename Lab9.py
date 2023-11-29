# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread(r'C:\Lab8\ATU.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)