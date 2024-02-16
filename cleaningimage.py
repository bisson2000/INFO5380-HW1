import cv2
import numpy as np
from PIL import Image
from skimage import morphology


image = cv2.imread('handdrawnimages/IMG_9462.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# finding contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# background
blank_image = np.zeros_like(gray)

# drawing contours
cv2.drawContours(blank_image, contours, -1, (255), thickness=cv2.FILLED)

#conversion from array to bool for morphology
bool_image = blank_image > 0

cleaned_bool_image = morphology.remove_small_objects(bool_image, min_size=140, connectivity=2)

cleaned_image = np.uint8(cleaned_bool_image) * 255
outline_image = np.ones_like(image) * 255
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(outline_image, contours, -1, (0, 0, 0), thickness=20)

pil_image = Image.fromarray(outline_image)
pil_image.save('cleaned_outline.png')