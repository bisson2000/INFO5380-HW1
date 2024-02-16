import cv2
import numpy as np
from PIL import Image


image = cv2.imread('handdrawnimages/IMG_9462.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blank_image = np.ones_like(image) * 255
cv2.drawContours(blank_image, contours, -1, (0, 0, 0), thickness=18) #cv2.FILLED for lighter contours
pil_image = Image.fromarray(blank_image)
pil_image.save('test2.png')