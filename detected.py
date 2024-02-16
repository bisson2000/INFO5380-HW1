import io
import re
import cv2
import numpy as np
from PIL import Image
from skimage import morphology
from google.cloud import vision

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Google Cloud Vision API
def detect_text_and_get_boxes(content):
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    boxes = []
    for text in texts[1:]: 
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        boxes.append(vertices)
    return texts, boxes



def preprocess_image_for_ocr(image_path):

    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Resize the image to make it larger to help OCR
    scale_percent = 150 
    width = int(eroded.shape[1] * scale_percent / 100)
    height = int(eroded.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(eroded, dim, interpolation=cv2.INTER_LINEAR)
    
    return resized


image_path = '/Users/harshinisaidonepudi/Desktop/Spring 2024/df/INFO5380-HW1/handdrawnimages/IMG_9490.jpg'



import numpy as np
import cv2

def detect_background_color(image, box):

    x, y, w, h = cv2.boundingRect(np.array([box]))
    margin = 5  
    x, y, w, h = max(0, x-margin), max(0, y-margin), w+2*margin, h+2*margin

    sample_points = [
        (x, y), (x+w, y), (x, y+h), (x+w, y+h)
    ]
    
    colors = [image[pt[1], pt[0]] for pt in sample_points]
    avg_color = np.mean(colors, axis=0).astype(int)
    background_color = tuple(int(c) for c in avg_color) 

    return background_color

def mask_text_regions_with_blur(image, boxes):
    masked_image = image.copy()

    for box in boxes:
        background_color = detect_background_color(image, box)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(masked_image, [pts], background_color)

        # Inflate box for blurring
        x, y, w, h = cv2.boundingRect(pts)
        margin = 10  
        x, y, w, h = max(0, x-margin), max(0, y-margin), w+2*margin, h+2*margin

       
        localized_area = masked_image[y:y+h, x:x+w]
        blurred_area = cv2.GaussianBlur(localized_area, (15, 15), sigmaX=0)
        masked_image[y:y+h, x:x+w] = blurred_area

    return masked_image



# Read the original image as binary for OCR
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

detected_texts, text_boxes = detect_text_and_get_boxes(content)

individual_texts = detected_texts[1:] if detected_texts else []

numeric_texts = set()  
for text in individual_texts:

    numeric_text = re.findall(r'\d+', text.description)
    numeric_texts.update(numeric_text)  

numeric = list(numeric_texts)  
print("Detected numbers:", numeric_texts)# dimensions in image

#contour processing
original_image = cv2.imread(image_path)
masked_image = mask_text_regions_with_blur(original_image, text_boxes)
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Finding contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Background for drawing contours
blank_image = np.zeros_like(gray)

# Drawing contours
cv2.drawContours(blank_image, contours, -1, (255), thickness=cv2.FILLED)

# Conversion from array to bool for morphology
bool_image = blank_image > 0
cleaned_bool_image = morphology.remove_small_objects(bool_image, min_size=140, connectivity=2)

cleaned_image = np.uint8(cleaned_bool_image) * 255
outline_image = np.ones_like(masked_image) * 255  
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(outline_image, contours, -1, (0, 0, 0), thickness=18)
final_image_path = 'cleaned_outline_with_ocr.png'
cv2.imwrite(final_image_path, outline_image)

print(f"Final image saved to {final_image_path}")


