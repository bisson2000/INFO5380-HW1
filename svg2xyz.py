# -*- coding: utf-8 -*-
"""
Digital Fabrication HW1 - Convert SVG path data to XYZ coordinates for wire bending.
William J. Reid (wjr83) - 02/06/2024
"""

import tkinter as tk
from tkinter import filedialog
import svgpathtools
import csv
import numpy as np
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import os

def select_file():
    """
    Prompt the user to select an SVG file via the file explorer.
    
    Returns:
        str: Path to the selected SVG file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("Select SVG files, PNG or JPG files", "*.svg;*.png;*.jpg")])
    return file_path

def svg_to_xyz(svg_file, output_file):
    """
    Convert SVG path data to XYZ coordinates and save to a CSV file.
    
    Parameters:
        svg_file (str): Path to the input SVG file.
        output_file (str): Path to the output CSV file.
    """
    # Load SVG file and parse its path data
    paths, _ = svgpathtools.svg2paths(svg_file)

    # Initialize an empty list to store the XYZ coordinates
    path_coordinates = []
    
    # Iterate through each path in the SVG file
    for path in paths:
        # Iterate through each segment in the path
        for segment in path:
            # If the segment is a straight line
            if isinstance(segment, svgpathtools.Line):
                # Get start and end points of the line segment
                start = (segment.start.real, segment.start.imag, 0)
                end = (segment.end.real, segment.end.imag, 0)
                # Add start and end points to the path coordinates
                path_coordinates.extend([start, end])
            # If the segment is an arc
            elif isinstance(segment, svgpathtools.Arc):
                # Extract points from the arc segment
                points = arc_to_points(segment)
                # Add points along the arc to the path coordinates
                for i in range(len(points) - 1):
                    start = (points[i].real, points[i].imag, 0)
                    end = (points[i + 1].real, points[i + 1].imag, 0)
                    path_coordinates.extend([start, end])
            # If the segment is a cubic Bezier curve
            elif isinstance(segment, svgpathtools.CubicBezier):
                # Extract points from the cubic Bezier segment
                points = bezier_to_points(segment)
                # Add points along the curve to the path coordinates
                for i in range(len(points) - 1):
                    start = (points[i].real, points[i].imag, 0)
                    end = (points[i + 1].real, points[i + 1].imag, 0)
                    path_coordinates.extend([start, end])
            # If the segment is a quadratic Bezier curve
            elif isinstance(segment, svgpathtools.QuadraticBezier):
                # Extract points from the quadratic Bezier segment
                points = bezier_to_points(segment)
                # Add points along the curve to the path coordinates
                for i in range(len(points) - 1):
                    start = (points[i].real, points[i].imag, 0)
                    end = (points[i + 1].real, points[i + 1].imag, 0)
                    path_coordinates.extend([start, end])

    # Write XYZ coordinates to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y', 'Z'])
        for point in path_coordinates:
            writer.writerow([point[0], point[1], point[2]])

def arc_to_points(arc, num_points=100):
    """
    Generate points along an arc segment.
    
    Parameters:
        arc (svgpathtools.Arc): The arc segment.
        num_points (int): Number of points to generate.
    
    Returns:
        list: List of points representing the arc.
    """
    # Extract relevant parameters of the arc segment
    start = arc.start
    end = arc.end
    center = arc.center
    radius = abs(arc.radius)
    # Compute start and end angles of the arc
    theta_start = np.angle(start - center)
    theta_end = np.angle(end - center)
    # Generate equally spaced points along the arc
    theta = np.linspace(theta_start, theta_end, num_points)
    points = [center + radius * np.exp(1j * t) for t in theta]
    return points

def bezier_to_points(bezier, num_points=100):
    """
    Generate points along a Bezier curve segment.
    
    Parameters:
        bezier (svgpathtools.CubicBezier or svgpathtools.QuadraticBezier): The Bezier curve segment.
        num_points (int): Number of points to generate.
    
    Returns:
        list: List of complex points representing the Bezier curve.
    """
    # Generate equally spaced parameter values along the curve
    t_values = np.linspace(0, 1, num_points)
    # Evaluate the Bezier curve at each parameter value
    points = [bezier.point(t) for t in t_values]
    return points


# TODO: Harshini & William → png of hand-drawn shape to svg (computer vision techniques in between / edge detection, etc.) → coordinates
def image2svg(image_path):
    # Read the uploaded image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Dilate the edges to ensure connectivity and single-pixel width
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Finding contours
    contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the largest contour with a simpler polygonal curve
    approx_largest_contour = cv2.approxPolyDP(largest_contour, 0.01 * cv2.arcLength(largest_contour, True), True) # Will create more lines than splines

    # Approximate the largest contour with a spline-like polygonal curve
    # epsilon = 0.005 * cv2.arcLength(largest_contour, True) # Adjust epsilon for spline-like approximation (tested values that are too small: 0.00001. 0.00025, 0.0005)
    # approx_largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert the largest contour to SVG path
    path_data = "M"
    for point in approx_largest_contour.squeeze():
        path_data += f"{point[0]},{point[1]} "
    path_data += "Z"
    
    # Get the filename without the extension
    file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]

    # Create a folder named "svg_files" if it doesn't exist
    if not os.path.exists("svg_files"):
        os.makedirs("svg_files")

    # Create an SVG file path based on the name of the original image
    file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]
    svg_output_path = f'svg_files/{file_name_without_extension}.svg' 

    # Create an SVG file and draw the largest contour as a path
    dwg = svgwrite.Drawing(svg_output_path, size=(image.shape[1], image.shape[0]))
    dwg.add(dwg.path(d=path_data, fill="none", stroke="black", stroke_width=1))
    dwg.save() # Save newly create svg file to the svg_file folder

    print("Conversion to SVG completed. New SVG saved to:", svg_output_path)
    return svg_output_path
  
    # For Troubleshooting
    #     # Display images
    #   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #   # Original image
    #   axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #   axes[0].set_title("Original Image")
    #   axes[0].axis('off')

    #   # Dilated edges
    #   axes[1].imshow(dilated_edges, cmap='gray')
    #   axes[1].set_title("Dilated Edges")
    #   axes[1].axis('off')

    #   # Largest contour
    #   largest_contour_image = np.zeros_like(gray)
    #   cv2.drawContours(largest_contour_image, [approx_largest_contour], -1, (255), thickness=cv2.FILLED)
    #   axes[2].imshow(largest_contour_image, cmap='gray')
    #   axes[2].set_title("Largest Contour, Smoothed Edges")
    #   axes[2].axis('off')

    #   # Display the cleaned SVG
    #   # display(SVG(dwg.tostring())) # Very large plot compared to png images displayed

    #   plt.show()
  # End of TODO


#TODO: Paul → Function: Handle overlapping lines



'''
Other TODO's:
TODO: Backburner (more challenging): picture to png to svg
TODO: Scale coordinates to produce shape of X size on the wire bender
    If SVG contains unconnected objects, how does it tell the machine to cut the wire before proceeding with bending the next shape?
    One potential approach: Ask the user to connect separated shapes or produce multiple output files (one for each shape/path detected).
'''

def main():
    # Prompt user to select a file
    file_path = select_file()
    if file_path:
        output_file = 'output.csv'
        # If SVG, then convert to coordinates
        if file_path.lower().endswith('.svg'):
            svg_to_xyz(file_path, output_file)
            print("Conversion from SVG file to XYZ coordinates completed.") 
            print("Output saved to output.csv")
        # If PNG or JPG, first convert to SVG, then to coordinates
        else:
            print("The selected file is not an SVG file, converting to SVG...")
            path_svg = image2svg(file_path) # saves image file as svg
            svg_to_xyz(path_svg, output_file) # pass the new svg file to image2svg function to convert to coordinates
            print("Conversion to XYZ coordinates completed.") 
            print("Output saved to output.csv")
            
if __name__ == "__main__":
    main()



'''
OTHER TODO's 

TODO: Where should the wire bender machine start? 

TODO: Should the points be "shifted" to start at some origin (per requirements of the machine)?

TODO: If SVG contains unconnected objects, how does it tell the machine to cut the wire before proceeding with bending the next shape?

TODO: Rather than drawing letters, it would make more sense to type letters to wire bend.

TODO: Beautiful polar coordinates functions?

TODO: Convert a picture to an edge to a wireframe 
'''

# How to use this script:

# Provide path to the SVG file:
# svg_file = r"C:\Users\Windows\Documents\Cornell Tech\Spring 2024\INFO 5380 - Digital Fabrication\HW1\Git-Repository\INFO5380-HW1\svg_files\spline.svg"
# output_file = r"C:\Users\Windows\Documents\Cornell Tech\Spring 2024\INFO 5380 - Digital Fabrication\HW1\Git-Repository\INFO5380-HW1\output.csv"
# svg_to_xyz(svg_file, output_file)
