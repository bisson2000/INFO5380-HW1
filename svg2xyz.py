# -*- coding: utf-8 -*-
"""
Digital Fabrication HW1 - Convert image & SVG path data to XYZ coordinates for wire bending.
William J. Reid (wjr83) - 02/13/2024
Harshini Donepudi (hsd39) - 02/13/2024
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
import math

def select_file():
    """
    Prompt the user to select an SVG file via the file explorer.
    
    Returns:
        str: Path to the selected SVG file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("Select an SVG, PNG or JPG file:", "*.svg;*.png;*.jpg;*.jpeg")]) # Filter files to show only SVG, PNG, and JPG
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

    path_coordinates = remove_duplicates_coords(path_coordinates)
    path_coordinates = scale_coords(path_coordinates)
    path_coordinates = set_origin_to_zero(path_coordinates)
    path_coordinates = clear_points_intersections(path_coordinates)

    # Write XYZ coordinates to CSV file
    write_coordinates_to_file(path_coordinates, output_file)


def write_coordinates_to_file(path_coordinates, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X', 'Y', 'Z'])
        for point in path_coordinates:
            writer.writerow([point[0], point[1], point[2]])


def remove_duplicates_coords(points: list):
    """
    Clears duplicates.
    
    Parameters:
        points (List[List[int]]): The XYZ coordinates.
    
    Returns:
        list: List of points without duplicates.
    """

    if len(points) <= 1:
        return points.copy()

    res = [points[0]]
    for i in range(1, len(points)):
        if not np.array_equal(res[-1], points[i]):
            res.append(points[i])
    
    return res


def scale_coords(points: list, max_size=100):
    """
    Scales the coordinates to fit the max_size.
    
    Parameters:
        points (List[List[int]]): The XYZ coordinates.
        max_size: The maximum size allowed
    
    Returns:
        list: List of points scaled.
    """
    if len(points) <= 1:
        return points.copy()
    
    maxX = points[0][0]
    maxY = points[0][1]
    minX = points[0][0]
    minY = points[0][1]
    for point in points:
        maxX = max(maxX, point[0])
        maxY = max(maxY, point[1])
        minX = min(minX, point[0])
        minY = min(minY, point[1])
    
    width = maxX - minX
    height = maxY - minY
    original_centerX = (maxX + minX) / 2
    original_centerY = (maxY + minY) / 2

    scaleFactor = 1
    if width > height:
        scaleFactor = max_size / width
    else:
        scaleFactor = max_size / height

    new_centerX = max_size / 2
    new_centerY = max_size / 2

    new_points = []
    for point in points:
        new_pointX = new_centerX + scaleFactor * (point[0] - original_centerX)
        new_pointY = new_centerY + scaleFactor * (point[1] - original_centerY)
        new_pointZ = point[2]

        new_points.append([new_pointX, new_pointY, new_pointZ])

    return new_points


def set_origin_to_zero(points: list):
    """
    Offsets all points to move the origin to 0
    
    Parameters:
        points (List[List[int]]): The XYZ coordinates.
    
    Returns:
        list: List of points scaled.
    """

    if len(points) == 0:
        return points.copy()

    offsetX = points[0][0]
    offsetY = points[0][1]

    new_points = []
    for point in points:
        new_pointX = point[0] - offsetX
        new_pointY = point[1] - offsetY
        new_pointZ = point[2]

        new_points.append([new_pointX, new_pointY, new_pointZ])

    return new_points


#TODO: Paul → Function: Handle overlapping lines
def clear_points_intersections(points: list, clear_distance: float = 1.6):
    """
    Clears the intersections.

    Strategy: 
        Intersecting points will be lowered.
        In other words, the previous points will appear as "Higher" than the
        next points who were intersecting.
        With this strategy, it will always be safe to go lower
    
    Parameters:
        points (List[List[int]]): The XYZ coordinates.
        clear_distance (float): The minimum distance allowed.
    
    Returns:
        list: List of points modified to clear the distance.
    """

    if len(points) <= 2:
        return points
    
    def distance_point_to_line(line_start, line_end, point):
        """
        Algorithm inspired from
        https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
        """
        
        line_vector = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
        end_point_vector = np.array([point[0] - line_end[0], point[1] - line_end[1]])
        start_point_vector = np.array([point[0] - line_start[0], point[1] - line_start[1]])

        line_end_dot = np.dot(line_vector, end_point_vector)
        line_start_dot = np.dot(line_vector, start_point_vector)

        if line_end_dot > 0:
            return np.sqrt(end_point_vector.dot(end_point_vector))
        if line_start_dot < 0:
            return np.sqrt(start_point_vector.dot(start_point_vector))

        x1 = line_vector[0]
        y1 = line_vector[1]
        x2 = start_point_vector[0]
        y2 = start_point_vector[1]
        mod = math.sqrt(x1 * x1 + y1 * y1)
        return abs(x1 * y2 - y1 * x2) / mod

    def lines_cosine(line1_start, line1_end, line2_start, line2_end):
        line1_vector = np.array([line1_end[0] - line1_start[0], line1_end[1] - line1_start[1]])
        line2_vector = np.array([line2_end[0] - line2_start[0], line2_end[1] - line2_start[1]])
        return np.dot(line1_vector, line2_vector) / (np.linalg.norm(line1_vector) * np.linalg.norm(line2_vector))
    
    def segments_intersect(line1_start, line1_end, line2_start, line2_end):
        """
        Algorithm inspired from
        https://paulbourke.net/geometry/pointlineplane/example.cpp
        """
        denom = ((line2_end[1] - line2_start[1])*(line1_end[0] - line1_start[0])) - \
                      ((line2_end[0] - line2_start[0])*(line1_end[1] - line1_start[1]))

        num_a = ((line2_end[0] - line2_start[0])*(line1_start[1] - line2_start[1])) - \
                       ((line2_end[1] - line2_start[1])*(line1_start[0] - line2_start[0]))

        num_b = ((line1_end[0] - line1_start[0])*(line1_start[1] - line2_start[1])) -\
                       ((line1_end[1] - line1_start[1])*(line1_start[0] - line2_start[0]))

        if denom == 0:
            return False

        ua = num_a / denom
        ub = num_b / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return True

        return False
    
    def get_euclidian_dist(p, q):
        return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))

    def lines_overlap(line1_start, line1_end, line2_start, line2_end):
        # Exception for connected points
        # Make sure the current end point is not too close to the previous start point
        # and that it is pointing the the same direction
        if np.array_equal(line1_end, line2_start):
            cosine = lines_cosine(line1_start, line1_end, line2_start, line2_end)
            return get_euclidian_dist(line1_start, line2_end) <= clear_distance and cosine < 0
        
        # are the segments intersecting
        if segments_intersect(line1_start, line1_end, line2_start, line2_end):
            return True

        # min distance between 2 segments
        distance = distance_point_to_line(line1_start, line1_end, line2_start)
        distance = min(distance_point_to_line(line1_start, line1_end, line2_end), distance)
        distance = min(distance_point_to_line(line2_start, line2_end, line1_start), distance)
        distance = min(distance_point_to_line(line2_start, line2_end, line1_end), distance)

        return distance <= clear_distance

    current_height: float = 0
    previous_search_end = 0
    new_points = [[points[0][0], points[0][1], current_height]]

    for i in range(0, len(points) - 1):
        point_i_2d = [points[i][0], points[i][1]]
        point_i_1_2d = [points[i + 1][0], points[i + 1][1]]

        is_previous_same_direction = True
        for j in range(i, previous_search_end, -1):
            point_j_1_2d = [points[j - 1][0], points[j - 1][1]]
            point_j_2d = [points[j][0], points[j][1]]

            # the previous points all go in the same direction,
            # therefore no intersection
            if is_previous_same_direction and lines_cosine(point_j_1_2d, point_j_2d, point_i_2d, point_i_1_2d) >= 0:
                continue
            else:
                is_previous_same_direction = False

            # find intersection
            # If there is one, break here
            if lines_overlap(point_j_1_2d, point_j_2d, point_i_2d, point_i_1_2d):
                current_height -= clear_distance
                previous_search_end = i
                new_points[i][2] = current_height
                break

        new_points.append([points[i + 1][0], points[i + 1][1], current_height])

    return new_points


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
    """
    Convert Image file (PNG or JPG) to an SVG file by extracting edges and smoothening hand-drawn shape.
    
    Parameters:
        image_path (str): Path to the input SVG file.
    
    Returns:
        svg_output_path (str): path to newly generated SVG file.
    """
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
            print("Output saved to ", output_file)
        # If PNG or JPG, first convert to SVG, then to coordinates
        else:
            print("The selected file is not an SVG file, converting to SVG...")
            path_svg = image2svg(file_path) # saves image file as svg
            svg_to_xyz(path_svg, output_file) # pass the new svg file to image2svg function to convert to coordinates
            
            print("Conversion to XYZ coordinates completed.") 
            print("Output saved to output.csv")
            
        # TODO: Create a method so the coordinates can be imported with ease into Tobi's Fusion 360 Plug-in 
        
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
