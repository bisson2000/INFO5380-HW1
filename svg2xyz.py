# -*- coding: utf-8 -*-
"""
Digital Fabrication HW1 - Convert SVG path data to XYZ coordinates for wire bending.
William J. Reid (wjr83) - 02/06/2024
"""

import svgpathtools
import csv
import numpy as np

def svg_to_xyz(svg_file, output_file):
    """
    Convert SVG path data to XYZ coordinates and save to a CSV file.
    
    Parameters:
        svg_file (str): Path to the input SVG file.
        output_file (str): Path to the output CSV file.
    """
    # Load SVG file and parse its path data
    paths, _ = svgpathtools.svg2paths(svg_file)

    # Extract path coordinates
    path_coordinates = []
    for path in paths:
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

'''
#TODO: Currently, the script can convert lines and arcs into xyz coordinates. 
A next step would be to create a function that can handle (and detect) splines.
'''
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

# Example usage
svg_file = r"C:\Users\Windows\Documents\Cornell Tech\Spring 2024\INFO 5380 - Digital Fabrication\HW1\Git-Repository\INFO5380-HW1\svg_files\spline.svg"
output_file = r"C:\Users\Windows\Documents\Cornell Tech\Spring 2024\INFO 5380 - Digital Fabrication\HW1\Git-Repository\INFO5380-HW1\output.csv"
svg_to_xyz(svg_file, output_file)
