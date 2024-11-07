from shapely.geometry import Polygon
import numpy as np

def rotate_point(px, py, ox, oy, angle):
    """
    Rotate a point (px, py) around another point (ox, oy) by the given angle.
    """
    s, c = np.sin(angle), np.cos(angle)
    px, py = px - ox, py - oy
    new_x = px * c - py * s + ox
    new_y = px * s + py * c + oy
    return new_x, new_y

def get_rectangle_vertices(center_x, center_y, width, height, angle):
    """
    Calculate the vertices of a rectangle given its center, dimensions, and rotation angle.
    """
    hw, hh = width / 2, height / 2
    # Calculate unrotated rectangle vertices
    vertices = [
        (center_x - hw, center_y - hh),
        (center_x + hw, center_y - hh),
        (center_x + hw, center_y + hh),
        (center_x - hw, center_y + hh),
    ]
    # Rotate each vertex around the center
    rotated_vertices = [rotate_point(vx, vy, center_x, center_y, angle) for vx, vy in vertices]
    return rotated_vertices

def calculate_intersection_area(rect1, rect2):
    """
    Calculate the intersection area of two rotated rectangles.
    """
    vertices1 = get_rectangle_vertices(*rect1)
    vertices2 = get_rectangle_vertices(*rect2)
    
    polygon1 = Polygon(vertices1)
    polygon2 = Polygon(vertices2)
    
    # Calculate the intersection of the two polygons
    intersection = polygon1.intersection(polygon2)
    # print(intersection)
    # Return the area of the intersection
    return intersection.area



# Example usage
if __name__ == "__main__":
    rect1 = (0, 0, 2, 2, np.pi / 4)  # center_x, center_y, width, height, rotation angle
    rect2 = (0, 0, 2, 2, 0)

    overlap_area = calculate_intersection_area(rect1, rect2)
    print("Overlap Area:", overlap_area)
