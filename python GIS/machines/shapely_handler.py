import json
import math
from typing import List, Dict, Any, Tuple, Union


def create_geometries(geometry_type: str, coordinates: List, properties: Dict = None) -> Dict[str, Any]:
    """
    Create geometry objects from input data

    Args:
        geometry_type: Type of geometry (Point, LineString, Polygon, etc.)
        coordinates: List of coordinate pairs/arrays
        properties: Optional properties dictionary

    Returns:
        JSON-like dictionary with geometry data
    """
    try:
        # Validate geometry type
        valid_types = ['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon']
        if geometry_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid geometry type. Must be one of: {valid_types}",
                "geometry": None
            }

        # Validate coordinates based on geometry type
        if geometry_type == 'Point' and len(coordinates) != 2:
            return {
                "success": False,
                "error": "Point coordinates must be [x, y]",
                "geometry": None
            }

        geometry = {
            "type": geometry_type,
            "coordinates": coordinates,
            "properties": properties or {}
        }

        return {
            "success": True,
            "error": None,
            "geometry": geometry,
            "bounds": _calculate_bounds(coordinates, geometry_type)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "geometry": None
        }

def check_geometry_intersections(geometry1: Dict, geometry2: Dict) -> Dict[str, Any]:
    """
    Check if two geometries intersect

    Args:
        geometry1: First geometry object
        geometry2: Second geometry object

    Returns:
        JSON-like dictionary with intersection results
    """
    try:
        # Extract coordinates and types
        coords1 = geometry1.get('coordinates', [])
        coords2 = geometry2.get('coordinates', [])
        type1 = geometry1.get('type', '')
        type2 = geometry2.get('type', '')

        # Simple intersection check for basic cases
        intersects = False
        intersection_points = []

        # Point-Point intersection
        if type1 == 'Point' and type2 == 'Point':
            intersects = coords1 == coords2
            if intersects:
                intersection_points = [coords1]

        # Point-LineString intersection
        elif (type1 == 'Point' and type2 == 'LineString') or (type1 == 'LineString' and type2 == 'Point'):
            point = coords1 if type1 == 'Point' else coords2
            line = coords2 if type2 == 'LineString' else coords1
            intersects = _point_on_line(point, line)
            if intersects:
                intersection_points = [point]

        # LineString-LineString intersection
        elif type1 == 'LineString' and type2 == 'LineString':
            intersects, intersection_points = _line_line_intersection(coords1, coords2)

        # Bounding box intersection for complex cases
        else:
            bounds1 = _calculate_bounds(coords1, type1)
            bounds2 = _calculate_bounds(coords2, type2)
            intersects = _bounds_intersect(bounds1, bounds2)

        return {
            "success": True,
            "intersects": intersects,
            "intersection_points": intersection_points,
            "geometry1_type": type1,
            "geometry2_type": type2,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "intersects": False,
            "intersection_points": [],
            "error": str(e)
        }

def calculate_geometry_distance(geometry1: Dict, geometry2: Dict) -> Dict[str, Any]:
    """
    Calculate distance between two geometries

    Args:
        geometry1: First geometry object
        geometry2: Second geometry object

    Returns:
        JSON-like dictionary with distance calculation results
    """
    try:
        coords1 = geometry1.get('coordinates', [])
        coords2 = geometry2.get('coordinates', [])
        type1 = geometry1.get('type', '')
        type2 = geometry2.get('type', '')

        distance = 0.0
        closest_points = []

        # Point-Point distance
        if type1 == 'Point' and type2 == 'Point':
            distance = _euclidean_distance(coords1, coords2)
            closest_points = [coords1, coords2]

        # Point-LineString distance
        elif (type1 == 'Point' and type2 == 'LineString') or (type1 == 'LineString' and type2 == 'Point'):
            point = coords1 if type1 == 'Point' else coords2
            line = coords2 if type2 == 'LineString' else coords1
            distance, closest_point = _point_to_line_distance(point, line)
            closest_points = [point, closest_point]

        # LineString-LineString distance
        elif type1 == 'LineString' and type2 == 'LineString':
            distance, closest_points = _line_to_line_distance(coords1, coords2)

        # General case - use centroids
        else:
            centroid1 = _calculate_centroid(coords1, type1)
            centroid2 = _calculate_centroid(coords2, type2)
            distance = _euclidean_distance(centroid1, centroid2)
            closest_points = [centroid1, centroid2]

        return {
            "success": True,
            "distance": round(distance, 6),
            "closest_points": closest_points,
            "geometry1_type": type1,
            "geometry2_type": type2,
            "units": "coordinate_units",
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "distance": None,
            "closest_points": [],
            "error": str(e)
        }

def generate_convex_hull(points: List[List[float]]) -> Dict[str, Any]:
    """
    Generate convex hull from a set of points

    Args:
        points: List of [x, y] coordinate pairs

    Returns:
        JSON-like dictionary with convex hull results
    """
    try:
        if len(points) < 3:
            return {
                "success": False,
                "error": "Need at least 3 points to generate convex hull",
                "hull_points": [],
                "hull_geometry": None
            }

        # Graham scan algorithm for convex hull
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Sort points lexicographically
        points = sorted(set(tuple(p) for p in points))
        points = [list(p) for p in points]

        if len(points) <= 1:
            return {
                "success": True,
                "hull_points": points,
                "hull_geometry": {
                    "type": "Point" if len(points) == 1 else "MultiPoint",
                    "coordinates": points[0] if len(points) == 1 else points
                },
                "area": 0.0,
                "perimeter": 0.0,
                "error": None
            }

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Remove last point of each half because it's repeated
        hull_points = lower[:-1] + upper[:-1]

        # Calculate area and perimeter
        area = _polygon_area(hull_points)
        perimeter = _polygon_perimeter(hull_points)

        hull_geometry = {
            "type": "Polygon",
            "coordinates": [hull_points + [hull_points[0]]]  # Close the polygon
        }

        return {
            "success": True,
            "hull_points": hull_points,
            "hull_geometry": hull_geometry,
            "area": round(area, 6),
            "perimeter": round(perimeter, 6),
            "point_count": len(hull_points),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "hull_points": [],
            "hull_geometry": None,
            "error": str(e)
        }

def fix_invalid_geometries(geometry: Dict) -> Dict[str, Any]:
    """
    Fix common geometry validation issues

    Args:
        geometry: Geometry object to validate and fix

    Returns:
        JSON-like dictionary with fixed geometry and validation results
    """
    try:
        original_geometry = geometry.copy()
        fixed_geometry = geometry.copy()
        issues_found = []
        fixes_applied = []

        geometry_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])

        # Check for empty coordinates
        if not coordinates:
            issues_found.append("Empty coordinates")
            return {
                "success": False,
                "original_geometry": original_geometry,
                "fixed_geometry": None,
                "issues_found": issues_found,
                "fixes_applied": [],
                "error": "Cannot fix geometry with empty coordinates"
            }

        # Fix Point geometry
        if geometry_type == 'Point':
            if len(coordinates) != 2:
                issues_found.append("Invalid Point coordinates")
                if len(coordinates) > 2:
                    fixed_geometry['coordinates'] = coordinates[:2]
                    fixes_applied.append("Truncated coordinates to [x, y]")

        # Fix LineString geometry
        elif geometry_type == 'LineString':
            if len(coordinates) < 2:
                issues_found.append("LineString must have at least 2 points")
            else:
                # Remove duplicate consecutive points
                clean_coords = [coordinates[0]]
                for i in range(1, len(coordinates)):
                    if coordinates[i] != coordinates[i - 1]:
                        clean_coords.append(coordinates[i])

                if len(clean_coords) != len(coordinates):
                    issues_found.append("Duplicate consecutive points")
                    fixed_geometry['coordinates'] = clean_coords
                    fixes_applied.append("Removed duplicate consecutive points")

        # Fix Polygon geometry
        elif geometry_type == 'Polygon':
            if not coordinates or not coordinates[0]:
                issues_found.append("Invalid Polygon coordinates")
            else:
                exterior_ring = coordinates[0]

                # Check if polygon is closed
                if len(exterior_ring) < 4:
                    issues_found.append("Polygon must have at least 4 points")
                elif exterior_ring[0] != exterior_ring[-1]:
                    issues_found.append("Polygon not closed")
                    exterior_ring.append(exterior_ring[0])
                    fixed_geometry['coordinates'][0] = exterior_ring
                    fixes_applied.append("Closed polygon ring")

                # Check winding order (should be counter-clockwise for exterior)
                if len(exterior_ring) >= 4:
                    area = _polygon_area(exterior_ring[:-1])  # Exclude closing point
                    if area < 0:
                        issues_found.append("Incorrect winding order")
                        fixed_geometry['coordinates'][0] = exterior_ring[::-1]
                        fixes_applied.append("Fixed winding order")

        # Validate coordinate values
        flat_coords = _flatten_coordinates(coordinates)
        for i, coord in enumerate(flat_coords):
            if not isinstance(coord, (int, float)) or math.isnan(coord) or math.isinf(coord):
                issues_found.append(f"Invalid coordinate value at position {i}: {coord}")

        # Calculate bounds for fixed geometry
        bounds = _calculate_bounds(fixed_geometry['coordinates'], geometry_type)

        return {
            "success": True,
            "original_geometry": original_geometry,
            "fixed_geometry": fixed_geometry,
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "is_valid": len(issues_found) == 0,
            "bounds": bounds,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "original_geometry": geometry,
            "fixed_geometry": None,
            "issues_found": [],
            "fixes_applied": [],
            "error": str(e)
        }

def _calculate_bounds(coordinates: List, geometry_type: str) -> Dict[str, float]:
    """Calculate bounding box for coordinates"""
    flat_coords = _flatten_coordinates(coordinates)
    if not flat_coords:
        return {"minx": 0, "miny": 0, "maxx": 0, "maxy": 0}

    x_coords = [coord for i, coord in enumerate(flat_coords) if i % 2 == 0]
    y_coords = [coord for i, coord in enumerate(flat_coords) if i % 2 == 1]

    return {
        "minx": min(x_coords),
        "miny": min(y_coords),
        "maxx": max(x_coords),
        "maxy": max(y_coords)
    }

def _flatten_coordinates(coordinates: List) -> List[float]:
    """Flatten nested coordinate arrays"""
    result = []
    for item in coordinates:
        if isinstance(item, list):
            if len(item) == 2 and all(isinstance(x, (int, float)) for x in item):
                result.extend(item)
            else:
                result.extend(_flatten_coordinates(item))
        elif isinstance(item, (int, float)):
            result.append(item)
    return result

def _euclidean_distance(p1: List[float], p2: List[float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def _point_on_line(point: List[float], line: List[List[float]]) -> bool:
    """Check if point lies on line segment"""
    for i in range(len(line) - 1):
        if _point_on_segment(point, line[i], line[i + 1]):
            return True
    return False

def _point_on_segment(point: List[float], seg_start: List[float], seg_end: List[float]) -> bool:
    """Check if point lies on line segment"""
    # Check if point is collinear with segment
    cross_product = (point[1] - seg_start[1]) * (seg_end[0] - seg_start[0]) - (point[0] - seg_start[0]) * (
                seg_end[1] - seg_start[1])
    if abs(cross_product) > 1e-10:
        return False

    # Check if point is within segment bounds
    return (min(seg_start[0], seg_end[0]) <= point[0] <= max(seg_start[0], seg_end[0]) and
            min(seg_start[1], seg_end[1]) <= point[1] <= max(seg_start[1], seg_end[1]))

def _line_line_intersection(line1: List[List[float]], line2: List[List[float]]) -> Tuple[bool, List[List[float]]]:
    """Find intersection points between two lines"""
    intersections = []
    for i in range(len(line1) - 1):
        for j in range(len(line2) - 1):
            intersection = _segment_intersection(line1[i], line1[i + 1], line2[j], line2[j + 1])
            if intersection:
                intersections.append(intersection)

    return len(intersections) > 0, intersections

def _segment_intersection(p1: List[float], p2: List[float], p3: List[float], p4: List[float]) -> List[float]:
    """Find intersection point between two line segments"""
    denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if abs(denom) < 1e-10:
        return None

    t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom
    u = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]

    return None

def _bounds_intersect(bounds1: Dict, bounds2: Dict) -> bool:
    """Check if two bounding boxes intersect"""
    return not (bounds1["maxx"] < bounds2["minx"] or bounds2["maxx"] < bounds1["minx"] or
                bounds1["maxy"] < bounds2["miny"] or bounds2["maxy"] < bounds1["miny"])

def _point_to_line_distance(point: List[float], line: List[List[float]]) -> Tuple[float, List[float]]:
    """Calculate minimum distance from point to line"""
    min_dist = float('inf')
    closest_point = None

    for i in range(len(line) - 1):
        dist, closest = _point_to_segment_distance(point, line[i], line[i + 1])
        if dist < min_dist:
            min_dist = dist
            closest_point = closest

    return min_dist, closest_point

def _point_to_segment_distance(point: List[float], seg_start: List[float], seg_end: List[float]) -> Tuple[float, List[float]]:
    """Calculate distance from point to line segment"""
    # Vector from seg_start to seg_end
    seg_vec = [seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]]
    # Vector from seg_start to point
    point_vec = [point[0] - seg_start[0], point[1] - seg_start[1]]

    # Project point onto line segment
    seg_len_sq = seg_vec[0] ** 2 + seg_vec[1] ** 2
    if seg_len_sq == 0:
        return _euclidean_distance(point, seg_start), seg_start

    t = max(0, min(1, (point_vec[0] * seg_vec[0] + point_vec[1] * seg_vec[1]) / seg_len_sq))
    projection = [seg_start[0] + t * seg_vec[0], seg_start[1] + t * seg_vec[1]]

    return _euclidean_distance(point, projection), projection

def _line_to_line_distance(line1: List[List[float]], line2: List[List[float]]) -> Tuple[float, List[List[float]]]:
    """Calculate minimum distance between two lines"""
    min_dist = float('inf')
    closest_points = []

    for point1 in line1:
        dist, closest = _point_to_line_distance(point1, line2)
        if dist < min_dist:
            min_dist = dist
            closest_points = [point1, closest]

    for point2 in line2:
        dist, closest = _point_to_line_distance(point2, line1)
        if dist < min_dist:
            min_dist = dist
            closest_points = [closest, point2]

    return min_dist, closest_points

def _calculate_centroid(coordinates: List, geometry_type: str) -> List[float]:
    """Calculate centroid of geometry"""
    flat_coords = _flatten_coordinates(coordinates)
    if not flat_coords:
        return [0, 0]

    x_coords = [coord for i, coord in enumerate(flat_coords) if i % 2 == 0]
    y_coords = [coord for i, coord in enumerate(flat_coords) if i % 2 == 1]

    return [sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]

def _polygon_area(points: List[List[float]]) -> float:
    """Calculate polygon area using shoelace formula"""
    if len(points) < 3:
        return 0

    area = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2

def _polygon_perimeter(points: List[List[float]]) -> float:
    """Calculate polygon perimeter"""
    if len(points) < 2:
        return 0

    perimeter = 0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        perimeter += _euclidean_distance(points[i], points[j])

    return perimeter


def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")
