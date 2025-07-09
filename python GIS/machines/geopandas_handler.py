import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# GEOPANDAS FUNCTIONS
# =============================================================================

def load_clean_vector_data(file_path, encoding='utf-8', clean_geometry=True, drop_invalid=True,
                           columns_to_keep=None, crs_override=None):
    """
    Load and clean vector data from various formats

    Args:
        file_path (str): Path to vector data file
        encoding (str): File encoding
        clean_geometry (bool): Clean invalid geometries
        drop_invalid (bool): Drop features with invalid geometries
        columns_to_keep (list): Specific columns to retain
        crs_override (str): Override CRS if not properly defined

    Returns:
        dict: JSON-like response with loaded data info and GeoDataFrame
    """
    try:
        # Load the vector data
        gdf = gpd.read_file(file_path, encoding=encoding)

        original_count = len(gdf)
        original_columns = list(gdf.columns)

        # Store original info
        result = {
            "status": "success",
            "file_path": file_path,
            "original_features": original_count,
            "original_columns": original_columns,
            "geometry_type": gdf.geom_type.value_counts().to_dict(),
            "original_crs": str(gdf.crs) if gdf.crs else None,
            "original_bounds": gdf.total_bounds.tolist() if not gdf.empty else None
        }

        # Override CRS if specified
        if crs_override:
            gdf = gdf.set_crs(crs_override, allow_override=True)
            result["crs_override_applied"] = crs_override

        # Clean geometries
        if clean_geometry:
            # Check for invalid geometries
            invalid_mask = ~gdf.geometry.is_valid
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                if drop_invalid:
                    gdf = gdf[~invalid_mask]
                    result["invalid_geometries_dropped"] = invalid_count
                else:
                    # Try to fix invalid geometries
                    gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
                    result["invalid_geometries_fixed"] = invalid_count

        # Remove empty geometries
        empty_mask = gdf.geometry.is_empty
        empty_count = empty_mask.sum()
        if empty_count > 0:
            gdf = gdf[~empty_mask]
            result["empty_geometries_removed"] = empty_count

        # Keep only specified columns
        if columns_to_keep:
            available_columns = [col for col in columns_to_keep if col in gdf.columns]
            if 'geometry' not in available_columns:
                available_columns.append('geometry')
            gdf = gdf[available_columns]
            result["columns_kept"] = available_columns
            result["columns_dropped"] = [col for col in original_columns if col not in available_columns]

        # Final statistics
        result.update({
            "final_features": len(gdf),
            "final_columns": list(gdf.columns),
            "features_removed": original_count - len(gdf),
            "final_crs": str(gdf.crs) if gdf.crs else None,
            "final_bounds": gdf.total_bounds.tolist() if not gdf.empty else None,
            "data_types": gdf.dtypes.astype(str).to_dict(),
            "null_counts": gdf.isnull().sum().to_dict(),
            "geodataframe": gdf  # Include the actual GeoDataFrame
        })

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load vector data: {str(e)}",
            "file_path": file_path,
            "geodataframe": None
        }


def filter_features_by_attribute(gdf, column_name, filter_value=None, filter_condition=None,
                                 filter_range=None, filter_list=None, case_sensitive=True):
    """
    Filter features based on attribute values

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame
        column_name (str): Column name to filter on
        filter_value (str/int/float): Exact value to match
        filter_condition (str): Condition like '>', '<', '>=', '<=', '!=', 'contains', 'startswith'
        filter_range (tuple): Range as (min_val, max_val)
        filter_list (list): List of values to match
        case_sensitive (bool): Case sensitivity for string operations

    Returns:
        dict: JSON-like response with filtered data
    """
    try:
        if not isinstance(gdf, gpd.GeoDataFrame):
            return {
                "status": "error",
                "message": "Input is not a valid GeoDataFrame",
                "original_features": 0,
                "filtered_features": 0
            }

        if column_name not in gdf.columns:
            return {
                "status": "error",
                "message": f"Column '{column_name}' not found in GeoDataFrame",
                "available_columns": list(gdf.columns),
                "original_features": len(gdf),
                "filtered_features": 0
            }

        original_count = len(gdf)
        filtered_gdf = gdf.copy()

        # Apply filters based on type
        if filter_value is not None:
            if isinstance(filter_value, str) and not case_sensitive:
                mask = filtered_gdf[column_name].astype(str).str.lower() == filter_value.lower()
            else:
                mask = filtered_gdf[column_name] == filter_value
            filtered_gdf = filtered_gdf[mask]
            filter_applied = f"Equal to '{filter_value}'"

        elif filter_condition and filter_value is not None:
            col_data = filtered_gdf[column_name]

            if filter_condition == '>':
                mask = col_data > filter_value
            elif filter_condition == '<':
                mask = col_data < filter_value
            elif filter_condition == '>=':
                mask = col_data >= filter_value
            elif filter_condition == '<=':
                mask = col_data <= filter_value
            elif filter_condition == '!=':
                mask = col_data != filter_value
            elif filter_condition == 'contains':
                if not case_sensitive:
                    mask = col_data.astype(str).str.lower().str.contains(str(filter_value).lower(), na=False)
                else:
                    mask = col_data.astype(str).str.contains(str(filter_value), na=False)
            elif filter_condition == 'startswith':
                if not case_sensitive:
                    mask = col_data.astype(str).str.lower().str.startswith(str(filter_value).lower())
                else:
                    mask = col_data.astype(str).str.startswith(str(filter_value))
            else:
                return {
                    "status": "error",
                    "message": f"Invalid filter condition: {filter_condition}",
                    "original_features": original_count
                }

            filtered_gdf = filtered_gdf[mask]
            filter_applied = f"{filter_condition} '{filter_value}'"

        elif filter_range:
            min_val, max_val = filter_range
            mask = (filtered_gdf[column_name] >= min_val) & (filtered_gdf[column_name] <= max_val)
            filtered_gdf = filtered_gdf[mask]
            filter_applied = f"Range [{min_val}, {max_val}]"

        elif filter_list:
            if isinstance(filter_list[0], str) and not case_sensitive:
                mask = filtered_gdf[column_name].astype(str).str.lower().isin([str(v).lower() for v in filter_list])
            else:
                mask = filtered_gdf[column_name].isin(filter_list)
            filtered_gdf = filtered_gdf[mask]
            filter_applied = f"In list: {filter_list}"

        else:
            return {
                "status": "error",
                "message": "No valid filter parameters provided",
                "original_features": original_count
            }

        filtered_count = len(filtered_gdf)

        return {
            "status": "success",
            "message": f"Filtered features successfully",
            "column_filtered": column_name,
            "filter_applied": filter_applied,
            "original_features": original_count,
            "filtered_features": filtered_count,
            "features_removed": original_count - filtered_count,
            "removal_percentage": round((original_count - filtered_count) / original_count * 100, 2),
            "unique_values_remaining": filtered_gdf[column_name].nunique(),
            "value_counts": filtered_gdf[column_name].value_counts().head(10).to_dict(),
            "geodataframe": filtered_gdf
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Filtering failed: {str(e)}",
            "original_features": len(gdf) if isinstance(gdf, gpd.GeoDataFrame) else 0,
            "filtered_features": 0
        }


def spatial_join_analysis(left_gdf, right_gdf, join_type='inner', spatial_predicate='intersects',
                          left_suffix='_left', right_suffix='_right'):
    """
    Perform spatial join analysis between two GeoDataFrames

    Args:
        left_gdf (GeoDataFrame): Left GeoDataFrame
        right_gdf (GeoDataFrame): Right GeoDataFrame
        join_type (str): Type of join ('inner', 'left', 'right')
        spatial_predicate (str): Spatial relationship ('intersects', 'within', 'contains', 'touches', 'crosses', 'overlaps')
        left_suffix (str): Suffix for left DataFrame columns
        right_suffix (str): Suffix for right DataFrame columns

    Returns:
        dict: JSON-like response with join results
    """
    try:
        if not isinstance(left_gdf, gpd.GeoDataFrame) or not isinstance(right_gdf, gpd.GeoDataFrame):
            return {
                "status": "error",
                "message": "Both inputs must be valid GeoDataFrames",
                "left_features": len(left_gdf) if isinstance(left_gdf, gpd.GeoDataFrame) else 0,
                "right_features": len(right_gdf) if isinstance(right_gdf, gpd.GeoDataFrame) else 0
            }

        # Ensure both GeoDataFrames have the same CRS
        if left_gdf.crs != right_gdf.crs:
            if left_gdf.crs and right_gdf.crs:
                right_gdf = right_gdf.to_crs(left_gdf.crs)
                crs_reprojected = True
            else:
                return {
                    "status": "error",
                    "message": "CRS mismatch and cannot reproject (missing CRS information)",
                    "left_crs": str(left_gdf.crs),
                    "right_crs": str(right_gdf.crs)
                }
        else:
            crs_reprojected = False

        # Store original info
        left_count = len(left_gdf)
        right_count = len(right_gdf)
        left_columns = list(left_gdf.columns)
        right_columns = list(right_gdf.columns)

        # Perform spatial join
        joined_gdf = gpd.sjoin(
            left_gdf,
            right_gdf,
            how=join_type,
            predicate=spatial_predicate,
            lsuffix=left_suffix,
            rsuffix=right_suffix
        )

        # Analyze join results
        joined_count = len(joined_gdf)

        # Count matches per left feature
        if 'index_right' in joined_gdf.columns:
            matches_per_left = joined_gdf.groupby(joined_gdf.index)['index_right'].count()
            avg_matches = matches_per_left.mean()
            max_matches = matches_per_left.max()
            features_with_matches = len(matches_per_left)
        else:
            avg_matches = 0
            max_matches = 0
            features_with_matches = 0

        # Calculate spatial overlap statistics if possible
        overlap_stats = None
        try:
            if spatial_predicate == 'intersects':
                # Calculate intersection areas for area-based geometries
                if left_gdf.geom_type.iloc[0] in ['Polygon', 'MultiPolygon'] and \
                        right_gdf.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
                    intersections = []
                    for idx, row in joined_gdf.iterrows():
                        left_geom = row.geometry
                        right_idx = row.get('index_right')
                        if right_idx is not None and right_idx < len(right_gdf):
                            right_geom = right_gdf.iloc[right_idx].geometry
                            intersection = left_geom.intersection(right_geom)
                            if not intersection.is_empty:
                                intersections.append(intersection.area)

                    if intersections:
                        overlap_stats = {
                            "total_intersection_area": sum(intersections),
                            "avg_intersection_area": np.mean(intersections),
                            "max_intersection_area": max(intersections),
                            "min_intersection_area": min(intersections)
                        }
        except:
            pass

        return {
            "status": "success",
            "message": f"Spatial join completed successfully",
            "join_type": join_type,
            "spatial_predicate": spatial_predicate,
            "left_features": left_count,
            "right_features": right_count,
            "joined_features": joined_count,
            "left_columns": left_columns,
            "right_columns": right_columns,
            "joined_columns": list(joined_gdf.columns),
            "crs_reprojected": crs_reprojected,
            "final_crs": str(joined_gdf.crs),
            "features_with_matches": features_with_matches,
            "avg_matches_per_feature": round(avg_matches, 2) if avg_matches > 0 else 0,
            "max_matches_per_feature": max_matches,
            "join_efficiency": round(joined_count / left_count * 100, 2) if left_count > 0 else 0,
            "overlap_statistics": overlap_stats,
            "geodataframe": joined_gdf
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Spatial join failed: {str(e)}",
            "join_type": join_type,
            "spatial_predicate": spatial_predicate,
            "left_features": len(left_gdf) if isinstance(left_gdf, gpd.GeoDataFrame) else 0,
            "right_features": len(right_gdf) if isinstance(right_gdf, gpd.GeoDataFrame) else 0
        }


def create_buffer_zones(gdf, buffer_distance, resolution=16, cap_style='round', join_style='round',
                        dissolve_buffers=False, buffer_column_name='buffer_dist'):
    """
    Create buffer zones around geometries

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame
        buffer_distance (float/list): Buffer distance(s) in map units
        resolution (int): Number of segments to approximate curves
        cap_style (str): Buffer cap style ('round', 'flat', 'square')
        join_style (str): Buffer join style ('round', 'mitre', 'bevel')
        dissolve_buffers (bool): Dissolve overlapping buffers
        buffer_column_name (str): Column name to store buffer distance

    Returns:
        dict: JSON-like response with buffered data
    """
    try:
        if not isinstance(gdf, gpd.GeoDataFrame):
            return {
                "status": "error",
                "message": "Input is not a valid GeoDataFrame",
                "original_features": 0
            }

        if gdf.empty:
            return {
                "status": "error",
                "message": "Input GeoDataFrame is empty",
                "original_features": 0
            }

        # Check if CRS is projected (needed for accurate distance calculations)
        is_projected = gdf.crs and gdf.crs.is_projected if gdf.crs else False

        original_count = len(gdf)

        # Handle multiple buffer distances
        if isinstance(buffer_distance, (list, tuple)):
            buffer_distances = buffer_distance
        else:
            buffer_distances = [buffer_distance]

        # Cap style mapping
        cap_style_dict = {'round': 1, 'flat': 2, 'square': 3}
        join_style_dict = {'round': 1, 'mitre': 2, 'bevel': 3}

        cap_style_val = cap_style_dict.get(cap_style, 1)
        join_style_val = join_style_dict.get(join_style, 1)

        all_buffers = []
        buffer_stats = []

        for dist in buffer_distances:
            # Create buffers
            buffered_gdf = gdf.copy()
            buffered_gdf['geometry'] = gdf.geometry.buffer(
                distance=dist,
                resolution=resolution,
                cap_style=cap_style_val,
                join_style=join_style_val
            )

            # Add buffer distance column
            buffered_gdf[buffer_column_name] = dist

            # Calculate buffer statistics
            buffer_areas = buffered_gdf.geometry.area
            original_areas = gdf.geometry.area

            stats = {
                "buffer_distance": dist,
                "total_buffer_area": buffer_areas.sum(),
                "avg_buffer_area": buffer_areas.mean(),
                "min_buffer_area": buffer_areas.min(),
                "max_buffer_area": buffer_areas.max(),
                "area_increase_ratio": (buffer_areas / original_areas).mean() if original_areas.sum() > 0 else None
            }
            buffer_stats.append(stats)

            # Dissolve buffers if requested
            if dissolve_buffers:
                dissolved = buffered_gdf.dissolve()
                dissolved[buffer_column_name] = dist
                buffered_gdf = dissolved.reset_index(drop=True)

            all_buffers.append(buffered_gdf)

        # Combine all buffers if multiple distances
        if len(all_buffers) > 1:
            result_gdf = pd.concat(all_buffers, ignore_index=True)
        else:
            result_gdf = all_buffers[0]

        return {
            "status": "success",
            "message": f"Buffer zones created successfully",
            "original_features": original_count,
            "buffered_features": len(result_gdf),
            "buffer_distances": buffer_distances,
            "buffer_parameters": {
                "resolution": resolution,
                "cap_style": cap_style,
                "join_style": join_style,
                "dissolved": dissolve_buffers
            },
            "crs_is_projected": is_projected,
            "crs_warning": "Buffer distances are in map units. Ensure CRS is appropriate for distance measurements." if not is_projected else None,
            "buffer_statistics": buffer_stats,
            "total_buffer_area": result_gdf.geometry.area.sum(),
            "original_bounds": gdf.total_bounds.tolist(),
            "buffered_bounds": result_gdf.total_bounds.tolist(),
            "geodataframe": result_gdf
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Buffer creation failed: {str(e)}",
            "original_features": len(gdf) if isinstance(gdf, gpd.GeoDataFrame) else 0,
            "buffer_distance": buffer_distance
        }


def convert_crs_geodata(gdf, target_crs, preserve_original=True):
    """
    Convert GeoDataFrame to different coordinate reference system

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame
        target_crs (str/int): Target CRS (e.g., 'EPSG:4326', 4326, '+proj=...')
        preserve_original (bool): Keep original geometry in separate column

    Returns:
        dict: JSON-like response with CRS conversion results
    """
    try:
        if not isinstance(gdf, gpd.GeoDataFrame):
            return {
                "status": "error",
                "message": "Input is not a valid GeoDataFrame",
                "original_crs": None,
                "target_crs": target_crs
            }

        original_crs = str(gdf.crs) if gdf.crs else None
        original_bounds = gdf.total_bounds.tolist() if not gdf.empty else None
        original_count = len(gdf)

        # Check if conversion is needed
        if str(gdf.crs) == str(target_crs):
            return {
                "status": "success",
                "message": "No conversion needed - already in target CRS",
                "original_crs": original_crs,
                "target_crs": str(target_crs),
                "conversion_performed": False,
                "features_count": original_count,
                "geodataframe": gdf
            }

        # Preserve original geometry if requested
        if preserve_original:
            gdf = gdf.copy()
            gdf['geometry_original'] = gdf.geometry

        # Perform CRS conversion
        converted_gdf = gdf.to_crs(target_crs)

        # Calculate conversion statistics
        converted_bounds = converted_gdf.total_bounds.tolist() if not converted_gdf.empty else None

        # Calculate area change if geometries are polygons
        area_change_stats = None
        if not gdf.empty and gdf.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
            try:
                if gdf.crs and gdf.crs.is_projected:
                    original_areas = gdf.geometry.area
                else:
                    # Convert to a projected CRS for area calculation
                    temp_gdf = gdf.to_crs('EPSG:3857')  # Web Mercator for rough area calculation
                    original_areas = temp_gdf.geometry.area

                if converted_gdf.crs and converted_gdf.crs.is_projected:
                    converted_areas = converted_gdf.geometry.area
                else:
                    # Convert to a projected CRS for area calculation
                    temp_converted = converted_gdf.to_crs('EPSG:3857')
                    converted_areas = temp_converted.geometry.area

                area_change_stats = {
                    "original_total_area": original_areas.sum(),
                    "converted_total_area": converted_areas.sum(),
                    "area_change_ratio": converted_areas.sum() / original_areas.sum() if original_areas.sum() > 0 else None,
                    "avg_area_change": (converted_areas - original_areas).mean()
                }
            except:
                pass

        # Determine CRS types
        original_crs_type = "Unknown"
        target_crs_type = "Unknown"

        try:
            if gdf.crs:
                original_crs_type = "Projected" if gdf.crs.is_projected else "Geographic"
            if converted_gdf.crs:
                target_crs_type = "Projected" if converted_gdf.crs.is_projected else "Geographic"
        except:
            pass

        return {
            "status": "success",
            "message": f"CRS conversion completed successfully",
            "original_crs": original_crs,
            "target_crs": str(converted_gdf.crs),
            "original_crs_type": original_crs_type,
            "target_crs_type": target_crs_type,
            "conversion_performed": True,
            "features_count": len(converted_gdf),
            "original_bounds": original_bounds,
            "converted_bounds": converted_bounds,
            "bounds_change": {
                "x_min_change": converted_bounds[0] - original_bounds[
                    0] if original_bounds and converted_bounds else None,
                "y_min_change": converted_bounds[1] - original_bounds[
                    1] if original_bounds and converted_bounds else None,
                "x_max_change": converted_bounds[2] - original_bounds[
                    2] if original_bounds and converted_bounds else None,
                "y_max_change": converted_bounds[3] - original_bounds[
                    3] if original_bounds and converted_bounds else None
            },
            "area_change_statistics": area_change_stats,
            "original_preserved": preserve_original,
            "geodataframe": converted_gdf
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"CRS conversion failed: {str(e)}",
            "original_crs": str(gdf.crs) if isinstance(gdf, gpd.GeoDataFrame) and gdf.crs else None,
            "target_crs": str(target_crs),
            "conversion_performed": False
        }


def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")
