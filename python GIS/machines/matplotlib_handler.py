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

def plot_static_thematic_map(gdf, column=None, figsize=(12, 8), cmap='viridis', title=None,
                             legend=True, save_path=None, dpi=300):
    """
    Create a static thematic map

    Args:
        gdf (GeoDataFrame): GeoDataFrame to plot
        column (str): Column name for choropleth mapping
        figsize (tuple): Figure size (width, height)
        cmap (str): Color map name
        title (str): Map title
        legend (bool): Show legend
        save_path (str): Path to save the map
        dpi (int): Resolution for saved image

    Returns:
        dict: JSON-like response with plot information
    """
    try:
        if not isinstance(gdf, gpd.GeoDataFrame):
            return {
                "status": "error",
                "message": "Input is not a valid GeoDataFrame",
                "plot_created": False
            }

        if gdf.empty:
            return {
                "status": "error",
                "message": "GeoDataFrame is empty",
                "plot_created": False
            }

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot parameters
        plot_params = {
            'ax': ax,
            'legend': legend,
            'cmap': cmap
        }

        # Add column for choropleth if specified
        if column and column in gdf.columns:
            plot_params['column'] = column

            # Get column statistics
            col_stats = {
                "min_value": gdf[column].min(),
                "max_value": gdf[column].max(),
                "mean_value": gdf[column].mean(),
                "std_value": gdf[column].std(),
                "null_count": gdf[column].isnull().sum(),
                "data_type": str(gdf[column].dtype)
            }
        else:
            col_stats = None
            plot_params['color'] = 'lightblue'
            plot_params['edgecolor'] = 'black'
            plot_params['linewidth'] = 0.5

        # Create the plot
        gdf.plot(**plot_params)

        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif column:
            ax.set_title(f'Thematic Map: {column}', fontsize=14, fontweight='bold')
        else:
            ax.set_title('Geographic Features', fontsize=14, fontweight='bold')

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Add coordinate information
        bounds = gdf.total_bounds
        ax.text(0.02, 0.02, f'CRS: {gdf.crs}', transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        plt.tight_layout()

        # Save if path provided
        saved = False
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                saved = True
            except Exception as e:
                saved = False
                save_error = str(e)

        return {
            "status": "success",
            "message": "Static thematic map created successfully",
            "plot_created": True,
            "features_plotted": len(gdf),
            "column_mapped": column,
            "column_statistics": col_stats,
            "map_bounds": bounds.tolist(),
            "crs": str(gdf.crs),
            "figure_size": figsize,
            "colormap": cmap,
            "legend_shown": legend,
            "title": title or (f'Thematic Map: {column}' if column else 'Geographic Features'),
            "saved_to_file": saved,
            "save_path": save_path if saved else None,
            "save_error": save_error if not saved and save_path else None,
            "matplotlib_figure": fig
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Plot creation failed: {str(e)}",
            "plot_created": False,
            "features_plotted": len(gdf) if isinstance(gdf, gpd.GeoDataFrame) else 0
        }


def overlay_layers_on_map(base_gdf, overlay_gdfs, figsize=(15, 10), base_color='lightgray',
                          overlay_colors=None, alpha=0.7, title="Multi-Layer Map", save_path=None, dpi=300):
    """
    Create a map with multiple overlaid layers

    Args:
        base_gdf (GeoDataFrame): Base layer GeoDataFrame
        overlay_gdfs (list): List of GeoDataFrames to overlay
        figsize (tuple): Figure size (width, height)
        base_color (str): Color for base layer
        overlay_colors (list): Colors for overlay layers
        alpha (float): Transparency for overlay layers
        title (str): Map title
        save_path (str): Path to save the map
        dpi (int): Resolution for saved image

    Returns:
        dict: JSON-like response with overlay map information
    """
    try:
        if not isinstance(base_gdf, gpd.GeoDataFrame):
            return {
                "status": "error",
                "message": "Base layer is not a valid GeoDataFrame",
                "plot_created": False
            }

        if not overlay_gdfs or not isinstance(overlay_gdfs, list):
            return {
                "status": "error",
                "message": "Overlay layers must be provided as a list of GeoDataFrames",
                "plot_created": False
            }

        # Validate overlay GeoDataFrames
        valid_overlays = []
        for i, gdf in enumerate(overlay_gdfs):
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                valid_overlays.append((i, gdf))

        if not valid_overlays:
            return {
                "status": "error",
                "message": "No valid overlay GeoDataFrames found",
                "plot_created": False
            }

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot base layer
        base_gdf.plot(ax=ax, color=base_color, edgecolor='white', linewidth=0.5, alpha=0.5)

        # Default colors for overlays
        if not overlay_colors:
            overlay_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'yellow']

        # Ensure CRS compatibility
        target_crs = base_gdf.crs
        reprojected_layers = []

        # Plot overlay layers
        layer_info = []
        for i, (original_idx, overlay_gdf) in enumerate(valid_overlays):
            try:
                # Reproject if necessary
                if overlay_gdf.crs != target_crs:
                    overlay_gdf = overlay_gdf.to_crs(target_crs)
                    reprojected_layers.append(original_idx)

                # Get color for this layer
                color = overlay_colors[i % len(overlay_colors)]

                # Plot the overlay
                overlay_gdf.plot(ax=ax, color=color, alpha=alpha, edgecolor='black', linewidth=0.3)

                # Store layer information
                layer_info.append({
                    "layer_index": original_idx,
                    "features_count": len(overlay_gdf),
                    "geometry_type": overlay_gdf.geom_type.value_counts().to_dict(),
                    "color": color,
                    "bounds": overlay_gdf.total_bounds.tolist(),
                    "crs": str(overlay_gdf.crs),
                    "reprojected": original_idx in reprojected_layers
                })

            except Exception as e:
                layer_info.append({
                    "layer_index": original_idx,
                    "error": f"Failed to plot layer: {str(e)}"
                })

        # Set title and styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])

        # Create custom legend
        legend_elements = [mpatches.Patch(color=base_color, alpha=0.5, label='Base Layer')]
        for info in layer_info:
            if 'color' in info:
                legend_elements.append(
                    mpatches.Patch(color=info['color'], alpha=alpha,
                                   label=f'Overlay {info["layer_index"] + 1}')
                )

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

        # Add map information
        total_bounds = base_gdf.total_bounds
        for info in layer_info:
            if 'bounds' in info:
                bounds = info['bounds']
                total_bounds[0] = min(total_bounds[0], bounds[0])  # min_x
                total_bounds[1] = min(total_bounds[1], bounds[1])  # min_y
                total_bounds[2] = max(total_bounds[2], bounds[2])  # max_x
                total_bounds[3] = max(total_bounds[3], bounds[3])  # max_y

        # Add scale and CRS info
        ax.text(0.02, 0.02, f'CRS: {target_crs}', transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()

        # Save if path provided
        saved = False
        save_error = None
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                saved = True
            except Exception as e:
                save_error = str(e)

        return {
            "status": "success",
            "message": "Multi-layer map created successfully",
            "plot_created": True,
            "base_layer_features": len(base_gdf),
            "overlay_layers_processed": len(layer_info),
            "layers_reprojected": len(reprojected_layers),
            "target_crs": str(target_crs),
            "layer_details": layer_info,
            "map_bounds": total_bounds.tolist(),
            "figure_size": figsize,
            "title": title,
            "saved_to_file": saved,
            "save_path": save_path if saved else None,
            "save_error": save_error,
            "matplotlib_figure": fig
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Overlay map creation failed: {str(e)}",
            "plot_created": False
        }


def export_map_image(fig, output_path, format='png', dpi=300, bbox_inches='tight',
                     transparent=False, facecolor='white'):
    """
    Export matplotlib figure to image file

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure to export
        output_path (str): Output file path
        format (str): Image format ('png', 'jpg', 'pdf', 'svg', 'eps')
        dpi (int): Resolution for raster formats
        bbox_inches (str): Bounding box ('tight' or None)
        transparent (bool): Transparent background
        facecolor (str): Background color

    Returns:
        dict: JSON-like response with export information
    """
    try:
        if not hasattr(fig, 'savefig'):
            return {
                "status": "error",
                "message": "Input is not a valid matplotlib figure",
                "exported": False,
                "output_path": output_path
            }

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare save parameters
        save_params = {
            'fname': output_path,
            'format': format,
            'dpi': dpi,
            'bbox_inches': bbox_inches,
            'transparent': transparent,
            'facecolor': facecolor
        }

        # Remove DPI for vector formats
        if format.lower() in ['svg', 'eps', 'pdf']:
            save_params.pop('dpi', None)

        # Save the figure
        fig.savefig(**save_params)

        # Get file information
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            file_size_mb = round(file_size / (1024 * 1024), 2)

            return {
                "status": "success",
                "message": f"Map exported successfully as {format.upper()}",
                "exported": True,
                "output_path": output_path,
                "format": format,
                "file_size_bytes": file_size,
                "file_size_mb": file_size_mb,
                "dpi": dpi if format.lower() not in ['svg', 'eps', 'pdf'] else None,
                "transparent_background": transparent,
                "background_color": facecolor if not transparent else None,
                "bbox_inches": bbox_inches,
                "export_timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "File was not created",
                "exported": False,
                "output_path": output_path
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Export failed: {str(e)}",
            "exported": False,
            "output_path": output_path,
            "format": format
        }


def plot_map_time_series(gdf_list, time_column, figsize=(20, 12), columns=4, title_prefix="Time Step",
                         cmap='viridis', save_path=None, dpi=300):
    """
    Create a time series of maps showing temporal changes

    Args:
        gdf_list (list): List of GeoDataFrames for different time periods
        time_column (str): Column name containing time/date information
        figsize (tuple): Overall figure size
        columns (int): Number of columns in subplot grid
        title_prefix (str): Prefix for subplot titles
        cmap (str): Color map for choropleth
        save_path (str): Path to save the time series plot
        dpi (int): Resolution for saved image

    Returns:
        dict: JSON-like response with time series plot information
    """
    try:
        if not gdf_list or not isinstance(gdf_list, list):
            return {
                "status": "error",
                "message": "Input must be a list of GeoDataFrames",
                "plot_created": False
            }

        # Validate GeoDataFrames
        valid_gdfs = []
        time_info = []

        for i, gdf in enumerate(gdf_list):
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                if time_column in gdf.columns:
                    valid_gdfs.append(gdf)
                    # Get time range for this GeoDataFrame
                    time_values = gdf[time_column].dropna()
                    if not time_values.empty:
                        time_info.append({
                            "index": i,
                            "min_time": str(time_values.min()),
                            "max_time": str(time_values.max()),
                            "time_count": len(time_values.unique()),
                            "feature_count": len(gdf)
                        })

        if not valid_gdfs:
            return {
                "status": "error",
                "message": f"No valid GeoDataFrames with '{time_column}' column found",
                "plot_created": False
            }

        # Calculate subplot layout
        n_plots = len(valid_gdfs)
        rows = (n_plots + columns - 1) // columns

        # Create subplots
        fig, axes = plt.subplots(rows, columns, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        # Ensure CRS consistency
        target_crs = valid_gdfs[0].crs
        reprojected_count = 0

        # Get global bounds for consistent extent
        all_bounds = []
        for gdf in valid_gdfs:
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
                reprojected_count += 1
            all_bounds.append(gdf.total_bounds)

        global_bounds = np.array(all_bounds)
        extent = [
            global_bounds[:, 0].min(),  # min_x
            global_bounds[:, 2].max(),  # max_x
            global_bounds[:, 1].min(),  # min_y
            global_bounds[:, 3].max()  # max_y
        ]

        # Plot each time step
        plot_info = []
        for i, gdf in enumerate(valid_gdfs):
            row = i // columns
            col = i % columns
            ax = axes[row, col] if rows > 1 else axes[col]

            # Reproject if necessary
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)

            # Create the plot
            gdf.plot(ax=ax, column=time_column, cmap=cmap, legend=False,
                     edgecolor='white', linewidth=0.1)

            # Set consistent extent
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # Styling
            ax.set_title(f"{title_prefix} {i + 1}", fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # Store plot information
            time_range = gdf[time_column].agg(['min', 'max'])
            plot_info.append({
                "subplot_index": i,
                "features_plotted": len(gdf),
                "time_range": [str(time_range['min']), str(time_range['max'])],
                "unique_time_values": len(gdf[time_column].unique()),
                "bounds": gdf.total_bounds.tolist()
            })

        # Hide empty subplots
        for i in range(n_plots, rows * columns):
            row = i // columns
            col = i % columns
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)

        # Add overall title
        fig.suptitle("Temporal Map Series", fontsize=16, fontweight='bold', y=0.95)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                            fraction=0.05, pad=0.08, aspect=50)
        cbar.set_label(time_column, fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)

        # Save if path provided
        saved = False
        save_error = None
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                saved = True
            except Exception as e:
                save_error = str(e)

        return {
            "status": "success",
            "message": "Time series map created successfully",
            "plot_created": True,
            "total_time_steps": len(valid_gdfs),
            "time_column": time_column,
            "subplot_layout": [rows, columns],
            "target_crs": str(target_crs),
            "reprojected_datasets": reprojected_count,
            "global_extent": extent,
            "time_information": time_info,
            "plot_details": plot_info,
            "colormap": cmap,
            "saved_to_file": saved,
            "save_path": save_path if saved else None,
            "save_error": save_error,
            "matplotlib_figure": fig
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Time series plot creation failed: {str(e)}",
            "plot_created": False
        }


def annotate_map_plot(fig, ax, annotations, annotation_type='text', fontsize=10,
                      bbox_props=None, arrow_props=None):
    """
    Add annotations to an existing map plot

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure
        ax (matplotlib.axes.Axes): Matplotlib axes
        annotations (list): List of annotation dictionaries
        annotation_type (str): Type of annotation ('text', 'arrow', 'point')
        fontsize (int): Font size for text annotations
        bbox_props (dict): Text box properties
        arrow_props (dict): Arrow properties

    Returns:
        dict: JSON-like response with annotation information
    """
    try:
        if not hasattr(fig, 'gca') or not hasattr(ax, 'annotate'):
            return {
                "status": "error",
                "message": "Invalid matplotlib figure or axes",
                "annotations_added": 0
            }

        if not annotations or not isinstance(annotations, list):
            return {
                "status": "error",
                "message": "Annotations must be provided as a list",
                "annotations_added": 0
            }

        # Default styling
        default_bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="black")
        default_arrow = dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='red', lw=2)

        bbox_props = bbox_props or default_bbox
        arrow_props = arrow_props or default_arrow

        added_annotations = []
        failed_annotations = []

        for i, annotation in enumerate(annotations):
            try:
                if not isinstance(annotation, dict):
                    failed_annotations.append({
                        "index": i,
                        "error": "Annotation must be a dictionary"
                    })
                    continue

                # Required fields
                if 'x' not in annotation or 'y' not in annotation:
                    failed_annotations.append({
                        "index": i,
                        "error": "Missing required 'x' and 'y' coordinates"
                    })
                    continue

                x, y = annotation['x'], annotation['y']
                text = annotation.get('text', f'Point {i + 1}')

                if annotation_type == 'text':
                    # Simple text annotation
                    ax.text(x, y, text, fontsize=fontsize,
                            bbox=bbox_props, ha='center', va='center')

                elif annotation_type == 'arrow':
                    # Arrow annotation
                    xy_text = annotation.get('text_offset', (20, 20))  # Offset for text
                    ax.annotate(text, xy=(x, y), xytext=xy_text,
                                textcoords='offset points', fontsize=fontsize,
                                bbox=bbox_props, arrowprops=arrow_props,
                                ha='center', va='center')

                elif annotation_type == 'point':
                    # Point marker with optional text
                    marker = annotation.get('marker', 'o')
                    color = annotation.get('color', 'red')
                    size = annotation.get('size', 100)

                    ax.scatter(x, y, marker=marker, c=color, s=size,
                               edgecolors='black', linewidth=1, zorder=1000)

                    if text:
                        ax.text(x, y + (size / 100) * 0.001, text, fontsize=fontsize,
                                bbox=bbox_props, ha='center', va='bottom')

                # Store successful annotation info
                added_annotations.append({
                    "index": i,
                    "type": annotation_type,
                    "x": x,
                    "y": y,
                    "text": text,
                    "properties": {
                        "marker": annotation.get('marker'),
                        "color": annotation.get('color'),
                        "size": annotation.get('size')
                    }
                })

            except Exception as e:
                failed_annotations.append({
                    "index": i,
                    "error": str(e)
                })

        # Refresh the plot
        fig.canvas.draw_idle()

        return {
            "status": "success" if added_annotations else "error",
            "message": f"Added {len(added_annotations)} annotations successfully",
            "annotations_added": len(added_annotations),
            "annotations_failed": len(failed_annotations),
            "annotation_type": annotation_type,
            "successful_annotations": added_annotations,
            "failed_annotations": failed_annotations,
            "styling": {
                "fontsize": fontsize,
                "bbox_properties": bbox_props,
                "arrow_properties": arrow_props if annotation_type == 'arrow' else None
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Annotation failed: {str(e)}",
            "annotations_added": 0
        }



def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")
