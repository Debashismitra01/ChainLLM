import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json
from datetime import datetime, timedelta
import fiona
from fiona.crs import from_epsg
import pyproj
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
import geopandas as gpd


def plot_weather_climate_data(temperature_data, precipitation_data, dates, location_coords=None):
    """
    Input: temperature_data (array), precipitation_data (array), dates (array), location_coords (optional tuple)
    Output: matplotlib figure with weather/climate visualizations and summary statistics dict
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Weather and Climate Data Analysis', fontsize=16, fontweight='bold')

    # Temperature trend
    ax1.plot(dates, temperature_data, color='red', alpha=0.7, linewidth=1)
    ax1.set_title('Daily Temperature Trends')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, alpha=0.3)

    # Monthly precipitation aggregation
    df = pd.DataFrame({'date': dates, 'precip': precipitation_data})
    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly_precip = df.groupby('month')['precip'].sum()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2.bar(months[:len(monthly_precip)], monthly_precip.values, color='blue', alpha=0.7)
    ax2.set_title('Monthly Precipitation')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.tick_params(axis='x', rotation=45)

    # Temperature distribution heatmap
    temp_matrix = temperature_data.reshape(-1, min(len(temperature_data) // 10, 10))[:10]
    im = ax3.imshow(temp_matrix, cmap='RdYlBu_r', aspect='auto')
    ax3.set_title('Temperature Distribution Pattern')
    plt.colorbar(im, ax=ax3, label='Temperature (°C)')

    # Climate statistics polar plot
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    seasonal_temps = [temperature_data[i:i + len(temperature_data) // 4].mean()
                      for i in range(0, len(temperature_data), len(temperature_data) // 4)][:4]
    theta = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ax4.bar(theta, seasonal_temps, width=2 * np.pi / 4, alpha=0.7, color='green')
    ax4.set_title('Seasonal Temperature Averages', pad=20)
    ax4.set_theta_zero_location('N')

    plt.tight_layout()

    return {
        'figure': fig,
        'stats': {
            'temp_mean': np.mean(temperature_data),
            'temp_max': np.max(temperature_data),
            'temp_min': np.min(temperature_data),
            'total_precip': np.sum(precipitation_data),
            'monthly_precip': monthly_precip.to_dict()
        }
    }


def overlay_basemap_layers(extent_bounds, layer_data_dict, transparency_levels=None):
    """
    Input: extent_bounds (dict with lon_min, lon_max, lat_min, lat_max),
           layer_data_dict (dict with layer names and 2D arrays),
           transparency_levels (dict with alpha values)
    Output: layered map figure and layer metadata dict
    """
    if transparency_levels is None:
        transparency_levels = {'base': 1.0, 'overlay1': 0.7, 'overlay2': 0.5}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Basemap Layer Overlays', fontsize=16, fontweight='bold')

    lon_min, lon_max = extent_bounds['lon_min'], extent_bounds['lon_max']
    lat_min, lat_max = extent_bounds['lat_min'], extent_bounds['lat_max']
    extent = [lon_min, lon_max, lat_min, lat_max]

    layer_names = list(layer_data_dict.keys())

    # Individual layers
    for i, (layer_name, layer_data) in enumerate(layer_data_dict.items()):
        if i >= 3:  # Max 3 individual layers
            break
        ax = axes.flat[i]
        if layer_name == 'elevation':
            plot = ax.imshow(layer_data, extent=extent, cmap='terrain', alpha=transparency_levels.get(layer_name, 0.8))
            plt.colorbar(plot, ax=ax, label='Elevation (m)')
        elif layer_name == 'landuse':
            plot = ax.imshow(layer_data, extent=extent, cmap='Set3', alpha=transparency_levels.get(layer_name, 0.8))
        else:
            plot = ax.imshow(layer_data, extent=extent, alpha=transparency_levels.get(layer_name, 0.8))

        ax.set_title(f'{layer_name.title()} Layer')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # Combined overlay
    ax_combined = axes.flat[-1]
    for i, (layer_name, layer_data) in enumerate(layer_data_dict.items()):
        alpha = transparency_levels.get(layer_name, 0.5 - i * 0.1)
        if layer_name == 'elevation':
            ax_combined.imshow(layer_data, extent=extent, cmap='terrain', alpha=alpha)
        else:
            ax_combined.imshow(layer_data, extent=extent, alpha=alpha)

    ax_combined.set_title('Combined Overlay')
    ax_combined.set_xlabel('Longitude')
    ax_combined.set_ylabel('Latitude')

    plt.tight_layout()

    return {
        'figure': fig,
        'layer_info': {
            'extent': extent,
            'layer_count': len(layer_data_dict),
            'layer_names': layer_names,
            'transparency_used': transparency_levels
        }
    }


def visualize_meteorological_flow(wind_u_component, wind_v_component, pressure_field, grid_coords):
    """
    Input: wind_u_component (2D array), wind_v_component (2D array),
           pressure_field (2D array), grid_coords (dict with X, Y meshgrid arrays)
    Output: meteorological flow visualization figure and flow statistics dict
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Meteorological Flow Visualization', fontsize=16, fontweight='bold')

    X, Y = grid_coords['X'], grid_coords['Y']
    U, V = wind_u_component, wind_v_component
    speed = np.sqrt(U ** 2 + V ** 2)

    # Wind vector field
    ax1.quiver(X, Y, U, V, speed, cmap='viridis', alpha=0.8)
    ax1.set_title('Wind Vector Field')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Distance (km)')
    ax1.grid(True, alpha=0.3)

    # Pressure contours
    contour = ax2.contour(X, Y, pressure_field, levels=15, colors='blue', linewidths=1)
    ax2.clabel(contour, inline=True, fontsize=8)
    contour_filled = ax2.contourf(X, Y, pressure_field, levels=15, cmap='RdBu_r', alpha=0.6)
    plt.colorbar(contour_filled, ax=ax2, label='Pressure (hPa)')
    ax2.set_title('Atmospheric Pressure System')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Distance (km)')

    # Streamlines
    ax3.streamplot(X, Y, U, V, color=speed, cmap='plasma', density=2, linewidth=1)
    ax3.set_title('Flow Streamlines')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Distance (km)')

    # Vorticity calculation
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    vorticity = np.gradient(V, dx, axis=1) - np.gradient(U, dy, axis=0)

    vort_plot = ax4.imshow(vorticity, extent=[X.min(), X.max(), Y.min(), Y.max()],
                           cmap='RdBu', origin='lower')
    plt.colorbar(vort_plot, ax=ax4, label='Vorticity (s⁻¹)')
    ax4.set_title('Atmospheric Vorticity')
    ax4.set_xlabel('Distance (km)')
    ax4.set_ylabel('Distance (km)')

    plt.tight_layout()

    return {
        'figure': fig,
        'flow_stats': {
            'max_wind_speed': np.max(speed),
            'avg_wind_speed': np.mean(speed),
            'pressure_range': [np.min(pressure_field), np.max(pressure_field)],
            'max_vorticity': np.max(np.abs(vorticity)),
            'dominant_flow_direction': np.arctan2(np.mean(V), np.mean(U)) * 180 / np.pi
        }
    }


def create_multi_panel_maps(main_dataset, regional_datasets, time_series_data, panel_config):
    """
    Input: main_dataset (2D array), regional_datasets (dict of 2D arrays),
           time_series_data (dict with dates and values), panel_config (dict with layout settings)
    Output: multi-panel figure and panel summary dict
    """
    fig = plt.figure(figsize=(panel_config.get('figsize', (16, 12))))
    fig.suptitle(panel_config.get('title', 'Multi-Panel Geographic Analysis'),
                 fontsize=18, fontweight='bold')

    # Grid layout based on config
    rows = panel_config.get('rows', 3)
    cols = panel_config.get('cols', 4)
    gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.3)

    # Main panel
    main_extent = panel_config.get('main_extent', [-180, 180, -90, 90])
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    main_plot = ax_main.imshow(main_dataset, cmap=panel_config.get('main_cmap', 'viridis'),
                               extent=main_extent)
    ax_main.set_title(panel_config.get('main_title', 'Main Dataset'), fontsize=14)
    ax_main.set_xlabel('Longitude')
    ax_main.set_ylabel('Latitude')
    plt.colorbar(main_plot, ax=ax_main, label=panel_config.get('main_label', 'Values'))

    # Regional panels
    panel_positions = [(0, 2), (0, 3), (1, 2), (1, 3)]
    for i, (region_name, region_data) in enumerate(regional_datasets.items()):
        if i >= len(panel_positions):
            break
        row, col = panel_positions[i]
        ax = fig.add_subplot(gs[row, col])
        extent = panel_config.get('regional_extents', {}).get(region_name, [-180, 180, -90, 90])
        ax.imshow(region_data, cmap='viridis', extent=extent)
        ax.set_title(region_name, fontsize=10)
        ax.tick_params(labelsize=8)

    # Time series panel
    ax_timeseries = fig.add_subplot(gs[2, 0:2])
    dates = time_series_data['dates']
    values = time_series_data['values']
    ax_timeseries.plot(dates, values, color='red', linewidth=2)
    ax_timeseries.set_title(panel_config.get('timeseries_title', 'Time Series'), fontsize=12)
    ax_timeseries.set_ylabel(panel_config.get('timeseries_ylabel', 'Values'))
    ax_timeseries.grid(True, alpha=0.3)

    # Statistics panel
    ax_stats = fig.add_subplot(gs[2, 2:])
    stats_data = {
        'Max': np.max(main_dataset),
        'Min': np.min(main_dataset),
        'Mean': np.mean(main_dataset),
        'Std': np.std(main_dataset)
    }
    ax_stats.bar(stats_data.keys(), stats_data.values(),
                 color=['red', 'blue', 'green', 'orange'])
    ax_stats.set_title('Dataset Statistics', fontsize=12)
    ax_stats.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    return {
        'figure': fig,
        'panel_summary': {
            'main_stats': stats_data,
            'regional_count': len(regional_datasets),
            'timeseries_length': len(values),
            'layout': f'{rows}x{cols}'
        }
    }


def customize_map_projection(geodata, source_crs, target_crs, extent_bounds=None):
    """
    Input: geodata (dict with coordinates/geometries), source_crs (string),
           target_crs (string), extent_bounds (optional dict)
    Output: transformed geodata and projection metadata dict
    """
    # Setup projection transformation
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    transformed_data = {}

    # Transform point data
    if 'points' in geodata:
        points = geodata['points']
        if isinstance(points, dict) and 'x' in points and 'y' in points:
            x_coords, y_coords = points['x'], points['y']
            x_trans, y_trans = transformer.transform(x_coords, y_coords)
            transformed_data['points'] = {'x': x_trans, 'y': y_trans}

    # Transform polygon data
    if 'polygons' in geodata:
        transformed_polygons = []
        for poly_coords in geodata['polygons']:
            x_coords = [coord[0] for coord in poly_coords]
            y_coords = [coord[1] for coord in poly_coords]
            x_trans, y_trans = transformer.transform(x_coords, y_coords)
            transformed_polygons.append(list(zip(x_trans, y_trans)))
        transformed_data['polygons'] = transformed_polygons

    # Transform extent bounds if provided
    if extent_bounds:
        x_bounds = [extent_bounds['xmin'], extent_bounds['xmax']]
        y_bounds = [extent_bounds['ymin'], extent_bounds['ymax']]
        x_trans, y_trans = transformer.transform(x_bounds, y_bounds)
        transformed_bounds = {
            'xmin': min(x_trans), 'xmax': max(x_trans),
            'ymin': min(y_trans), 'ymax': max(y_trans)
        }
        transformed_data['extent'] = transformed_bounds

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original projection
    if 'points' in geodata:
        ax1.scatter(geodata['points']['x'], geodata['points']['y'],
                    c='red', alpha=0.6, s=20)
    if 'polygons' in geodata:
        for poly in geodata['polygons']:
            x_coords = [coord[0] for coord in poly] + [poly[0][0]]
            y_coords = [coord[1] for coord in poly] + [poly[0][1]]
            ax1.plot(x_coords, y_coords, 'b-', alpha=0.7)

    ax1.set_title(f'Original Projection ({source_crs})')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, alpha=0.3)

    # Transformed projection
    if 'points' in transformed_data:
        ax2.scatter(transformed_data['points']['x'], transformed_data['points']['y'],
                    c='red', alpha=0.6, s=20)
    if 'polygons' in transformed_data:
        for poly in transformed_data['polygons']:
            x_coords = [coord[0] for coord in poly] + [poly[0][0]]
            y_coords = [coord[1] for coord in poly] + [poly[0][1]]
            ax2.plot(x_coords, y_coords, 'b-', alpha=0.7)

    ax2.set_title(f'Transformed Projection ({target_crs})')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    return {
        'transformed_data': transformed_data,
        'figure': fig,
        'projection_info': {
            'source_crs': source_crs,
            'target_crs': target_crs,
            'transformation_type': 'coordinate_system_conversion',
            'data_types_transformed': list(transformed_data.keys())
        }
    }


# Template for all handler files

def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")