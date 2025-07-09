import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Optional


def plot_interactive_choropleth(data: Union[Dict, pd.DataFrame, str],geojson_data: Union[Dict, str] = None,location_col: str = 'location',value_col: str = 'value',color_scale: str = 'Viridis',title: str = 'Interactive Choropleth Map') -> Dict:
    """
    Create an interactive choropleth map with JSON-like response

    Args:
        data: Input data (dict, DataFrame, or JSON string)
        geojson_data: GeoJSON data for boundaries (optional)
        location_col: Column name for location identifiers
        value_col: Column name for values to visualize
        color_scale: Color scale for the map
        title: Map title

    Returns:
        Dict with plot object and metadata
    """
    try:
        # Handle different input types
        if isinstance(data, str):
            df = pd.read_json(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Handle GeoJSON data
        if isinstance(geojson_data, str):
            geojson_data = json.loads(geojson_data)

        # Create choropleth map
        if geojson_data:
            fig = px.choropleth(
                df,
                geojson=geojson_data,
                locations=location_col,
                color=value_col,
                color_continuous_scale=color_scale,
                title=title,
                hover_data=[col for col in df.columns if col not in [location_col, value_col]]
            )
        else:
            # Use built-in geography
            fig = px.choropleth(
                df,
                locations=location_col,
                color=value_col,
                color_continuous_scale=color_scale,
                title=title,
                hover_data=[col for col in df.columns if col not in [location_col, value_col]]
            )

        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            height=600,
            width=900
        )

        # Return JSON-like response
        return {
            "status": "success",
            "plot": fig,
            "metadata": {
                "type": "choropleth",
                "data_points": len(df),
                "columns": list(df.columns),
                "color_scale": color_scale,
                "title": title
            },
            "config": {
                "location_column": location_col,
                "value_column": value_col,
                "has_geojson": geojson_data is not None
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "plot": None,
            "metadata": None
        }


def visualize_geojson_boundaries(geojson_data: Union[Dict, str],properties_to_show: List[str] = None,color_property: str = None,title: str = 'GeoJSON Boundaries Visualization') -> Dict:
    """
    Visualize GeoJSON boundaries with interactive features

    Args:
        geojson_data: GeoJSON data (dict or JSON string)
        properties_to_show: List of properties to display in hover
        color_property: Property to use for coloring (optional)
        title: Map title

    Returns:
        Dict with plot object and metadata
    """
    try:
        # Handle input data
        if isinstance(geojson_data, str):
            geojson = json.loads(geojson_data)
        else:
            geojson = geojson_data

        # Extract properties for analysis
        properties = []
        for feature in geojson.get('features', []):
            props = feature.get('properties', {})
            properties.append(props)

        df = pd.DataFrame(properties)

        # Create the map
        fig = go.Figure()

        if color_property and color_property in df.columns:
            # Color by property
            fig = px.choropleth_mapbox(
                df,
                geojson=geojson,
                locations=df.index,
                color=color_property,
                mapbox_style="open-street-map",
                title=title,
                hover_data=properties_to_show or list(df.columns)[:5]
            )
        else:
            # Simple boundary visualization
            fig.add_trace(go.Choroplethmapbox(
                geojson=geojson,
                locations=[i for i in range(len(geojson['features']))],
                z=[1] * len(geojson['features']),
                colorscale='Blues',
                marker_opacity=0.7,
                marker_line_width=2,
                showscale=False
            ))

            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=0, lon=0), zoom=2),
                title=title
            )

        fig.update_layout(height=600, width=900)

        return {
            "status": "success",
            "plot": fig,
            "metadata": {
                "type": "geojson_boundaries",
                "features_count": len(geojson.get('features', [])),
                "properties": list(df.columns) if not df.empty else [],
                "color_property": color_property,
                "title": title
            },
            "geojson_info": {
                "type": geojson.get('type'),
                "crs": geojson.get('crs'),
                "bbox": geojson.get('bbox')
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "plot": None,
            "metadata": None
        }


def animate_time_series_map(data: Union[Dict, pd.DataFrame, str],lat_col: str = 'lat',lon_col: str = 'lon',time_col: str = 'time',value_col: str = 'value',size_col: str = None,animation_speed: int = 500,title: str = 'Animated Time Series Map') -> Dict:
    """
    Create an animated time series map

    Args:
        data: Input data with time series information
        lat_col: Latitude column name
        lon_col: Longitude column name
        time_col: Time column name
        value_col: Value column for coloring
        size_col: Column for point sizes (optional)
        animation_speed: Animation speed in milliseconds
        title: Map title

    Returns:
        Dict with plot object and metadata
    """
    try:
        # Handle input data
        if isinstance(data, str):
            df = pd.read_json(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Ensure time column is datetime
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col)

        # Create animated scatter mapbox
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=value_col,
            size=size_col if size_col else None,
            animation_frame=time_col,
            hover_data=[col for col in df.columns if col not in [lat_col, lon_col, time_col]],
            color_continuous_scale='Viridis',
            size_max=15,
            zoom=3,
            mapbox_style='open-street-map',
            title=title
        )

        # Update animation settings
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = animation_speed
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = animation_speed // 2

        fig.update_layout(height=600, width=900)

        # Get time range info
        time_range = {
            "start": df[time_col].min().isoformat() if time_col in df.columns else None,
            "end": df[time_col].max().isoformat() if time_col in df.columns else None,
            "frames": len(df[time_col].unique()) if time_col in df.columns else 0
        }

        return {
            "status": "success",
            "plot": fig,
            "metadata": {
                "type": "animated_time_series",
                "data_points": len(df),
                "time_range": time_range,
                "animation_speed": animation_speed,
                "title": title
            },
            "config": {
                "lat_column": lat_col,
                "lon_column": lon_col,
                "time_column": time_col,
                "value_column": value_col,
                "size_column": size_col
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "plot": None,
            "metadata": None
        }


def create_3d_terrain_surface(elevation_data: Union[Dict, pd.DataFrame, str, np.ndarray],x_col: str = 'x',y_col: str = 'y',z_col: str = 'elevation',color_scale: str = 'terrain',title: str = '3D Terrain Surface') -> Dict:
    """
    Create a 3D terrain surface visualization

    Args:
        elevation_data: Elevation data (various formats supported)
        x_col: X coordinate column name
        y_col: Y coordinate column name
        z_col: Elevation/Z coordinate column name
        color_scale: Color scale for elevation
        title: Plot title

    Returns:
        Dict with plot object and metadata
    """
    try:
        # Handle different input types
        if isinstance(elevation_data, str):
            df = pd.read_json(elevation_data)
        elif isinstance(elevation_data, dict):
            df = pd.DataFrame(elevation_data)
        elif isinstance(elevation_data, np.ndarray):
            # Assume it's a 2D elevation grid
            x = np.arange(elevation_data.shape[1])
            y = np.arange(elevation_data.shape[0])
            X, Y = np.meshgrid(x, y)

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=elevation_data,
                colorscale=color_scale,
                showscale=True
            )])

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Elevation",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600,
                width=900
            )

            return {
                "status": "success",
                "plot": fig,
                "metadata": {
                    "type": "3d_terrain_surface",
                    "data_shape": elevation_data.shape,
                    "elevation_range": {
                        "min": float(np.min(elevation_data)),
                        "max": float(np.max(elevation_data))
                    },
                    "color_scale": color_scale,
                    "title": title
                },
                "config": {
                    "input_type": "numpy_array"
                }
            }
        else:
            df = elevation_data.copy()

        # For DataFrame input, create surface from scattered points
        if isinstance(df, pd.DataFrame):
            # Pivot data to create a surface grid
            pivot_df = df.pivot_table(index=y_col, columns=x_col, values=z_col, fill_value=0)

            fig = go.Figure(data=[go.Surface(
                x=pivot_df.columns,
                y=pivot_df.index,
                z=pivot_df.values,
                colorscale=color_scale,
                showscale=True
            )])

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col,
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600,
                width=900
            )

            return {
                "status": "success",
                "plot": fig,
                "metadata": {
                    "type": "3d_terrain_surface",
                    "data_points": len(df),
                    "grid_shape": pivot_df.shape,
                    "elevation_range": {
                        "min": float(df[z_col].min()),
                        "max": float(df[z_col].max())
                    },
                    "color_scale": color_scale,
                    "title": title
                },
                "config": {
                    "x_column": x_col,
                    "y_column": y_col,
                    "z_column": z_col,
                    "input_type": "dataframe"
                }
            }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "plot": None,
            "metadata": None
        }


def add_hover_info_to_points(data: Union[Dict, pd.DataFrame, str],lat_col: str = 'lat',lon_col: str = 'lon',hover_cols: List[str] = None,color_col: str = None,size_col: str = None,custom_hover_template: str = None,title: str = 'Interactive Points with Hover Info') -> Dict:
    """
    Create an interactive map with detailed hover information for points

    Args:
        data: Input data with point locations
        lat_col: Latitude column name
        lon_col: Longitude column name
        hover_cols: Columns to include in hover info
        color_col: Column for point colors
        size_col: Column for point sizes
        custom_hover_template: Custom hover template string
        title: Map title

    Returns:
        Dict with plot object and metadata
    """
    try:
        # Handle input data
        if isinstance(data, str):
            df = pd.read_json(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Set default hover columns if not specified
        if hover_cols is None:
            hover_cols = [col for col in df.columns if col not in [lat_col, lon_col]][:5]

        # Create the scatter mapbox plot
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=color_col,
            size=size_col,
            hover_data=hover_cols,
            color_continuous_scale='Viridis' if color_col else None,
            size_max=15,
            zoom=3,
            mapbox_style='open-street-map',
            title=title
        )

        # Apply custom hover template if provided
        if custom_hover_template:
            fig.update_traces(hovertemplate=custom_hover_template)
        else:
            # Create a comprehensive hover template
            hover_template = f"<b>Location</b><br>"
            hover_template += f"Latitude: %{{lat}}<br>"
            hover_template += f"Longitude: %{{lon}}<br>"

            for col in hover_cols:
                if col in df.columns:
                    hover_template += f"{col}: %{{customdata[{hover_cols.index(col)}]}}<br>"

            hover_template += "<extra></extra>"

            fig.update_traces(
                hovertemplate=hover_template,
                customdata=df[hover_cols].values if hover_cols else None
            )

        # Add click event data
        fig.update_traces(
            marker=dict(
                opacity=0.8,
                line=dict(width=1, color='white')
            )
        )

        fig.update_layout(height=600, width=900)

        # Calculate statistics for hover columns
        hover_stats = {}
        for col in hover_cols:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    hover_stats[col] = {
                        "type": "numeric",
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean())
                    }
                else:
                    hover_stats[col] = {
                        "type": "categorical",
                        "unique_values": len(df[col].unique()),
                        "most_common": df[col].value_counts().index[0] if not df[col].empty else None
                    }

        return {
            "status": "success",
            "plot": fig,
            "metadata": {
                "type": "interactive_points_with_hover",
                "data_points": len(df),
                "hover_columns": hover_cols,
                "hover_stats": hover_stats,
                "has_custom_template": custom_hover_template is not None,
                "title": title
            },
            "config": {
                "lat_column": lat_col,
                "lon_column": lon_col,
                "color_column": color_col,
                "size_column": size_col,
                "hover_columns": hover_cols
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "plot": None,
            "metadata": None
        }






def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")
