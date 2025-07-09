import folium
from folium import plugins
import json
import pandas as pd
import os

# =============================================================================
# FOLIUM FUNCTIONS
# =============================================================================

def create_interactive_map(center_coords, zoom_level=10, map_style='OpenStreetMap',
                           width='100%', height='600px', additional_layers=None):
    """
    Input: center_coords (tuple: lat, lon), zoom_level (int), map_style (string),
           width (string), height (string), additional_layers (list of layer configs)
    Output: folium Map object and map configuration dict
    """
    # Create base map
    m = folium.Map(
        location=center_coords,
        zoom_start=zoom_level,
        tiles=map_style,
        width=width,
        height=height
    )

    # Add additional tile layers if specified
    if additional_layers:
        for layer_config in additional_layers:
            if layer_config['type'] == 'tile_layer':
                folium.TileLayer(
                    tiles=layer_config['tiles'],
                    attr=layer_config.get('attribution', ''),
                    name=layer_config.get('name', 'Custom Layer'),
                    overlay=layer_config.get('overlay', False)
                ).add_to(m)

            elif layer_config['type'] == 'wms':
                folium.raster_layers.WmsTileLayer(
                    url=layer_config['url'],
                    layers=layer_config['layers'],
                    name=layer_config.get('name', 'WMS Layer'),
                    format=layer_config.get('format', 'image/png'),
                    transparent=layer_config.get('transparent', True)
                ).add_to(m)

    # Add layer control if multiple layers exist
    if additional_layers:
        folium.LayerControl().add_to(m)

    # Add scale bar
    plugins.MeasureControl().add_to(m)

    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)

    # Configuration summary
    map_config = {
        'center': center_coords,
        'zoom': zoom_level,
        'style': map_style,
        'dimensions': {'width': width, 'height': height},
        'additional_layers_count': len(additional_layers) if additional_layers else 0,
        'features_added': ['scale_bar', 'minimap', 'measure_control']
    }

    return {
        'map_object': m,
        'config': map_config,
        'bounds': m.get_bounds() if hasattr(m, 'get_bounds') else None
    }


def add_markers_with_popups(folium_map, marker_data, marker_style=None, cluster_markers=False):
    """
    Input: folium_map (folium Map object), marker_data (list of dicts with lat, lon, popup, tooltip),
           marker_style (dict with style options), cluster_markers (boolean)
    Output: updated folium map and marker statistics dict
    """
    if marker_style is None:
        marker_style = {
            'radius': 8,
            'color': 'blue',
            'fill': True,
            'fillColor': 'lightblue',
            'fillOpacity': 0.7,
            'popup_max_width': 300
        }

    marker_stats = {
        'total_markers': len(marker_data),
        'clustered': cluster_markers,
        'marker_types': {}
    }

    # Create marker cluster if requested
    if cluster_markers:
        marker_cluster = plugins.MarkerCluster().add_to(folium_map)
        container = marker_cluster
    else:
        container = folium_map

    # Add markers
    for i, marker_info in enumerate(marker_data):
        lat, lon = marker_info['lat'], marker_info['lon']
        popup_content = marker_info.get('popup', f'Marker {i + 1}')
        tooltip_content = marker_info.get('tooltip', f'Point {i + 1}')
        marker_type = marker_info.get('type', 'default')

        # Track marker types
        marker_stats['marker_types'][marker_type] = marker_stats['marker_types'].get(marker_type, 0) + 1

        # Create popup with custom styling
        popup = folium.Popup(
            html=popup_content,
            max_width=marker_style['popup_max_width']
        )

        # Different marker styles based on type
        if marker_type == 'circle':
            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_style['radius'],
                popup=popup,
                tooltip=tooltip_content,
                color=marker_info.get('color', marker_style['color']),
                fill=marker_style['fill'],
                fillColor=marker_info.get('fillColor', marker_style['fillColor']),
                fillOpacity=marker_style['fillOpacity']
            ).add_to(container)

        elif marker_type == 'icon':
            icon_config = marker_info.get('icon', {})
            icon = folium.Icon(
                color=icon_config.get('color', 'blue'),
                icon=icon_config.get('icon', 'info-sign'),
                prefix=icon_config.get('prefix', 'glyphicon')
            )
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=tooltip_content,
                icon=icon
            ).add_to(container)

        else:  # default marker
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                tooltip=tooltip_content
            ).add_to(container)

    return {
        'map_object': folium_map,
        'marker_stats': marker_stats,
        'clustering_enabled': cluster_markers
    }


def build_map_dashboard(dashboard_config, data_layers, control_panels=None):
    """
    Input: dashboard_config (dict with layout settings), data_layers (dict of layer data),
           control_panels (list of control configurations)
    Output: comprehensive folium dashboard map and dashboard metadata
    """
    # Create base map
    center = dashboard_config.get('center', [0, 0])
    m = folium.Map(
        location=center,
        zoom_start=dashboard_config.get('zoom', 6),
        tiles=dashboard_config.get('base_tiles', 'OpenStreetMap')
    )

    dashboard_metadata = {
        'layers_added': [],
        'controls_added': [],
        'plugins_used': []
    }

    # Add data layers
    for layer_name, layer_data in data_layers.items():
        layer_type = layer_data.get('type', 'markers')

        if layer_type == 'markers':
            # Add marker layer
            marker_group = folium.FeatureGroup(name=layer_name)
            for point in layer_data['data']:
                folium.Marker(
                    location=[point['lat'], point['lon']],
                    popup=point.get('popup', ''),
                    tooltip=point.get('tooltip', '')
                ).add_to(marker_group)
            marker_group.add_to(m)
            dashboard_metadata['layers_added'].append(f'{layer_name} (markers)')

        elif layer_type == 'heatmap':
            # Add heatmap layer
            heat_data = [[point['lat'], point['lon'], point.get('weight', 1)]
                         for point in layer_data['data']]
            plugins.HeatMap(
                heat_data,
                name=layer_name,
                radius=layer_data.get('radius', 15),
                blur=layer_data.get('blur', 15),
                max_zoom=layer_data.get('max_zoom', 18)
            ).add_to(m)
            dashboard_metadata['layers_added'].append(f'{layer_name} (heatmap)')
            dashboard_metadata['plugins_used'].append('HeatMap')

        elif layer_type == 'choropleth':
            # Add choropleth layer
            folium.Choropleth(
                geo_data=layer_data['geo_data'],
                data=layer_data['data'],
                columns=layer_data['columns'],
                key_on=layer_data['key_on'],
                fill_color=layer_data.get('fill_color', 'YlOrRd'),
                fill_opacity=layer_data.get('fill_opacity', 0.7),
                line_opacity=layer_data.get('line_opacity', 0.2),
                legend_name=layer_data.get('legend_name', layer_name)
            ).add_to(m)
            dashboard_metadata['layers_added'].append(f'{layer_name} (choropleth)')

    # Add control panels
    if control_panels:
        for control_config in control_panels:
            control_type = control_config.get('type', 'layer_control')

            if control_type == 'layer_control':
                folium.LayerControl(
                    position=control_config.get('position', 'topright'),
                    collapsed=control_config.get('collapsed', False)
                ).add_to(m)
                dashboard_metadata['controls_added'].append('LayerControl')

            elif control_type == 'fullscreen':
                plugins.Fullscreen(
                    position=control_config.get('position', 'topleft'),
                    title=control_config.get('title', 'Full Screen'),
                    title_cancel=control_config.get('title_cancel', 'Exit Full Screen')
                ).add_to(m)
                dashboard_metadata['controls_added'].append('Fullscreen')
                dashboard_metadata['plugins_used'].append('Fullscreen')

            elif control_type == 'draw':
                draw = plugins.Draw(
                    export=control_config.get('export', True),
                    position=control_config.get('position', 'topleft')
                )
                draw.add_to(m)
                dashboard_metadata['controls_added'].append('Draw')
                dashboard_metadata['plugins_used'].append('Draw')

    # Add search functionality if requested
    if dashboard_config.get('add_search', False):
        plugins.Search(
            layer=m,
            search_label='popup',
            placeholder='Search locations...',
            collapsed=False
        ).add_to(m)
        dashboard_metadata['plugins_used'].append('Search')

    return {
        'dashboard_map': m,
        'metadata': dashboard_metadata,
        'layer_count': len(data_layers),
        'control_count': len(control_panels) if control_panels else 0
    }


def render_choropleth_map(geo_data, attribute_data, join_columns, map_config=None):
    """
    Input: geo_data (GeoJSON dict or file path), attribute_data (DataFrame or dict),
           join_columns (list: [geo_key, data_key]), map_config (dict with styling options)
    Output: folium map with choropleth visualization and statistics dict
    """
    if map_config is None:
        map_config = {
            'center': [0, 0],
            'zoom': 6,
            'fill_color': 'YlOrRd',
            'fill_opacity': 0.7,
            'line_opacity': 0.2,
            'line_color': 'white',
            'line_weight': 2
        }

    # Create base map
    m = folium.Map(
        location=map_config['center'],
        zoom_start=map_config['zoom'],
        tiles=map_config.get('tiles', 'OpenStreetMap')
    )

    # Convert attribute_data to DataFrame if needed
    if isinstance(attribute_data, dict):
        df = pd.DataFrame(attribute_data)
    else:
        df = attribute_data

    # Get the value column (assumes last column if not specified)
    value_column = map_config.get('value_column', df.columns[-1])

    # Calculate statistics
    stats = {
        'value_column': value_column,
        'min_value': df[value_column].min(),
        'max_value': df[value_column].max(),
        'mean_value': df[value_column].mean(),
        'std_value': df[value_column].std(),
        'total_features': len(df)
    }

    # Create choropleth
    choropleth = folium.Choropleth(
        geo_data=geo_data,
        data=df,
        columns=[join_columns[1], value_column],
        key_on=f'feature.properties.{join_columns[0]}',
        fill_color=map_config['fill_color'],
        fill_opacity=map_config['fill_opacity'],
        line_opacity=map_config['line_opacity'],
        line_color=map_config['line_color'],
        line_weight=map_config['line_weight'],
        legend_name=map_config.get('legend_name', value_column),
        bins=map_config.get('bins', 5),
        reset=map_config.get('reset', False)
    )

    choropleth.add_to(m)

    # Add tooltips if requested
    if map_config.get('add_tooltips', True):
        # Create a feature group for tooltips
        tooltip_layer = folium.FeatureGroup(name="Tooltips")

        # Add GeoJson layer with tooltips
        if isinstance(geo_data, str):
            with open(geo_data, 'r') as f:
                geo_json_data = json.load(f)
        else:
            geo_json_data = geo_data

        folium.GeoJson(
            geo_json_data,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'transparent',
                'weight': 0
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=[join_columns[0]],
                aliases=[map_config.get('tooltip_alias', join_columns[0])],
                localize=True,
                sticky=True,
                labels=True,
                style="""
                    background-color: white;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """
            )
        ).add_to(tooltip_layer)

        tooltip_layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return {
        'choropleth_map': m,
        'statistics': stats,
        'classification_breaks': choropleth.color_scale if hasattr(choropleth, 'color_scale') else None,
        'config_used': map_config
    }


def export_html_leaflet_map(folium_map, output_path, export_options=None):
    """
    Input: folium_map (folium Map object), output_path (string),
           export_options (dict with export configurations)
    Output: exported HTML file and export metadata dict
    """
    if export_options is None:
        export_options = {
            'embed_css': True,
            'embed_js': True,
            'minify': False,
            'add_metadata': True
        }

    # Get map HTML
    map_html = folium_map._repr_html_()

    # Add custom CSS if specified
    custom_css = export_options.get('custom_css', '')
    if custom_css:
        css_tag = f'<style>\n{custom_css}\n</style>'
        map_html = map_html.replace('</head>', f'{css_tag}\n</head>')

    # Add custom JavaScript if specified
    custom_js = export_options.get('custom_js', '')
    if custom_js:
        js_tag = f'<script>\n{custom_js}\n</script>'
        map_html = map_html.replace('</body>', f'{js_tag}\n</body>')

    # Add metadata comment if requested
    if export_options.get('add_metadata', True):
        from datetime import datetime
        metadata_comment = f'''
        <!-- 
        Map exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Export options: {export_options}
        -->
        '''
        map_html = metadata_comment + map_html

    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(map_html)

        # Get file size
        file_size = os.path.getsize(output_path)

        export_metadata = {
            'export_success': True,
            'output_path': output_path,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'export_options_used': export_options,
            'export_timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        export_metadata = {
            'export_success': False,
            'error': str(e),
            'output_path': output_path,
            'export_options_used': export_options
        }

    return export_metadata


def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")