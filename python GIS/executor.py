from datetime import datetime
import importlib
from typing import Optional


# You can extend this as more actions are defined
ACTION_MAP = {
    'rasterio_actions' : {"read_satellite_bands", 'clip_raster_to_boundary', 'write_processed_raster','extract_pixel_values', 'reproject_resample_raster'},
    'cartopy_actions' : {'plot_weather_climate_data', 'overlay_basemap_layers', 'visualize_meteorological_flow','visualize_meteorological_flow', 'create_multi_panel_maps','customize_map_projection'},
    'fiona_actions' : {'read_vector_file_metadata', 'write_cleaned_shapefile', 'inspect_schema_and_crs','handle_multiple_layer_files', 'stream_large_vector'},
    'folium_actions' : {'create_interactive_map', 'add_markers_with_popups', 'build_map_dashboard','build_map_dashboard', 'render_choropleth_map',
    'export_html_leaflet_map'},
    'GDAL_actions' : {'reproject_large_raster', 'crop_merge_satellite_images', 'convert_geotiff_to_png','extract_raster_metadata', 'align_rasters_with_warp'},
    'geopandas_actions' : {'load_clean_vector_data', 'filter_features_by_attribute', 'spatial_join_analysis','create_buffer_zones', 'convert_crs_geodata'},
    'matplotlib_actions' : {'plot_static_thematic_map', 'overlay_layers_on_map', 'export_map_images','plot_map_time_series', 'annotate_map_plot'},
    'plotly_actions' : {'plot_interactive_choropleth', 'visualize_geojson_boundaries', 'animate_time_series_map','create_3d_terrain_surface', 'add_hover_info_to_points'},
    'pyproj_actions' : {'classify_ndvi_raster', 'run_dem_filter_window', 'aggregate_pixel_statistics','mask_raster_array', 'compute_multi_band_index', 'plot_interactive_choropleth','visualize_geojson_boundaries', 'animate_time_series_map', 'create_3d_terrain_surface','add_hover_info_to_points'},
    'shapely_actions' : {'create_geometries', 'check_geometry_intersections', 'calculate_geometry_distance','generate_convex_hull', 'fix_invalid_geometries', '_calculate_bounds','_flatten_coordinates', '_euclidean_distance', '_point_on_line', '_point_on_segment','_line_line_intersection', '_segment_intersection', '_bounds_intersect','_point_to_line_distance', '_point_to_segment_distance', '_line_to_line_distance','_calculate_centroid', '_polygon_area', '_polygon_perimeter'},
    'numpy_actions' : {"classify_ndvi_raster", 'run_dem_filter_window', 'aggregate_pixel_statistics','mask_raster_array', 'compute_multi_band_index'},
    }

def resolve_machine(action: str) -> Optional[str]:
    for machine, actions in ACTION_MAP.items():
        if action in actions:
            return machine
    return None


def execute_workflow(workflow: list[dict]) -> tuple[dict, list[dict], bool, datetime]:
    env = {}
    results = []
    success = True

    for step in workflow:
        action = step["action"]
        output_name = step.get("output")

        machine = resolve_machine(action)
        if not machine:
            results.append({
                "action": action,
                "error": f"No handler found for action '{action}'",
                "status": "error"
            })
            success = False
            continue

        try:
            module = importlib.import_module(f"machines.{machine}_handler")
            result = module.run(action, step, env)

            if output_name:
                env[output_name] = result

            results.append({
                "action": action,
                "output_name": output_name,
                "status": "ok"
            })
        except Exception as e:
            results.append({
                "action": action,
                "error": str(e),
                "status": "error"
            })
            success = False

    return env, results, success, datetime.now()

