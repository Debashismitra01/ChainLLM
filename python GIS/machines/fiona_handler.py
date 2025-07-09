import fiona


def read_vector_file_metadata(file_path):
    """
    Input: file_path (string path to vector file)
    Output: metadata dictionary with file information
    """
    try:
        with fiona.open(file_path) as src:
            metadata = {
                'driver': src.driver,
                'crs': dict(src.crs) if src.crs else None,
                'schema': dict(src.schema),
                'bounds': src.bounds,
                'feature_count': len(src),
                'geometry_type': src.schema['geometry'],
                'properties': list(src.schema['properties'].keys()),
                'property_types': dict(src.schema['properties'])
            }
        return metadata
    except Exception as e:
        return {'error': str(e), 'file_path': file_path}


def write_cleaned_shapefile(geodataframe, output_path, clean_operations=None):
    """
    Input: geodataframe (GeoDataFrame), output_path (string),
           clean_operations (list of cleaning operations)
    Output: success status dict and cleaning report
    """
    if clean_operations is None:
        clean_operations = ['remove_invalid', 'fix_geometry', 'remove_duplicates']

    cleaning_report = {'operations_performed': [], 'records_affected': 0}
    cleaned_gdf = geodataframe.copy()
    original_count = len(cleaned_gdf)

    # Remove invalid geometries
    if 'remove_invalid' in clean_operations:
        invalid_mask = ~cleaned_gdf.geometry.is_valid
        invalid_count = invalid_mask.sum()
        cleaned_gdf = cleaned_gdf[~invalid_mask]
        cleaning_report['operations_performed'].append(f'Removed {invalid_count} invalid geometries')

    # Fix geometries
    if 'fix_geometry' in clean_operations:
        try:
            cleaned_gdf.geometry = cleaned_gdf.geometry.buffer(0)
            cleaning_report['operations_performed'].append('Applied buffer(0) to fix geometries')
        except:
            cleaning_report['operations_performed'].append('Geometry fixing failed')

    # Remove duplicates
    if 'remove_duplicates' in clean_operations:
        duplicate_count = cleaned_gdf.duplicated().sum()
        cleaned_gdf = cleaned_gdf.drop_duplicates()
        cleaning_report['operations_performed'].append(f'Removed {duplicate_count} duplicate records')

    # Remove empty geometries
    empty_mask = cleaned_gdf.geometry.is_empty
    empty_count = empty_mask.sum()
    cleaned_gdf = cleaned_gdf[~empty_mask]
    cleaning_report['operations_performed'].append(f'Removed {empty_count} empty geometries')

    cleaning_report['records_affected'] = original_count - len(cleaned_gdf)

    try:
        cleaned_gdf.to_file(output_path)
        return {
            'success': True,
            'output_path': output_path,
            'original_count': original_count,
            'final_count': len(cleaned_gdf),
            'cleaning_report': cleaning_report
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'cleaning_report': cleaning_report
        }


def inspect_schema_and_crs(file_path):
    """
    Input: file_path (string path to vector file)
    Output: detailed schema and CRS information dict
    """
    try:
        with fiona.open(file_path) as src:
            # Basic file info
            file_info = {
                'file_format': src.driver,
                'feature_count': len(src),
                'bounds': {
                    'minx': src.bounds[0], 'miny': src.bounds[1],
                    'maxx': src.bounds[2], 'maxy': src.bounds[3]
                }
            }

            # Schema details
            schema_info = {
                'geometry_type': src.schema['geometry'],
                'properties': {}
            }

            for prop_name, prop_type in src.schema['properties'].items():
                schema_info['properties'][prop_name] = {
                    'type': prop_type,
                    'python_type': str(type(prop_type).__name__)
                }

            # CRS information
            crs_info = {
                'crs_dict': dict(src.crs) if src.crs else None,
                'proj4_string': src.crs_wkt if hasattr(src, 'crs_wkt') else None,
                'epsg_code': None
            }

            if src.crs and 'init' in src.crs:
                crs_info['epsg_code'] = src.crs['init']

            # Sample feature for data inspection
            sample_features = []
            for i, feature in enumerate(src):
                if i >= 3:  # Limit to first 3 features
                    break
                sample_features.append({
                    'geometry_type': feature['geometry']['type'],
                    'properties': feature['properties'],
                    'has_geometry': feature['geometry'] is not None
                })

            return {
                'file_info': file_info,
                'schema': schema_info,
                'crs': crs_info,
                'sample_features': sample_features,
                'inspection_status': 'success'
            }

    except Exception as e:
        return {
            'inspection_status': 'failed',
            'error': str(e),
            'file_path': file_path
        }


def handle_multi_layer_files(file_path, layer_operations=None):
    """
    Input: file_path (string path to multi-layer file),
           layer_operations (dict with operations for each layer)
    Output: layer information dict and processing results
    """
    if layer_operations is None:
        layer_operations = {'list_layers': True, 'read_all': False, 'sample_each': True}

    try:
        # List available layers
        layers = fiona.listlayers(file_path)
        layer_info = {'available_layers': layers, 'layer_count': len(layers)}

        layer_details = {}

        for layer_name in layers:
            try:
                with fiona.open(file_path, layer=layer_name) as src:
                    layer_details[layer_name] = {
                        'driver': src.driver,
                        'feature_count': len(src),
                        'geometry_type': src.schema['geometry'],
                        'crs': dict(src.crs) if src.crs else None,
                        'bounds': src.bounds,
                        'properties': list(src.schema['properties'].keys()),
                        'property_count': len(src.schema['properties'])
                    }

                    # Sample features if requested
                    if layer_operations.get('sample_each', False):
                        sample_count = min(2, len(src))
                        samples = []
                        for i, feature in enumerate(src):
                            if i >= sample_count:
                                break
                            samples.append({
                                'properties': feature['properties'],
                                'geometry_type': feature['geometry']['type'] if feature['geometry'] else None
                            })
                        layer_details[layer_name]['sample_features'] = samples

            except Exception as layer_error:
                layer_details[layer_name] = {'error': str(layer_error)}

        return {
            'file_path': file_path,
            'layer_summary': layer_info,
            'layer_details': layer_details,
            'processing_status': 'success'
        }

    except Exception as e:
        return {
            'file_path': file_path,
            'processing_status': 'failed',
            'error': str(e)
        }


def stream_large_vector_file(file_path, chunk_size=1000, processing_function=None):
    """
    Input: file_path (string), chunk_size (int), processing_function (callable)
    Output: streaming processing results and file statistics
    """
    if processing_function is None:
        # Default processing: count features by geometry type
        def processing_function(features_chunk):
            geometry_counts = {}
            for feature in features_chunk:
                geom_type = feature['geometry']['type'] if feature['geometry'] else 'None'
                geometry_counts[geom_type] = geometry_counts.get(geom_type, 0) + 1
            return geometry_counts

    try:
        with fiona.open(file_path) as src:
            total_features = len(src)
            processed_count = 0
            chunk_results = []

            current_chunk = []

            for feature in src:
                current_chunk.append(feature)

                # Process chunk when it reaches the specified size
                if len(current_chunk) >= chunk_size:
                    chunk_result = processing_function(current_chunk)
                    chunk_results.append({
                        'chunk_index': len(chunk_results),
                        'chunk_size': len(current_chunk),
                        'result': chunk_result
                    })
                    processed_count += len(current_chunk)
                    current_chunk = []

            # Process remaining features
            if current_chunk:
                chunk_result = processing_function(current_chunk)
                chunk_results.append({
                    'chunk_index': len(chunk_results),
                    'chunk_size': len(current_chunk),
                    'result': chunk_result
                })
                processed_count += len(current_chunk)

            # Aggregate results
            aggregated_results = {}
            for chunk in chunk_results:
                if isinstance(chunk['result'], dict):
                    for key, value in chunk['result'].items():
                        aggregated_results[key] = aggregated_results.get(key, 0) + value

            return {
                'file_path': file_path,
                'total_features': total_features,
                'processed_features': processed_count,
                'chunk_count': len(chunk_results),
                'chunk_size_used': chunk_size,
                'chunk_results': chunk_results,
                'aggregated_results': aggregated_results,
                'streaming_status': 'completed'
            }

    except Exception as e:
        return {
            'file_path': file_path,
            'streaming_status': 'failed',
            'error': str(e)
        }

# Template for all handler files
def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")
