import numpy as np
import json


# NumPy Functions
def classify_ndvi_raster(ndvi_array, thresholds=None):
    """
    Classify NDVI raster into vegetation categories

    Args:
        ndvi_array: numpy array with NDVI values (-1 to 1)
        thresholds: dict with classification thresholds

    Returns:
        dict with classified array and metadata
    """
    if thresholds is None:
        thresholds = {
            'water': -1.0,
            'bare_soil': 0.1,
            'sparse_vegetation': 0.3,
            'moderate_vegetation': 0.6,
            'dense_vegetation': 1.0
        }

    # Create classification array
    classified = np.zeros_like(ndvi_array, dtype=np.int8)

    # Apply classification
    classified[ndvi_array <= thresholds['water']] = 0  # Water
    classified[(ndvi_array > thresholds['water']) & (ndvi_array <= thresholds['bare_soil'])] = 1  # Bare soil
    classified[(ndvi_array > thresholds['bare_soil']) & (
                ndvi_array <= thresholds['sparse_vegetation'])] = 2  # Sparse vegetation
    classified[(ndvi_array > thresholds['sparse_vegetation']) & (
                ndvi_array <= thresholds['moderate_vegetation'])] = 3  # Moderate vegetation
    classified[ndvi_array > thresholds['moderate_vegetation']] = 4  # Dense vegetation

    return {
        'classified_array': classified.tolist(),
        'classes': {
            0: 'Water',
            1: 'Bare Soil',
            2: 'Sparse Vegetation',
            3: 'Moderate Vegetation',
            4: 'Dense Vegetation'
        },
        'thresholds': thresholds,
        'statistics': {
            'total_pixels': int(np.size(classified)),
            'class_counts': {str(i): int(np.sum(classified == i)) for i in range(5)}
        }
    }


def run_dem_filter_window(dem_array, filter_type='gaussian', window_size=3, sigma=1.0):
    """
    Apply filtering to DEM data

    Args:
        dem_array: numpy array with elevation data
        filter_type: type of filter ('gaussian', 'mean', 'median')
        window_size: size of the filter window
        sigma: standard deviation for gaussian filter

    Returns:
        dict with filtered array and metadata
    """
    from scipy import ndimage

    if filter_type == 'gaussian':
        filtered = ndimage.gaussian_filter(dem_array, sigma=sigma)
    elif filter_type == 'mean':
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        filtered = ndimage.convolve(dem_array, kernel)
    elif filter_type == 'median':
        filtered = ndimage.median_filter(dem_array, size=window_size)
    else:
        filtered = dem_array  # No filtering

    # Calculate statistics
    original_stats = {
        'mean': float(np.mean(dem_array)),
        'std': float(np.std(dem_array)),
        'min': float(np.min(dem_array)),
        'max': float(np.max(dem_array))
    }

    filtered_stats = {
        'mean': float(np.mean(filtered)),
        'std': float(np.std(filtered)),
        'min': float(np.min(filtered)),
        'max': float(np.max(filtered))
    }

    return {
        'filtered_array': filtered.tolist(),
        'filter_parameters': {
            'type': filter_type,
            'window_size': window_size,
            'sigma': sigma
        },
        'original_statistics': original_stats,
        'filtered_statistics': filtered_stats,
        'smoothing_effect': float(original_stats['std'] - filtered_stats['std'])
    }


def aggregate_pixel_statistics(raster_array, aggregation_method='mean', block_size=2):
    """
    Aggregate pixel statistics using specified method

    Args:
        raster_array: numpy array to aggregate
        aggregation_method: method for aggregation ('mean', 'sum', 'max', 'min', 'std')
        block_size: size of aggregation blocks

    Returns:
        dict with aggregated array and metadata
    """
    h, w = raster_array.shape
    new_h, new_w = h // block_size, w // block_size

    # Trim array to fit block size
    trimmed = raster_array[:new_h * block_size, :new_w * block_size]

    # Reshape for aggregation
    reshaped = trimmed.reshape(new_h, block_size, new_w, block_size)

    # Apply aggregation method
    if aggregation_method == 'mean':
        aggregated = np.mean(reshaped, axis=(1, 3))
    elif aggregation_method == 'sum':
        aggregated = np.sum(reshaped, axis=(1, 3))
    elif aggregation_method == 'max':
        aggregated = np.max(reshaped, axis=(1, 3))
    elif aggregation_method == 'min':
        aggregated = np.min(reshaped, axis=(1, 3))
    elif aggregation_method == 'std':
        aggregated = np.std(reshaped, axis=(1, 3))
    else:
        aggregated = np.mean(reshaped, axis=(1, 3))  # Default to mean

    return {
        'aggregated_array': aggregated.tolist(),
        'original_shape': raster_array.shape,
        'aggregated_shape': aggregated.shape,
        'aggregation_method': aggregation_method,
        'block_size': block_size,
        'compression_ratio': float(raster_array.size / aggregated.size),
        'statistics': {
            'mean': float(np.mean(aggregated)),
            'std': float(np.std(aggregated)),
            'min': float(np.min(aggregated)),
            'max': float(np.max(aggregated))
        }
    }


def mask_raster_array(raster_array, mask_condition, mask_value=None, condition_type='greater_than'):
    """
    Apply mask to raster array based on condition

    Args:
        raster_array: numpy array to mask
        mask_condition: value for masking condition
        mask_value: value to set masked pixels (None for NaN)
        condition_type: type of condition ('greater_than', 'less_than', 'equal', 'between')

    Returns:
        dict with masked array and metadata
    """
    masked_array = raster_array.copy().astype(float)

    if condition_type == 'greater_than':
        mask = raster_array > mask_condition
    elif condition_type == 'less_than':
        mask = raster_array < mask_condition
    elif condition_type == 'equal':
        mask = raster_array == mask_condition
    elif condition_type == 'between' and isinstance(mask_condition, (list, tuple)):
        mask = (raster_array >= mask_condition[0]) & (raster_array <= mask_condition[1])
    else:
        mask = np.zeros_like(raster_array, dtype=bool)

    # Apply mask
    if mask_value is None:
        masked_array[mask] = np.nan
    else:
        masked_array[mask] = mask_value

    return {
        'masked_array': masked_array.tolist(),
        'mask': mask.tolist(),
        'mask_parameters': {
            'condition': mask_condition,
            'condition_type': condition_type,
            'mask_value': mask_value
        },
        'statistics': {
            'total_pixels': int(raster_array.size),
            'masked_pixels': int(np.sum(mask)),
            'mask_percentage': float(np.sum(mask) / raster_array.size * 100),
            'valid_pixels': int(raster_array.size - np.sum(mask))
        }
    }


def compute_multi_band_index(band_arrays, index_type='ndvi', custom_formula=None):
    """
    Compute multi-band indices (NDVI, NDWI, EVI, etc.)

    Args:
        band_arrays: dict with band names as keys and numpy arrays as values
        index_type: type of index to compute ('ndvi', 'ndwi', 'evi', 'custom')
        custom_formula: custom formula for index calculation

    Returns:
        dict with computed index and metadata
    """
    if index_type == 'ndvi' and 'nir' in band_arrays and 'red' in band_arrays:
        # NDVI = (NIR - Red) / (NIR + Red)
        nir = band_arrays['nir'].astype(float)
        red = band_arrays['red'].astype(float)
        denominator = nir + red
        denominator[denominator == 0] = 1e-10  # Avoid division by zero
        index = (nir - red) / denominator
        formula = "(NIR - Red) / (NIR + Red)"

    elif index_type == 'ndwi' and 'green' in band_arrays and 'nir' in band_arrays:
        # NDWI = (Green - NIR) / (Green + NIR)
        green = band_arrays['green'].astype(float)
        nir = band_arrays['nir'].astype(float)
        denominator = green + nir
        denominator[denominator == 0] = 1e-10
        index = (green - nir) / denominator
        formula = "(Green - NIR) / (Green + NIR)"

    elif index_type == 'evi' and all(band in band_arrays for band in ['nir', 'red', 'blue']):
        # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        nir = band_arrays['nir'].astype(float)
        red = band_arrays['red'].astype(float)
        blue = band_arrays['blue'].astype(float)
        denominator = nir + 6 * red - 7.5 * blue + 1
        denominator[denominator == 0] = 1e-10
        index = 2.5 * ((nir - red) / denominator)
        formula = "2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))"

    elif index_type == 'custom' and custom_formula:
        # This would require a more complex parser - simplified version
        index = np.zeros_like(list(band_arrays.values())[0])
        formula = custom_formula
    else:
        # Default to first band if no valid index can be computed
        index = list(band_arrays.values())[0]
        formula = "First band (no valid index computed)"

    return {
        'index_array': index.tolist(),
        'index_type': index_type,
        'formula': formula,
        'bands_used': list(band_arrays.keys()),
        'statistics': {
            'mean': float(np.nanmean(index)),
            'std': float(np.nanstd(index)),
            'min': float(np.nanmin(index)),
            'max': float(np.nanmax(index)),
            'valid_pixels': int(np.sum(~np.isnan(index)))
        }
    }



def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")
