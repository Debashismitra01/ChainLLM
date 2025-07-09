import json
import os
from osgeo import gdal, osr
import numpy as np


def reproject_large_raster(input_path, output_path, target_crs, resampling_method='bilinear'):
    """
    Reproject a large raster to a new coordinate reference system

    Args:
        input_path (str): Path to input raster
        output_path (str): Path for output raster
        target_crs (str): Target CRS (e.g., 'EPSG:4326', 'EPSG:3857')
        resampling_method (str): Resampling method ('nearest', 'bilinear', 'cubic', etc.)

    Returns:
        dict: JSON-like response with status and metadata
    """
    try:
        # Open input dataset
        src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if src_ds is None:
            return {
                "status": "error",
                "message": f"Could not open input file: {input_path}",
                "input_path": input_path,
                "output_path": output_path
            }

        # Get source projection
        src_proj = src_ds.GetProjection()
        src_geotransform = src_ds.GetGeoTransform()

        # Create target spatial reference
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(int(target_crs.split(':')[1]) if 'EPSG:' in target_crs else 4326)

        # Resampling method mapping
        resampling_dict = {
            'nearest': gdal.GRA_NearestNeighbour,
            'bilinear': gdal.GRA_Bilinear,
            'cubic': gdal.GRA_Cubic,
            'cubicspline': gdal.GRA_CubicSpline,
            'lanczos': gdal.GRA_Lanczos
        }

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Perform reprojection
        gdal.Warp(
            output_path,
            src_ds,
            dstSRS=target_srs.ExportToWkt(),
            resampleAlg=resampling_dict.get(resampling_method, gdal.GRA_Bilinear),
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'TILED=YES']
        )

        # Get output info
        out_ds = gdal.Open(output_path, gdal.GA_ReadOnly)
        out_geotransform = out_ds.GetGeoTransform()

        return {
            "status": "success",
            "message": "Raster reprojected successfully",
            "input_path": input_path,
            "output_path": output_path,
            "source_crs": src_proj,
            "target_crs": target_crs,
            "resampling_method": resampling_method,
            "input_size": [src_ds.RasterXSize, src_ds.RasterYSize],
            "output_size": [out_ds.RasterXSize, out_ds.RasterYSize],
            "input_geotransform": src_geotransform,
            "output_geotransform": out_geotransform
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Reprojection failed: {str(e)}",
            "input_path": input_path,
            "output_path": output_path,
            "target_crs": target_crs
        }


def crop_merge_satellite_images(input_files, output_path, bbox=None, target_crs=None):
    """
    Crop and merge multiple satellite images

    Args:
        input_files (list): List of input raster file paths
        output_path (str): Path for merged output raster
        bbox (list): Bounding box [minx, miny, maxx, maxy] for cropping
        target_crs (str): Target CRS for output (optional)

    Returns:
        dict: JSON-like response with status and metadata
    """
    try:
        if not input_files:
            return {
                "status": "error",
                "message": "No input files provided",
                "input_files": input_files,
                "output_path": output_path
            }

        # Validate input files
        valid_files = []
        for file_path in input_files:
            if os.path.exists(file_path):
                ds = gdal.Open(file_path, gdal.GA_ReadOnly)
                if ds is not None:
                    valid_files.append(file_path)
                    ds = None

        if not valid_files:
            return {
                "status": "error",
                "message": "No valid input files found",
                "input_files": input_files,
                "output_path": output_path
            }

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build VRT (Virtual Raster) first for efficient merging
        vrt_options = gdal.BuildVRTOptions(
            outputBounds=bbox if bbox else None,
            outputSRS=target_crs if target_crs else None,
            resampleAlg='bilinear'
        )

        vrt_path = output_path.replace('.tif', '_temp.vrt')
        vrt_ds = gdal.BuildVRT(vrt_path, valid_files, options=vrt_options)

        # Translate VRT to final format
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
        )

        gdal.Translate(output_path, vrt_ds, options=translate_options)

        # Clean up temporary VRT
        if os.path.exists(vrt_path):
            os.remove(vrt_path)

        # Get output metadata
        out_ds = gdal.Open(output_path, gdal.GA_ReadOnly)
        out_geotransform = out_ds.GetGeoTransform()
        out_projection = out_ds.GetProjection()

        return {
            "status": "success",
            "message": f"Successfully merged {len(valid_files)} images",
            "input_files": valid_files,
            "output_path": output_path,
            "files_processed": len(valid_files),
            "files_skipped": len(input_files) - len(valid_files),
            "output_size": [out_ds.RasterXSize, out_ds.RasterYSize],
            "output_bands": out_ds.RasterCount,
            "output_projection": out_projection,
            "output_geotransform": out_geotransform,
            "bbox_applied": bbox,
            "target_crs_applied": target_crs
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Merge operation failed: {str(e)}",
            "input_files": input_files,
            "output_path": output_path
        }


def convert_geotiff_to_png(input_path, output_path, band_selection=None, scale_factor=1.0, nodata_transparent=True):
    """
    Convert GeoTIFF to PNG format

    Args:
        input_path (str): Path to input GeoTIFF
        output_path (str): Path for output PNG
        band_selection (list): List of band numbers to use (e.g., [1,2,3] for RGB)
        scale_factor (float): Scale factor for output size
        nodata_transparent (bool): Make nodata values transparent

    Returns:
        dict: JSON-like response with status and metadata
    """
    try:
        # Open input dataset
        src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if src_ds is None:
            return {
                "status": "error",
                "message": f"Could not open input file: {input_path}",
                "input_path": input_path,
                "output_path": output_path
            }

        # Get dataset info
        cols = src_ds.RasterXSize
        rows = src_ds.RasterYSize
        bands = src_ds.RasterCount

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Set up translation options
        translate_options = {
            'format': 'PNG',
            'width': int(cols * scale_factor),
            'height': int(rows * scale_factor),
            'creationOptions': []
        }

        # Handle band selection
        if band_selection:
            if max(band_selection) <= bands:
                translate_options['bandList'] = band_selection
            else:
                return {
                    "status": "error",
                    "message": f"Band selection {band_selection} exceeds available bands ({bands})",
                    "input_path": input_path,
                    "available_bands": bands
                }

        # Handle scaling and stretching for better visualization
        if bands >= 3:  # RGB or multispectral
            translate_options['scaleParams'] = [[0, 255, 1, 99]]  # 1-99 percentile stretch

        # Perform conversion
        gdal.Translate(
            output_path,
            src_ds,
            **translate_options
        )

        # Verify output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)

            return {
                "status": "success",
                "message": "GeoTIFF converted to PNG successfully",
                "input_path": input_path,
                "output_path": output_path,
                "input_size": [cols, rows],
                "output_size": [int(cols * scale_factor), int(rows * scale_factor)],
                "input_bands": bands,
                "bands_used": band_selection if band_selection else list(range(1, bands + 1)),
                "scale_factor": scale_factor,
                "output_file_size_mb": round(file_size / (1024 * 1024), 2),
                "nodata_transparent": nodata_transparent
            }
        else:
            return {
                "status": "error",
                "message": "PNG file was not created",
                "input_path": input_path,
                "output_path": output_path
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Conversion failed: {str(e)}",
            "input_path": input_path,
            "output_path": output_path
        }


def extract_raster_metadata(input_path):
    """
    Extract comprehensive metadata from a raster file

    Args:
        input_path (str): Path to input raster file

    Returns:
        dict: JSON-like response with complete metadata
    """
    try:
        # Open dataset
        ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if ds is None:
            return {
                "status": "error",
                "message": f"Could not open file: {input_path}",
                "input_path": input_path
            }

        # Basic dataset info
        metadata = {
            "status": "success",
            "input_path": input_path,
            "driver": ds.GetDriver().ShortName,
            "size": {
                "width": ds.RasterXSize,
                "height": ds.RasterYSize,
                "bands": ds.RasterCount
            }
        }

        # Geospatial info
        geotransform = ds.GetGeoTransform()
        if geotransform:
            metadata["geotransform"] = {
                "origin_x": geotransform[0],
                "pixel_width": geotransform[1],
                "rotation_x": geotransform[2],
                "origin_y": geotransform[3],
                "rotation_y": geotransform[4],
                "pixel_height": geotransform[5]
            }

            # Calculate extent
            width = ds.RasterXSize
            height = ds.RasterYSize
            metadata["extent"] = {
                "min_x": geotransform[0],
                "max_x": geotransform[0] + width * geotransform[1],
                "min_y": geotransform[3] + height * geotransform[5],
                "max_y": geotransform[3]
            }

        # Projection info
        projection = ds.GetProjection()
        if projection:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(projection)
            metadata["projection"] = {
                "wkt": projection,
                "proj4": srs.ExportToProj4(),
                "epsg": srs.GetAttrValue("AUTHORITY", 1) if srs.GetAttrValue("AUTHORITY") == "EPSG" else None,
                "name": srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS")
            }

        # Band information
        bands_info = []
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            band_info = {
                "band_number": i,
                "data_type": gdal.GetDataTypeName(band.DataType),
                "color_interpretation": gdal.GetColorInterpretationName(band.GetColorInterpretation()),
                "nodata_value": band.GetNoDataValue(),
                "min_value": None,
                "max_value": None,
                "mean": None,
                "std": None
            }

            # Get statistics (this might be slow for large rasters)
            try:
                stats = band.GetStatistics(True, True)
                if stats:
                    band_info.update({
                        "min_value": stats[0],
                        "max_value": stats[1],
                        "mean": stats[2],
                        "std": stats[3]
                    })
            except:
                pass

            bands_info.append(band_info)

        metadata["bands"] = bands_info

        # File metadata
        metadata["file_info"] = {
            "size_bytes": os.path.getsize(input_path),
            "size_mb": round(os.path.getsize(input_path) / (1024 * 1024), 2),
            "last_modified": os.path.getmtime(input_path)
        }

        # Dataset metadata
        dataset_metadata = ds.GetMetadata()
        if dataset_metadata:
            metadata["dataset_metadata"] = dataset_metadata

        return metadata

    except Exception as e:
        return {
            "status": "error",
            "message": f"Metadata extraction failed: {str(e)}",
            "input_path": input_path
        }


def align_rasters_with_warp(input_files, output_dir, reference_raster=None, target_resolution=None, target_crs=None,
                            resampling_method='bilinear'):
    """
    Align multiple rasters to the same grid using gdalwarp

    Args:
        input_files (list): List of input raster file paths
        output_dir (str): Directory for aligned output rasters
        reference_raster (str): Path to reference raster for alignment (optional)
        target_resolution (tuple): Target resolution as (x_res, y_res) (optional)
        target_crs (str): Target CRS (optional)
        resampling_method (str): Resampling method

    Returns:
        dict: JSON-like response with alignment results
    """
    try:
        if not input_files:
            return {
                "status": "error",
                "message": "No input files provided",
                "input_files": input_files
            }

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Determine reference parameters
        if reference_raster and os.path.exists(reference_raster):
            ref_ds = gdal.Open(reference_raster, gdal.GA_ReadOnly)
            ref_geotransform = ref_ds.GetGeoTransform()
            ref_projection = ref_ds.GetProjection()
            ref_extent = [
                ref_geotransform[0],  # min_x
                ref_geotransform[3] + ref_ds.RasterYSize * ref_geotransform[5],  # min_y
                ref_geotransform[0] + ref_ds.RasterXSize * ref_geotransform[1],  # max_x
                ref_geotransform[3]  # max_y
            ]
            target_resolution = target_resolution or (abs(ref_geotransform[1]), abs(ref_geotransform[5]))
            target_crs = target_crs or ref_projection

        # Resampling method mapping
        resampling_dict = {
            'nearest': gdal.GRA_NearestNeighbour,
            'bilinear': gdal.GRA_Bilinear,
            'cubic': gdal.GRA_Cubic,
            'cubicspline': gdal.GRA_CubicSpline,
            'lanczos': gdal.GRA_Lanczos
        }

        results = {
            "status": "success",
            "message": "Raster alignment completed",
            "output_dir": output_dir,
            "reference_raster": reference_raster,
            "target_resolution": target_resolution,
            "target_crs": target_crs,
            "resampling_method": resampling_method,
            "processed_files": [],
            "failed_files": []
        }

        # Process each input file
        for input_file in input_files:
            try:
                if not os.path.exists(input_file):
                    results["failed_files"].append({
                        "file": input_file,
                        "error": "File not found"
                    })
                    continue

                # Generate output filename
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(output_dir, f"{basename}_aligned.tif")

                # Set up warp options
                warp_options = {
                    'dstSRS': target_crs,
                    'xRes': target_resolution[0] if target_resolution else None,
                    'yRes': target_resolution[1] if target_resolution else None,
                    'resampleAlg': resampling_dict.get(resampling_method, gdal.GRA_Bilinear),
                    'format': 'GTiff',
                    'creationOptions': ['COMPRESS=LZW', 'TILED=YES']
                }

                # Add extent if reference raster provided
                if reference_raster and 'ref_extent' in locals():
                    warp_options['outputBounds'] = ref_extent

                # Perform warping
                gdal.Warp(output_file, input_file, **warp_options)

                # Verify output and get info
                out_ds = gdal.Open(output_file, gdal.GA_ReadOnly)
                if out_ds:
                    results["processed_files"].append({
                        "input_file": input_file,
                        "output_file": output_file,
                        "output_size": [out_ds.RasterXSize, out_ds.RasterYSize],
                        "output_bands": out_ds.RasterCount
                    })
                else:
                    results["failed_files"].append({
                        "file": input_file,
                        "error": "Output file not created"
                    })

            except Exception as e:
                results["failed_files"].append({
                    "file": input_file,
                    "error": str(e)
                })

        # Update final status
        if results["failed_files"] and not results["processed_files"]:
            results["status"] = "error"
            results["message"] = "All files failed to process"
        elif results["failed_files"]:
            results["status"] = "partial_success"
            results[
                "message"] = f"Processed {len(results['processed_files'])} files, {len(results['failed_files'])} failed"

        return results

    except Exception as e:
        return {
            "status": "error",
            "message": f"Alignment operation failed: {str(e)}",
            "input_files": input_files,
            "output_dir": output_dir
        }

def run(action, step, env):
    func = globals().get(action)
    if callable(func):
        return func()
    raise AttributeError(f"No function named '{action}' found.")
