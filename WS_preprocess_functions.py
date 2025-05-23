# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:58:41 2022

A series of functions to pre-process the IACS dataset before it can be used for
WaTEM-SEDEM simulations

@author: u0133999
"""
from matplotlib import colors
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd 
from shapely.geometry import box
import sys
import os
import pickle
import rasterio
from rasterstats import zonal_stats
import numpy as np
import json 
from fiona.crs import from_epsg
from rasterio.mask import mask
from osgeo import gdal
import imageio
import re


def array_check(array, out_path):
#A function to check an array before creating a raster
#returns statistics of the array which can be evaluated    

    array_info = {}
    array_info['out_path'] = out_path
    array_info['dtype'] = array.dtype.name
    array_info['a_min'] = np.amin(array)
    array_info['a_max'] = np.amax(array)
    array_info['nans'] = np.isnan(array).any()
    array_info['a_shape'] = array.shape
    
    if out_path.endswith('.rst') and array_info['dtype'] == 'float64':
        array = array.astype('float32')
        array_info['dtype'] = array.dtype.name
    
    
    return array, array_info

def plot_R(df):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    
    df['RE'].plot(style = 'o', ax = ax1)
    plt.xlabel("Year");  # custom x label using Matplotlib
    plt.ylabel("EMO-5 rainfall erosivity events");
        
    

def plot_image(array, name_str = None, out_name = None, export = True, show = False, 
               vmin = 0, vmax = 1, cmap = 'Oranges', centre_cmap = False):
#plot an array as an image

    f, ax = plt.subplots(figsize=(15, 7))
    if centre_cmap == True:
        from matplotlib import colors
        divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        im = ax.imshow(array, interpolation='nearest', cmap = 'bwr_r', alpha = 0.8)
    else:
        im = ax.imshow(array, interpolation='nearest', vmin  = vmin, vmax = vmax, cmap = cmap)
        
    if name_str is not None:
        ax.text(20, 20, name_str[:10], color = 'black', fontsize = 'x-large')
    plt.colorbar(im, ax = ax, location = 'right')
    plt.title(name_str)
    ax.set_xlim(20, 140)
    ax.set_ylim(15, 100)
    if show == True:
        plt.show()
    if export == True:
        plt.savefig(out_name + '.png', bbox_inches='tight')
    plt.close()
    
def get_gif(f_path):
#create a gif from a folder of images
    images = []
    for filename in os.listdir(f_path):
        if '.gif' in filename:
            continue
        images.append(imageio.imread(os.path.join(f_path, filename)))
    imageio.mimsave(os.path.join(f_path,'all_time_Series.gif'), images, fps = 2)

def read_raster(path):
#a core function to read raster data and pack it into a dictionary
#containing the array and relevant metadata
    #open the template raster
    rst = rasterio.open(path)
    
    gt = rst.transform
    #get raster cell size from the raster metadata
    cell_size = gt[0]
    nd_val = rst.nodatavals
    array = rst.read(1)
    
    #pack values into a dictionary - easier to add more
    raster_data = {}
    raster_data['array'] = array
    raster_data['cell size'] = cell_size
    raster_data['no data value'] = nd_val
    #copy all metadata across 
    raster_data['all metadata']= rst.meta.copy()
    raster_data['bounds'] = rst.bounds
    raster_data['affine'] = gt
    
    #close the raster
    rst.close()
    
    #return values
    return raster_data

def write_raster(array, metadata, out_path, output_type = 'GTiff'):
#a core function to write an array to a raster
#the metadata needs to be provided in the standard format returned by the 'read_raster'
#function. Any modifications need to be made in advance.
    #the correct metadata needs to be provided in advance
    #write clipped DEM to output directory 
    
    #ensure array is 2-dimensional
    if array.shape[0] == 1:
        array = array[0, :, :]
    array, array_info = array_check(array, out_path)
    
    if output_type == 'RST':
        metadata['driver'] = 'RST'
        metadata['crs'] = ''
        #metadata['crs'] = 'Plane: Projected'
    metadata['dtype'] = array_info['dtype']

    with rasterio.open(out_path, "w", **metadata) as dest:
        dest.write(array, indexes = 1)

def read_IACS(iacs_path, bbox, crs_):
#a simple read function to read IACS data        
    #read the iacs data within the provided bounding box
    iacs = gpd.read_file(iacs_path, bbox = bbox)
        
    return iacs

def remove_small_parcels(iacs, col, lower_lim):
    #a function to remove small field parcels.
    iacs = iacs[iacs[col] >= lower_lim]
    return iacs


def bbox_tuple_to_shp(tuple_, id_, in_crs, out_crs = None):
    #get a shapefile from a bounding box
    df = gpd.GeoDataFrame({"id":id_,"geometry":[box(*tuple_)]}).set_crs(in_crs)
    if out_crs is not None:
        df = df.to_crs(out_crs)
    return df

def remap_IACS_columns(iacs):
    cols = iacs.columns 
    print('Original IACS :')
    print(cols)
    return iacs
    
    
def reclassify_IACS(iacs, sampled_crops_ref_path, reclassification_ref_path):
    from remap_crops import remap_crops
    
    iacs = remap_crops(iacs, sampled_crops_ref_path, reclassification_ref_path)

    return iacs



def get_json_box(gdf_row, epsg_dest, buffer_distance = 500, return_tuple = False):
    '''
    Get the bounding box of an irregular polygon as a json

    Parameters
    ----------
    country_outlines : GEODATAFRAME
        Geodataframe containing polygon from which to take the bounding box.
    country : STRING
        Name of polygon in geodataframe to extract.
    epsg_dest : STRING
        Target coordinate system in epsg format.

    Returns
    -------
    box_json : JSON
        JSON of bounding box output.

    '''

    #get the total bounds of the irregular polygon
    target_box = gdf_row.total_bounds
    #convert to a tuple
    box_ = tuple(target_box.reshape(1, -1)[0])
    #add buffer distances to bounding boxes
    box_a = np.array(list(box_))
    box_b_a = np.array([-buffer_distance, -buffer_distance, buffer_distance, 
                        buffer_distance])
    box_ = box_a + box_b_a 
    #index variables from tuple
    minx, miny, maxx, maxy = box_[0], box_[1], box_[2], box_[3]
    
    bbox_tuple = (minx, miny, maxx, maxy)
    #create bounding box
    bbox = box(minx, miny, maxx, maxy)
    #create a geodataframe with geometric coordinates
    target_bounds = gpd.GeoDataFrame({'geometry': bbox}, index = [0], crs= epsg_dest)
    #create a json object for export 
    box_json = [json.loads(target_bounds.to_json())['features'][0]['geometry']]
    if return_tuple == True:
        return bbox_tuple
    else:
        return box_json

def add_jsons_to_gdf(gdf, epsg_dest):
    '''
    Return a json object for a row of a geodataframe

    Parameters
    ----------
    gdf : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    box_json : TYPE
        DESCRIPTION.

    '''
    all_jsons = []
    for name in gdf['Watershed ID'].unique():
        s = gdf[gdf['Watershed ID'] == name]
        #get boundaries of the geometry
        box_ = s.iloc[0].geometry.bounds
        #index relevent extents
        minx, miny, maxx, maxy = box_[0], box_[1], box_[2], box_[3]
        #create bounding box
        bbox = box(minx, miny, maxx, maxy)
        #create geodataframe with only geometry
        target_bounds = gpd.GeoDataFrame({'geometry': bbox}, index = [0], crs= epsg_dest)
        #create json from geometry
        box_json = [json.loads(target_bounds.to_json())['features'][0]['geometry']]
        all_jsons.append(box_json)
    gdf['bbox_json'] = all_jsons
    return gdf

def clip_raster_to_box(mask_type, file_in, out_path, epsg_dest, catchment = None,
                       predefined_box_tuple = None, nd_value_in = -9999, nd_value_out = -9999,
                       target_resolution = 25, dtype = 'float64', routine = None):
    '''
    Clip a large raster file to a bounding box

    Parameters
    ----------
    file_in : STRING
        Pathway to the input raster file.
    file_out : STRING
        Pathway to the output raster file
    box_json : JSON
        Boundary coordinates of the desired bounding box.

    Returns
    -------
    None.

    '''
    if predefined_box_tuple is not None:
        bounds = predefined_box_tuple
        crop = True
    else:
        if predefined_box_tuple is None and mask_type == 'box':
            #for the GDAL routine 
            bounds = get_json_box(catchment, epsg_dest, return_tuple = True)
            crop = True
        elif mask_type == 'catchment':
            #for the rasterio routine - raster will read as an array 
            crop = False
            bounds = catchment.geometry
        else:
            sys.exit('specify valid mask type string')
        
    if mask_type == 'box':
        #use gdal warp to perform the singular operation. This will resample and
        #clip the data layer in one operation 
        
        if routine == 'DEM':
            ds_rp = gdal.Warp(out_path, file_in, outputBounds = bounds,
                          dstSRS = epsg_dest, xRes= target_resolution, yRes= target_resolution, 
                          resampleAlg=gdal.GRA_NearestNeighbour, outputType = gdal.GDT_Float32,
                          copyMetadata = True)
        else:
            ds_rp = gdal.Warp(out_path, file_in, outputBounds = bounds,
                          dstSRS = epsg_dest, xRes= target_resolution, yRes= target_resolution, 
                          resampleAlg=gdal.GRA_NearestNeighbour, outputType = gdal.GDT_Float32,
                          copyMetadata = True)
        ds_rp = None 
    
    
    elif mask_type == 'catchment':
        #define the driver
        if out_path.endswith('.rst'):
            driver = 'RST'
        else:
            driver = 'GTiff'
        
        #use rasterio to open entire DEM and mask it based on the catchment area
        all_rast = rasterio.open(file_in)
        #crs_orig = all_rast.crs 
        
        #do a data type check
        if all_rast.meta['dtype'] == 'float32':
            nd_value = np.float32(nd_value_out)
        else:
            print('check data type of nodata value - not float32')
            
        
        #mask based on the json extent 
        out_img, out_transform = mask(all_rast, shapes= bounds, crop=crop, 
                                      nodata = nd_value_out, all_touched = True)
        
        #copy all metadata across 
        out_meta = all_rast.meta.copy()
        
        #update relevant metadata fields 
        out_meta.update({"driver": driver,
                          "height": out_img.shape[1],
                          "width": out_img.shape[2],
                          "transform": out_transform,
                          "nodata": nd_value})
        
        all_rast.close()
        if routine == 'K-factor':
            #convert from t to kg and make array an integer type

            out_img = out_img * 1000
            out_img = np.where(out_img < 0, -9999, out_img)
            out_img = out_img.astype(int)
            out_meta['dtype'] = 'int16'
        
        
        write_raster(out_img, out_meta, out_path, output_type = driver)
        '''
        out_img, array_info = array_check(out_img, out_path)

        #write clipped DEM to output directory 
        with rasterio.open(out_path, "w", **out_meta) as dest:
             dest.write(out_img)
        '''
        
def raster_burner(raster_base_path, out_path, shp, shp_col, reclass = None, 
                  rc_source_col = None, rc_target_col = None, nd_value_in = -9999,
                  nd_value_out = -9999, fa_path = None, fa_threshold = None, 
                  streams_shp = None, stream_value = -1, paths_shp_path = None, path_value = -10, 
                  dtype = 'float', plot = False, image_folder = None):
    #A function to burn a raster with standard dimensions from a shapefile containing 
    #specific values. The column of the shapefile to burn needs to be specified.
    
    #open the template (landcover) raster
    rst = rasterio.open(raster_base_path)
    lc_array = rst.read(1)

    #copy metadata to make a target raster
    meta = rst.meta.copy() 
    meta.update({'nodata': nd_value_in})      
    rst.close()
    
    
    #create a target raster    
    with rasterio.open(out_path, 'w+', **meta) as out:
        #read array (nodata rasterio array)
        out_arr = out.read(1)
        # this is where we create a generator of geom, value pairs to use in rasterizing
        #here the specified column values are attributed to each shape
        shapes = ((geom,value) for geom, value in zip(shp.geometry, shp[shp_col]))
        
        #burn shapefile features onto target array
        burned = rasterio.features.rasterize(shapes=shapes, fill= nd_value_in, out=out_arr, transform=out.transform)
        #ensure that array is all float values
        burned = burned.astype(float)
        
        if fa_path is not None and streams_shp is not None:
            print('Stream shapefile and flow accumulation layer given for streams. Defaulting to flow accumulation.')
        #get masks of the streams and paths to burn on to the array later
        if streams_shp is not None:
            
            burned_streams = rasterio.features.rasterize(shapes=streams_shp.geometry, fill= nd_value_in,
                                                         out=out_arr, transform=out.transform, default_value= -555)
            stream_mask = burned_streams == -555
            
        elif fa_path is not None:
            fa_ = rasterio.open(fa_path)
            fa_array = fa_.read(1)
            
            if not fa_array.shape == lc_array.shape:
                sys.exit('Mismatch between dimensions of landcover and flow accumulation layer. Check layers.')
            if fa_threshold >= np.amax(fa_array):
                fa_threshold = np.amax(fa_array)
                print('Flow accumulation threshold exceeds layer max. Stream given maximum FA cell value.')
            #copy metadata to make a target raster
            fa_meta = fa_.meta.copy()       
            fa_.close()
            #create a stream mask where flow accumulation exceeds threshold
            stream_mask = fa_array >= fa_threshold
            
            
            
        if paths_shp_path is not None:
            #open paths shapefile
            paths_shp = gpd.read_file(paths_shp_path, driver = 'ESRI Shapefile')
            paths_shp = paths_shp[paths_shp['geometry'] != None]
            
            burned_paths = rasterio.features.rasterize(shapes=paths_shp.geometry, 
                                                       fill= nd_value_in, out=out_arr, transform=out.transform, default_value= -333)
            path_mask = burned_paths == -333  
            
        #if a value reclassification is to be made, enter code section
        if reclass is not None:
            #create disctionary of masks 
            mask_dict = {}
            #go through a series of mask reclassifications to add permanent lc elements
            for i in np.arange(len(reclass)):
                lc_row = reclass.iloc[i]
                #set the mask where lc value is the desired value and no parcel is present
                #this prioritises areas with field parcels so that a landcover map only
                #fills between. The condition is AND. 

                mask = (lc_array == lc_row[rc_source_col]) & (burned == nd_value_in)
                
                if mask.sum() > 0:
                    #if landcover element is present, burn it on to raster
                    burned = np.where(mask == True, float(lc_row[rc_target_col]), burned)
                    #store masks as a dictionary 
                    mask_dict[lc_row['Description']] = mask
        
            
        #include the path pixels
        if paths_shp_path is not None: 
            burned = np.where(path_mask == True, path_value, burned)  
            
        #include the stream pixels
        #These burn on top of paths to ensure priority
        if 'streams_shp' in locals(): 
            burned = np.where(stream_mask == True, stream_value, burned)
        
        #set the out of bounds delineation from the landcover raster 
        bounds_mask = lc_array == nd_value_in
        if dtype == 'integer':
            #use a value of zero for no data
            burned = np.where(bounds_mask == True, 0, burned)
        else:
            #otherwise set to the desired output nodata value
            burned =  np.where(bounds_mask == True, nd_value_out, burned)
            
        #convert to an integer and update relevant metadata
        if dtype == 'integer':
            burned = burned.astype(int)
            meta.update({'dtype':'int16'}) 

        if plot == True:
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            f_name = 'C_factor_gif_' + re.findall(r'\d+', out_path)[-1]
            out_name = os.path.join(image_folder, f_name)
            plot_image(burned, shp_col, out_name)

        #close the rasterio object but keep the array and updated metadata dictionary
        out.close()
        #use the write function to create a new raster with the correct drivers
        
        if out_path.endswith('.rst'):
            driver = 'RST'
        else:
            driver = 'GTiff'
        
        write_raster(burned, meta, out_path, output_type = driver)

def raster_burner2(raster_base_path, out_path, shp, shp_col, reclass = None, 
                  rc_source_col = None, rc_target_col = None, nd_value_in = -9999,
                  nd_value_out = -9999, fa_path = None, fa_threshold = None, 
                  streams_shp = None, stream_value = -1, paths_shp_path = None, path_value = -10, 
                  dtype = 'float', plot = False, image_folder = None):
    #A function to burn a raster with standard dimensions from a shapefile containing 
    #specific values. The column of the shapefile to burn needs to be specified.
    
    #open the template (landcover) raster
    rst = rasterio.open(raster_base_path)
    lc_array = rst.read(1)

    #copy metadata to make a target raster
    meta = rst.meta.copy() 
    meta.update({'nodata': nd_value_in}) 
    meta.update({'dtype': dtype}) 
    print(meta)     
    rst.close()
    
    
    #create a target raster    
    
    with rasterio.open(out_path, 'w+', **meta) as out:
        #read array (nodata rasterio array)
        out_arr = out.read(1)
        # this is where we create a generator of geom, value pairs to use in rasterizing
        #here the specified column values are attributed to each shape
        shapes = ((geom,value) for geom, value in zip(shp.geometry, shp[shp_col]))
        
        #burn shapefile features onto target array
        burned = rasterio.features.rasterize(shapes=shapes, fill= nd_value_in, out=out_arr, transform=out.transform)
        #ensure that array is all float values
        burned = burned.astype(float)
        
        if fa_path is not None and streams_shp is not None:
            print('Stream shapefile and flow accumulation layer given for streams. Defaulting to flow accumulation.')
        #get masks of the streams and paths to burn on to the array later
        if streams_shp is not None:
            
            burned_streams = rasterio.features.rasterize(shapes=streams_shp.geometry, fill= nd_value_in,
                                                         out=out_arr, transform=out.transform, default_value= -555)
            stream_mask = burned_streams == -555
            
        elif fa_path is not None:
            fa_ = rasterio.open(fa_path)
            fa_array = fa_.read(1)
            
            if not fa_array.shape == lc_array.shape:
                sys.exit('Mismatch between dimensions of landcover and flow accumulation layer. Check layers.')
            if fa_threshold >= np.amax(fa_array):
                fa_threshold = np.amax(fa_array)
                print('Flow accumulation threshold exceeds layer max. Stream given maximum FA cell value.')
            #copy metadata to make a target raster
            fa_meta = fa_.meta.copy()       
            fa_.close()
            #create a stream mask where flow accumulation exceeds threshold
            stream_mask = fa_array >= fa_threshold
            
            
            
        if paths_shp_path is not None:
            #open paths shapefile
            paths_shp = gpd.read_file(paths_shp_path, driver = 'ESRI Shapefile')
            paths_shp = paths_shp[paths_shp['geometry'] != None]
            
            burned_paths = rasterio.features.rasterize(shapes=paths_shp.geometry, 
                                                       fill= nd_value_in, out=out_arr, transform=out.transform, default_value= -333)
            path_mask = burned_paths == -333  
            
        #if a value reclassification is to be made, enter code section
        if reclass is not None:
            #create disctionary of masks 
            mask_dict = {}
            #go through a series of mask reclassifications to add permanent lc elements
            for i in np.arange(len(reclass)):
                lc_row = reclass.iloc[i]
                #set the mask where lc value is the desired value and no parcel is present
                #this prioritises areas with field parcels so that a landcover map only
                #fills between. The condition is AND. 

                mask = (lc_array == lc_row[rc_source_col]) & (burned == nd_value_in)
                
                if mask.sum() > 0:
                    #if landcover element is present, burn it on to raster
                    burned = np.where(mask == True, float(lc_row[rc_target_col]), burned)
                    #store masks as a dictionary 
                    mask_dict[lc_row['Description']] = mask
        
            
        #include the path pixels
        if paths_shp_path is not None: 
            burned = np.where(path_mask == True, path_value, burned)  
            
        #include the stream pixels
        #These burn on top of paths to ensure priority
        if streams_shp  is not None: 
            burned = np.where(stream_mask == True, stream_value, burned)
        
        #set the out of bounds delineation from the landcover raster 
        bounds_mask = lc_array == nd_value_in
        if dtype == 'integer':
            #use a value of zero for no data
            burned = np.where(bounds_mask == True, 0, burned)
        else:
            #otherwise set to the desired output nodata value
            burned =  np.where(bounds_mask == True, nd_value_out, burned)
            
        #convert to an integer and update relevant metadata
        if dtype == 'integer':
            burned = burned.astype(int)
            meta.update({'dtype':'int16'}) 

        if plot == True:
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            f_name = 'C_factor_gif_' + re.findall(r'\d+', out_path)[-1]
            out_name = os.path.join(image_folder, f_name)
            plot_image(burned, shp_col, out_name)

        #close the rasterio object but keep the array and updated metadata dictionary
        out.close()
        #use the write function to create a new raster with the correct drivers
        
        if out_path.endswith('.rst'):
            driver = 'RST'
        else:
            driver = 'GTiff'
        
        write_raster(burned, meta, out_path, output_type = driver)



def get_p_factor(raster_base_path, out_path, output_type = 'RST', dtype = 'float'):
#create a p factor raster
    #read the raster to extract info
    in_raster_data = read_raster(raster_base_path)
    p_array = in_raster_data['array']
    metadata = in_raster_data['all metadata']
    
    #reclassify the array and set to an integer
    p_array = p_array.astype(float)
    p_array = np.where(p_array == metadata['nodata'], 0., 1.)
    metadata['dtype'] = 'float32'
    
    if dtype == 'integer':
        p_array = p_array.astype(int)
        metadata['dtype'] = 'int16'
        metadata['nodata'] = 0
    write_raster(p_array, metadata, out_path, output_type = output_type)
    
    
def check_dynamic_cfactor(slr_ts_event, RE_col): 
#check the C-factor to see if dynamic values make sense
    #only take columns with integer parcel indices
    cols = [s for s in slr_ts_event.columns if isinstance(s, int)]
    df_parcels = slr_ts_event[cols]
    #sum the total R-factor
    r_sum = slr_ts_event[RE_col].sum()
    r_event = slr_ts_event[RE_col]

    df_parcels = df_parcels.multiply(r_event, axis = 0)
    
    c_parcels = df_parcels.sum()/r_sum
    
    return c_parcels
    
    
def get_zonal_stats(raster_path, gdf_path, nd_value = None, ero_dep = None,
                    remove_extremes = False):
    #example: https://gis.stackexchange.com/questions/297076/how-to-calculate-mean-value-of-a-raster-for-each-polygon-in-a-shapefile
    gdf = gpd.read_file(gdf_path)
    with rasterio.open(raster_path) as src:
        affine = src.transform
        array = src.read(1)
            
        if nd_value is not None:
            array = np.where(array == nd_value, 0, array)
            
        if ero_dep is not None:
            if ero_dep == 'erosion':
                array = np.where(array < 0, array, 0)
            elif ero_dep == 'deposition':
                array = np.where(array > 0, array, 0)
                
        if remove_extremes == True:
            array = np.where(array < -30, 0, array)
            array = np.where(array < 30, array, 0)
                
        df_zonal_stats = pd.DataFrame(zonal_stats(gdf, array, affine=affine,
                                                  stats=['min', 'max', 'median', 'mean', 'sum', 'count']))

    gdf2 = pd.concat([gdf, df_zonal_stats], axis=1) 
    
    return gdf2
    