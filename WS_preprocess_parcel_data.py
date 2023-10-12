# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:22:17 2022

@author: u0133999
"""
from WS_preprocess_functions import read_IACS, reclassify_IACS, bbox_tuple_to_shp, get_json_box
import geopandas as gpd
import os 
import sys
import numpy as np

def process_iacs(iacs_path, catchment_id, catchment_path, iacs_crs = 'EPSG:3035', outdir = '', 
                 export_files = True, catch_bbox_buff = 500, 
                 min_parcel_size = None, reclassify = False, out_crs = 'EPSG:3035'):
    '''
    

    Parameters
    ----------
    iacs_path : TYPE
        DESCRIPTION.
    catchment_id : TYPE
        DESCRIPTION.
    export_files : TYPE, optional
        DESCRIPTION. The default is True.
    catch_bbox_buff : TYPE, optional
        DESCRIPTION. The default is 500.

    Returns
    -------
    iacs : TYPE
        DESCRIPTION.
    bbox_buff_shp : TYPE
        DESCRIPTION.

    '''
    
    
    #define and create output files 
    ws_output_dir = os.path.join(outdir, 'ws_input_files')
    gee_dir = os.path.join(outdir, 'gee_input_files')
    
    for folder in [ws_output_dir, gee_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    #read the catchment shapefile and convert it into same crs as IACS data    
    catch = gpd.read_file(catchment_path).to_crs(iacs_crs)

    #retirn a tuple of the bounding box with buffering
    catch_bbox = get_json_box(catch, iacs_crs, buffer_distance = catch_bbox_buff, return_tuple = True)
    #obtain iacs parcels intersecting bbox with a set buffer distance - the buffer
    #distance should also be compatible/allign with the other raster inputs 
    iacs = read_IACS(iacs_path, catch_bbox, iacs_crs)
    iacs.insert(0, 'UID_c_str', np.arange(len(iacs)).astype(str))

    #convert the bbox tuple to a shpfile
    bbox_buff_shp = bbox_tuple_to_shp(catch_bbox, catchment_id, iacs_crs, out_crs)
    
    
    #optionally remove parcels that are very small
    #this can be used to avoid issues with mixing between boundaries when using satellite data
    if min_parcel_size is not None:
        iacs['Area (ha) calculated'] = iacs.area/10000
        iacs = iacs[iacs['Area (ha) calculated'] >= min_parcel_size]
    
    
    if reclassify == True:
        country_name = 'belgium_flanders'
        reclassification_ref_path = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/Data_and_software_resources/LPIS_information/GSAA_sampled_crops_list/Crop_name_reclassification.csv'
        sampled_crops_ref_path = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/C_factor/gsaasCrops/GSAA_sampled_crops_list/GSAA_crops_to_sample_' + country_name + '.csv'
        #reorganise columns 
        iacs = iacs.rename(columns = {'CULT_NAME':'crop'})
        iacs['OBJECTID'] = iacs['OBJECTID'].astype('int32')
        #reclassify iacs data
        iacs = reclassify_IACS(iacs, sampled_crops_ref_path, reclassification_ref_path)
        
    if len(iacs) == 0:
        print('Catchment IACS dataframe is empty')
        return None, None
    
    if export_files == True:
        #export in LAEA crs for WaTEM SEDEM
        iacs.to_crs(out_crs).to_file(os.path.join(ws_output_dir, 'iacs_id_'+catchment_id+'.shp'))
        bbox_buff_shp.to_crs(out_crs).to_file(os.path.join(ws_output_dir, 'bbox_id_'+catchment_id+'.shp'))
        #export in wgs84 for GEE processing 
        iacs.to_crs('EPSG:4326').to_file(os.path.join(gee_dir, 'iacs_wgs84_id_'+catchment_id+'.shp'))
        bbox_buff_shp.to_crs('EPSG:4326').to_file(os.path.join(gee_dir, 'bbox_wgs84_id_'+catchment_id+'.shp'))
    
    #return iacs dataset and bounding box for the simulation 
    return iacs, bbox_buff_shp 


#after running first function, there should be a bounding box
#all raster files need to be clipped to the bounding box
#all raster nan values need to be set appropriately 





#can iterate through function to get iacs data for all possible catchments 
#function prepares all files neccessary for GEE    





