# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:28:54 2022

This script pre-processes all input layers neccessary to run WaTEM/SEDEM
in a multitemporal format. The input folder needs to be specified in advance
and thereafter all relevant input layers are processed. The relevant paths 
are packed into a dictionary and exported, to be read during the model 
implementation routine.

@author: Francis Matthews (fmatthews1381@gmail.com)
"""

from WS_preprocess_parcel_data import process_iacs
from WS_preprocess_functions import get_json_box, clip_raster_to_box, raster_burner, get_p_factor, get_gif, check_dynamic_cfactor
from CN_calculations import NCRS_CN, calc_NAPI
from Rfactor import correct_rfactor
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import sys
import pickle
from EUSEDcollab_time_series_functions import format_catchment_p, add_rfactor
from Bias_correction_functions import ei30_from_ts

ID_eused = 6
id_ = str(ID_eused)
Country = 'BE'

f_dir_in = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_preprocessing/WS_inputs_EUSEDcollab'
f_dir_out = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_preprocessing/WS_processed_EUSEDcollab/WS_inputs_id_'+id_
os.chdir(f_dir_in)

epsg_dest = 'EPSG:3035'
start_date = '01-01-1996'
end_date = '01-01-2000'
R_longterm = 880
#n cells is 50 hectares (typical channel threshold for Flanders)
fa_threshold = 800  #800 #15000 #2000
resample_time = True
time_resolution = '15D'
use_gauge_precip = True
use_gauge_rfactor = True

file_paths_all = {}

create_gifs = False
plot = False

'''
Define all the files. This is done as dictionaries for export later
'''
wc_to_cfactor = os.path.join(f_dir_in,'Worldcover_to_Cfactor.csv')
cn_table_p = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_preprocessing/CN_parameter_values.csv'

#path to k-factor
k_paths = {}
k_paths['f_in_k'] = os.path.join(f_dir_in,'K_new_crop.tif')
k_paths['f_out_k'] = f'gdal_K_factor_id_{id_}.tif'
k_paths['ws_k'] = f'K_factor_catchment_id_{id_}.rst'#.rst
#path to dem file
dem_paths = {}
dem_paths['f_in_dem'] = os.path.join(f_dir_in, f'Pysheds_outputs/{id_}all_DEM.tif')
dem_paths['ws_dem'] = f'dem_catchment_id_{id_}.rst'#.rst
#path to flow accumulation file for river network
fa_paths = {}
fa_paths['f_in_fa'] = os.path.join(f_dir_in, f'Pysheds_outputs/{id_}FA.tif')
fa_paths['f_out_fa'] = f'gdal_FA_id{id_}.tif'
fa_paths['f_out_fa2'] = f'FA_catchment_id{id_}.tif'#.rst
#path to landcover
lc_paths = {}
lc_paths['f_in_lc'] = os.path.join(f_dir_in, f'Worldcover/Worldcover_id_{id_}.tif')
lc_paths['f_out_lc'] = f'gdal_LC_id_{id_}.tif'
lc_paths['f_out_lc2'] = f'LC_catchment_id_{id_}.tif'
lc_paths['ws_lc'] = f'Landcover_id_{id_}.rst'#.rst
lc_paths['ws_lc tif'] = f'Landcover_id_{id_}.tif'
lc_paths['f_in_paths'] = os.path.join(f_dir_in,f'Open_streetmaps/OSM_roads_id_{id_}.shp')
#path to P-factor layer -- only an output
p_paths = {}
p_paths['ws_p'] = f'P_factor_catchment_id{id_}.rst'#.rst 
#path to C-factor shapefile
c_paths = {}
c_paths['slr_ts_in'] = os.path.join(f_dir_in,f'C-factor/SLR_timeseries/id_{id_}_SLR_timeseries_10_ndvi_py.csv')
c_paths['c_factor_in'] = os.path.join(f_dir_in,f'C-factor/SLR_timeseries/id_{id_}_Cfactor.shp')
#record the input parameters that were selected to create data layers
input_parameters = {}
input_parameters['R-factor annual average'] = R_longterm
input_parameters['River flow acc threshold'] = fa_threshold
input_parameters['LC_reclassification'] = wc_to_cfactor
input_parameters['catchment shapefile'] = os.path.join(f_dir_in, f'Pysheds_outputs/{id_}.shp')
input_parameters['EnS'] = 'ATC4'
input_parameters['Start date'] = start_date
input_parameters['End date'] = end_date
#define some output folders for dynamic layers
dynamic_layers_paths = {}
dynamic_layers_paths['EMO_precip_in'] = os.path.join(f_dir_in, 'EUSEDcollab_timeseries_pr6.csv')
dynamic_layers_paths['EUSEDcollab_precip_in'] = os.path.join(f_dir_in, f'PRECIPITATION/ID_{ID_eused}_PRECIP_{Country}.csv')
dynamic_layers_paths['Gauge_R-factor_in'] = os.path.join(f_dir_in,f'Rfactor_gauge_data/ID_{ID_eused}_PRECIP.csv')
dynamic_layers_paths['c_out_folder'] = f_dir_out
dynamic_layers_paths['cn_r_out_folder'] = f_dir_out
dynamic_layers_paths['cfactor_images'] = 'Cfactor_gifs'
dynamic_layers_paths['cn_images'] = 'CN_gifs'

'''
Create all neccessary output folders
'''

#make folders if they don't exist
if not os.path.exists(f_dir_out):
    os.makedirs(f_dir_out)
    
#change directory to write all files to out directory    
os.chdir(f_dir_out)
    
for key in dynamic_layers_paths.keys():
    if not os.path.exists(dynamic_layers_paths[key]):
        os.makedirs(dynamic_layers_paths[key])

'''
Run the processing for the rainfall. The pre-processing has several options,
including the possibility to use EMO data to force the model. The rainfall 
data ultimately forms the model forcing which defines time periods with erosive
rainfall. The rainfall erosivity is defined at the event scale and then resampled 
(here to 15-days).
'''

alpha_p = os.path.join(f_dir_in, 'alpha_params_v2.shp')
beta_p = os.path.join(f_dir_in, 'beta_params_v2.shp')
emo5_pr = pd.read_csv(dynamic_layers_paths['EMO_precip_in'])
emo5_pr.index = pd.to_datetime(pd.to_datetime(emo5_pr['Date'], dayfirst = True, format='%Y-%m-%d %H:%M:%S'))
alpha_m = gpd.read_file(alpha_p)
beta_m = gpd.read_file(beta_p)
station_name = 'Station_Id ' + id_
emo5_pr_st = pd.DataFrame(emo5_pr[station_name])
r_events = ei30_from_ts(emo5_pr_st, input_parameters['EnS'], station_name, alpha_m, beta_m, time_resolution = 6)
#r_events = r_events[pd.to_datetime(start_date, dayfirst=True) : pd.to_datetime(end_date, dayfirst=True)]
napi_ts = calc_NAPI(emo5_pr_st.rename(columns = {station_name: 'precip mm'}))
r_ts_in = dynamic_layers_paths['EMO_precip_in']

if use_gauge_precip == True:
    print('Overwriting precipitation data with catchment gauge measurements')
    #OVERWRITING THE R-FACTOR WITH THE CATCHMENT DATA
    precip_catch = pd.read_csv(dynamic_layers_paths['EUSEDcollab_precip_in'], usecols = [0,1,2])
    precip_dict = format_catchment_p(precip_catch)
    r_events = add_rfactor(precip_dict['Precip events'], 'ATC4')
    r_ts_in = dynamic_layers_paths['EUSEDcollab_precip_in']


#resample to 15-day and add relevant columns
if resample_time == True:
    r_events = r_events.resample('15D', origin = 'epoch').sum()
    print('Resampling time resolution of the sediment yield data to: 15day')
    #don't consider 0 or very insignificant events
    r_events = r_events[r_events['RE'] > 2]
#tidy up the dataframe after resampling
r_events['Month'] = r_events.index.month
r_events = r_events.rename(columns = {'New event': 'N events in sum'})
r_events['Start timestamp'] = r_events.index
r_events['End timestamp'] = r_events['Start timestamp'] + pd.DateOffset(days=15)
r_events['Event_index'] = np.arange(len(r_events))
r_events = r_events[start_date:end_date]
r_events = correct_rfactor(r_events, 'RE', R_longterm)


if use_gauge_rfactor == True:
    if use_gauge_precip == False:
        sys.exit('Use catchment rainfall. Select: use_gauge_precip == True')
    #add the measured events from the Kinderveld - made using the R-factor algorithm
    rf_events_gauge = pd.read_csv(dynamic_layers_paths['Gauge_R-factor_in'])
    rf_events_gauge.index = pd.to_datetime(rf_events_gauge['datetime'], yearfirst= True)
    if resample_time == True:
        rf_events_gauge = rf_events_gauge.resample('15D', origin = 'epoch').sum()
        rf_events_gauge['max_30min_intensity'] = rf_events_gauge.resample('15D', origin = 'epoch').max()['max_30min_intensity']
        rf_events_gauge = rf_events_gauge[rf_events_gauge['erosivity'] > 0]
    rf_events_gauge = rf_events_gauge[start_date:end_date]
    rf_events_gauge = correct_rfactor(rf_events_gauge, 'erosivity', R_longterm)
    #overwrite the R-factor with the gauge data
    print('Overwriting RE with gauge calculations')
    r_events['RE_gauge'] = rf_events_gauge['erosivity']
    r_events['RE corr_gauge'] = rf_events_gauge['erosivity corr']
    r_events['I30_gauge'] = rf_events_gauge['max_30min_intensity']
    r_ts_in = dynamic_layers_paths['Gauge_R-factor_in']


#write the r-factor information into the dictionary
dynamic_layers_paths['r_factor_path'] = r_ts_in
dynamic_layers_paths['r_factor_ts'] = r_events
dynamic_layers_paths['resample time'] = resample_time
dynamic_layers_paths['time_resolution'] = time_resolution 

'''
Process the C-factor as a dataframe.
'''

#read catchment shapefile
catchment = gpd.read_file(input_parameters['catchment shapefile'])
#read C-factor shapefile
c_factor_shp = gpd.read_file(c_paths['c_factor_in'])
#read SLR and create a full c-factor and slr dataframe
slr_ts = pd.read_csv(c_paths['slr_ts_in']).set_index('object_id').sort_index()
#THIS LINE IS NEEDED FOR KINDERVELD DATA
slr_ts.columns = pd.to_datetime(slr_ts.columns, dayfirst = True).strftime('%Y-%m-%d %h:%m:%s')
#mask very low slr values and fill them later. Arable land shouldn't overlap with non-arable land because of C-factor
slr_ts = slr_ts.mask(slr_ts < 0.03)
slr_ts[slr_ts > 0.9] = 0.9

#Merge the SLR timeseries. Any na parcels will be considered as grassland.
c_factor_shp = c_factor_shp.merge(slr_ts, on = 'object_id', how = 'left').fillna(0.03)

#create a column to give an integer id value to each parcel
c_factor_shp['WS_parcel_id'] = np.arange(1, len(c_factor_shp) + 1)
parameter_inputs = pd.read_csv(wc_to_cfactor)

'''
Get k-factor, DEM and landcover formatted into the catchment bounds
- these can all pack into a dictionary and looped through
'''
for dic in [k_paths, dem_paths, lc_paths, fa_paths]:
    keys = list(dic.keys())
    #if there are 3 paths, an extra clip to the catchment needs to be undertaken
    if len(keys) >= 3:
        catchment_routine =True
        #define the final output file path
        out_f = dic[keys[2]]
    else:
        catchment_routine = False
        #define the final output file path
        out_f = dic[keys[1]]
    
    #define if k-factor processing is required. This needs some additional 
    #processing
    if 'K_factor' in out_f:
        routine = 'K-factor'
    elif 'dem_' in out_f:
        routine = 'DEM'
    else:
        routine = None
    
    clip_raster_to_box('box', dic[keys[0]], dic[keys[1]], epsg_dest, catchment = catchment, 
                       routine = routine)
    
    if catchment_routine == True:
        #clip raster box to a catchment geometry - areas outside are masked and given no
        #data value - this can be specific for WATEM-SEDEM 
        clip_raster_to_box('catchment', dic[keys[1]], dic[keys[2]], epsg_dest, catchment = catchment, 
                           routine = routine)

'''
Here the time periods with rainfall erosivity are merged with the parcel-scale 
SLR values to describe the catchment situation.
'''

#transpose so a datetime index is present 
slr_ts_event = slr_ts.transpose()
slr_ts_event = slr_ts_event.set_index(pd.to_datetime(slr_ts_event.index, dayfirst = True))
#merge with the event template to get the SLR per event. This means the same SLR
#time series can be associated with close events in time
slr_ts_event['SLR date'] = slr_ts_event.index
slr_ts_event = pd.merge_asof(r_events, slr_ts_event, left_index = True, right_index = True,
                             direction = 'nearest')

cfactor_check = check_dynamic_cfactor(slr_ts_event, 'RE corr')
c_factor_shp['C-factor all ts'] = cfactor_check

event_indexes = slr_ts_event['Event_index'].values
event_dates = slr_ts_event['SLR date'].astype(str).values


'''
Process the parcel map as a .rst file and a .tif file. The tif file is used for 
further processing in the runoff module.
'''

#export landcover as an integer in .rst format. no data value is 0 (in function)
raster_burner(raster_base_path = lc_paths['f_out_lc2'], out_path = lc_paths['ws_lc'], shp = c_factor_shp, shp_col = 'WS_parcel_id', 
              reclass = parameter_inputs, rc_source_col = 'WC_value', rc_target_col = 'Landcover', 
              nd_value_in = -9999, nd_value_out = 0, fa_path = fa_paths['f_out_fa2'], fa_threshold = fa_threshold, 
              stream_value = -1, paths_shp_path = lc_paths['f_in_paths'], path_value = -10, dtype = 'integer')
    #export landcover as a float in goetiff format
raster_burner(raster_base_path = lc_paths['f_out_lc2'], out_path = lc_paths['ws_lc tif'], shp = c_factor_shp, shp_col = 'WS_parcel_id', 
              reclass = parameter_inputs, rc_source_col = 'WC_value', rc_target_col = 'Landcover', 
              nd_value_in = -9999, nd_value_out = -9999, fa_path = fa_paths['f_out_fa2'], fa_threshold = fa_threshold,
              stream_value = -1, paths_shp_path = lc_paths['f_in_paths'], path_value = -10)

'''
Process an SLR raster and CN runoff coefficient for each event 
'''

lc_cn_reclass = dict(zip(parameter_inputs['Landcover'].values, parameter_inputs['CN_approximation'].values))

c_files = []
runoff_files = []
count = 0
for i in np.arange(len(slr_ts_event)):
    
    col_name = str(slr_ts_event['SLR date'].iloc[i])
    event_index = event_indexes[i]
    
    file_c = f'Cfactor_id{id_}_{event_index}.rst'
    out_path_c = file_c
    #the C_factor has a 0 value for out of bounds
    raster_burner(raster_base_path = lc_paths['f_out_lc2'], out_path = out_path_c, shp = c_factor_shp, shp_col = col_name,
                  reclass = parameter_inputs, rc_source_col = 'WC_value', rc_target_col = 'C_factor',
                  nd_value_in = -9999, nd_value_out = 0, fa_path = fa_paths['f_out_fa2'], fa_threshold = fa_threshold, 
                  stream_value = 0, paths_shp_path = lc_paths['f_in_paths'], path_value = 1, plot = plot, 
                  image_folder = dynamic_layers_paths['cfactor_images'])
    
    c_files.append([event_index, out_path_c])
    
    event_index = event_indexes[i]   
    event_date = event_dates[i] 
    napi_event = napi_ts.loc[pd.to_datetime(event_date, yearfirst=True)]['NAPI']
    
    
    file_cn = f'CN_rf_id{id_}_{event_index}.rst'
    out_path_r =  file_cn
    
    rf_event_mm = slr_ts_event['RE'].values[i]
    event_d = slr_ts_event['Event dur (d)'].values[i]
    
    Q_grid_m, Q_dist_m3, Q_lumped_m3 = NCRS_CN(lc_paths['ws_lc tif'], parameter_inputs, rf_event_mm, Ia_coeff = 0.02, runoff_type = 'coefficient', 
                                               duration_d = event_d, slr_array_p = out_path_c, NAPI = napi_event, export_raster = True, 
                                               out_path = out_path_r, plot_grid = plot, HSG_catchment = 4,
                                               event_date = event_date, image_folder = dynamic_layers_paths['cn_images'])

    
    runoff_files.append([event_index, out_path_r])
    
    count = count + 1


if create_gifs == True:
    get_gif(dynamic_layers_paths['cfactor_images'])
    get_gif(dynamic_layers_paths['cn_images'])
    
dynamic_layers_paths['slr all file paths'] = pd.DataFrame(c_files, columns = ['Event_index', 'slr file path'])
dynamic_layers_paths['runoff all file paths'] = pd.DataFrame(runoff_files, columns = ['Event_index', 'runoff file path'])    
dynamic_layers_paths['C-factor_shapefile'] = c_factor_shp

'''
Process a P-factor layer corresponding to the boundary conditions
'''

get_p_factor(lc_paths['ws_lc tif'], p_paths['ws_p'], output_type = 'RST')

'''
Compile all dictionaries into a master dictionary for export 
'''

file_paths_all['k_paths'] = k_paths
file_paths_all['dem_paths'] = dem_paths
file_paths_all['fa_paths'] = fa_paths
file_paths_all['lc_paths'] = lc_paths
file_paths_all['p_paths'] = p_paths
file_paths_all['c_paths'] = c_paths
file_paths_all['dynamic_layers_paths'] = dynamic_layers_paths
file_paths_all['EUSEDcollab_key'] = f'ID_{id_}'
file_paths_all['directory_name'] = f_dir_out
file_paths_all['input_parameters'] = input_parameters
pickle_p = os.path.join(f_dir_out, 'ws_file_paths.pickle')
pickle.dump(file_paths_all, open(pickle_p, 'wb'))
