# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:07:39 2023

@author: u0133999
"""

import sys
sys.path.insert(0, 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/Fluves_code')
sys.path.insert(0, 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/Data_and_software_resources/Monitored_watershed_data/All_python_scripts')
sys.path.insert(0, 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_preprocessing/All_python_scripts')
import cnws_kuleuven
from EUSEDcollab_time_series_functions import get_catchment_ts
import os
import numpy as np
import pandas as pd
from fnmatch import fnmatch
import re
from WS_preprocess_functions import raster_burner


def split_timeseries(r_ts_events_m, method):
    '''
    A function to split the timeseries into a calibration and validation dataset
    before running the model. The possible methods are random (50:50 event split),
    a half-half temporal split or a balanced ssl which splits the sample randomly
    but only allows a solution with a total SSL difference less than 20%.
    '''
    if method == 'random':
        ts_cal = r_ts_events_m.sample(frac = 0.6, random_state = 1).copy().sort_index()
        ts_val = r_ts_events_m[~r_ts_events_m['Event_index'].isin(ts_cal['Event_index'])].copy().sort_index()
    elif method == 'time':
        t_start = r_ts_events_m.index[0]
        t_end = r_ts_events_m.index[-1]
        t_diff = t_end - t_start
        t_middle = t_start + (t_diff/2)
        
        p1 = r_ts_events_m[:t_middle]
        p2 = r_ts_events_m[t_middle:]
        
        if len(p2) > len(p1):
            ts_cal = p2
            ts_val = p1
        else:
            ts_cal = p1
            ts_val = p2
    elif method == 'random_balanced_ssl':
        diff_pcnt = np.inf
        while diff_pcnt > 20:
            ts_cal = r_ts_events_m.sample(frac = 0.5).copy().sort_index()
            ts_val = r_ts_events_m[~r_ts_events_m['Event_index'].isin(ts_cal['Event_index'])].copy().sort_index()

            diff = ts_cal['SSL (t ha-1)'].sum() - ts_val['SSL (t ha-1)'].sum()
            diff_pcnt = int((diff / ts_cal['SSL (t ha-1)'].sum()) * 100)
        
    return ts_cal, ts_val

def run_WS_dynamic(file_paths, cnws_path, calibrate = False, n_iterations = 'All', event_indexes = None, 
                   TC_model = 'Dynamic_v1', RE_name = 'RE', ktc_cal_p = None,
                   ktc_low = 10, ktc_high = 20, WS_params = None):
    '''
    A function to run the dynamic W/S model. This will iterate through all 
    input events and produce a model simulation. This function is built upon
    the fluves pycnws class which creates the .ini file for each model run
    and launches the model through the command prompt. 
    '''
    
    if calibrate == True and ktc_cal_p == None:
        sys.exit('Provide calibration paramaters')
    elif calibrate == False and None in [ktc_low, ktc_high]:
        sys.exit('Provide ktc paramaters for non-calibration model runs')
    
    if calibrate == False and None in [ktc_low, ktc_high]:
        print('No ktc value pair provided for model outside of calibration-mode. Using default values')

    #get the r-factor time series
    r_factor_ts = file_paths['dynamic_layers_paths']['r_factor_ts']
    #get the dataframe of c-factor file paths
    slr_inputs = file_paths['dynamic_layers_paths']['slr all file paths']
    #get the dataframe of runoff 
    runoff_inputs = file_paths['dynamic_layers_paths']['runoff all file paths']
    
    
    #loop through the event iterations and update the dynamic layers each time
    if n_iterations != 'All':
        iterations = r_factor_ts['Event_index'].values[0: n_iterations]
    elif event_indexes is not None:
        iterations = r_factor_ts[r_factor_ts['Event_index'].isin(event_indexes)]['Event_index']
    else:
        iterations = r_factor_ts['Event_index']
    
    counter = 0

    for event_i in iterations:
        
        cnws = cnws_kuleuven.CNWS()
        
        x = str(event_i)
        cnws.catchm_name = f'test_event_{x}'
        
        cnws.set_choices() # set default user choices
        cnws.Variables['ktc low'] = ktc_low # change default ktc low to 7
        cnws.Variables['ktc high'] = ktc_high # change default ktc high to 12
        cnws.Variables['LS correction'] = 1 # set LS cor to 0.3
        
        if calibrate == True:
            cnws.Calibration["Calibrate"] = 1
            #if a dictionary of calibration parameters is passed. Otherwise use the defaults
            if ktc_cal_p is not None:
                cnws.Calibration["KTcHigh_lower"] = ktc_cal_p["KTcHigh_lower"]
                cnws.Calibration["KTcHigh_upper"] = ktc_cal_p["KTcHigh_upper"]
                cnws.Calibration["KTcLow_lower"] = ktc_cal_p["KTcLow_lower"]
                cnws.Calibration["KTcLow_upper"] = ktc_cal_p["KTcLow_upper"]
                cnws.Calibration["steps"] = ktc_cal_p["steps"]
        else:
            cnws.Calibration["Calibrate"] = 0
            
        
        cnws.infolder = file_paths['directory_name']
        cnws.outfolder = os.path.join(file_paths['out_folder'], 'Test_run_dynamic_event_' + x)
    
        #read the static layers
        cnws.lu = file_paths['lc_paths']['ws_lc']
        cnws.p = file_paths['p_paths']['ws_p']
        cnws.k = file_paths['k_paths']['ws_k']
        cnws.dem = file_paths['dem_paths']['ws_dem']
        #set transport capacity to the dynamic version
        cnws.ModelOptions['TC model'] = TC_model
        cnws.ModelOptions['L model'] = 'Desmet1996_Vanoost2003' #'Desmet1996_McCool' #'Desmet1996_Vanoost2003'
        cnws.ModelOptions['S model'] = 'McCool1987' #'McCool1987' #'Nearing1997'
        cnws.ModelOptions['Deposition_limited'] = 1
        cnws.ModelOptions['Deposition_limit_mm'] = 2
        

        #access the relevant intput layers for each event
        r_factor_ei = int(round(r_factor_ts[r_factor_ts['Event_index'] == event_i][RE_name]))
        slr_path = str(slr_inputs[slr_inputs['Event_index'] == event_i]['slr file path'].iloc[0])
        runoff_path = str(runoff_inputs[runoff_inputs['Event_index'] == event_i]['runoff file path'].iloc[0])
        
        cnws.c = slr_path
        cnws.r = r_factor_ei    #annual is 880
        cnws.runoff = runoff_path
        
        #overwrite any default parameter values with input values
        #the keys all need to match the ones in the cnws python module to be overwritten
        
        #here we check that no passed input parameters differ from the options
        input_var_check = list(set(WS_params.keys()) - set(cnws.Variables.keys()))
        if len(input_var_check) > 0:
            print('Input variable passed differs to the cn_ws python module options. Check inputs.')
            sys.exit()
        #set all cnws options to the passed parameters
        if WS_params is not None:
            for key in WS_params.keys():
                cnws.Variables[key] = WS_params[key]
        else:
            print('No input parameters passed. Using default values for all.')
            
        cnws.create_ini()
        cnws.copy_ini(cnws_path)
        cnws.run_model(cnws_path)
        
        counter = counter + 1
        
def collect_WS_output(events_directory, calibration = False):
    '''
    This function collects all of the W/S output into a singular dataframe. These 
    outputs are the lumped statistics produced by the model. The function processes
    either the calibration output (WS internal procedure) or the individual model
    runs from individual model runs. 
    '''
    if calibration == True:
        pattern = "*.txt"
        results = pd.DataFrame()
        for path, subdirs, files in os.walk(events_directory):
            for name in files:
                if fnmatch(name, pattern):
                    #file must contain 'Total_sediment' in the name to be read 
                    if 'calibration' in name:
                         cal_out = pd.read_csv(os.path.join(path, name), sep = ';') 
                         cal_out['Event_index'] = float(re.findall(r'\d+', path)[-1])
                         cal_out['SSL-WS (t event-1)'] = (cal_out['sed_river'] / 1000)
                         results = pd.concat([results, cal_out], ignore_index = True)
    else:
        pattern = "*.txt"
        all_vals = []
        variables = ['Event_index', 'Total gross erosion (kg)', 'Total gross deposition (kg)', 'Sediment export via river (kg)', 
                     'Sediment export (other sinks) (kg)', 'Sediment in buffers (kg)']
        for path, subdirs, files in os.walk(events_directory):
            for name in files:
                if fnmatch(name, pattern):
                    #file must contain 'Total_sediment' in the name to be read 
                    if 'Total sediment' in name:
                        #extract all values and convert them to numerical values
                        with open(os.path.join(path, name)) as file:
                            event_vals = []
                            #take the last number in the file name (event index)
                            event_vals.append(float(re.findall(r'\d+', path)[-1]))
                            for item in file:
                                vals = re.findall(r'\d+', item)
                                if 'Total erosion:' in item:
                                    val = vals[0]
                                    event_vals.append(float(val) * -1)
                                elif 'Total deposition:' in item:
                                    val = vals[0]
                                    event_vals.append(float(val))                                
                                elif 'Sediment leaving the catchment, via the river:' in item:
                                    val = vals[0]
                                    event_vals.append(float(val))                                
                                elif 'Sediment leaving the catchment, not via the river:' in item:
                                    val = vals[0]
                                    event_vals.append(float(val))                                
                                elif 'Sediment trapped in buffers:' in item:
                                    val = vals[0]
                                    event_vals.append(float(val))                                 
                                
                        all_vals.append(event_vals)
        results = pd.DataFrame(all_vals, columns = variables)
        results['SSL-WS (kg event-1)'] = results['Sediment export via river (kg)']
        results['SSL-WS (t event-1)'] = (results['Sediment export via river (kg)'] / 1000)
        
    results = results.sort_values('Event_index').reset_index(drop = True)
    
    
    return results


    
def slice_period_to_validation(df_sim, df_val, buffer_period = None):
    start_d = df_val.index[0]
    end_d = df_val.index[-1]
    if buffer_period is not None:
        start_d = start_d - pd.Timedelta(buffer_period)
        end_d = end_d + pd.Timedelta(buffer_period)
    
    df_sim = df_sim[start_d: end_d]                    
    
    return df_sim
    
def format_catchment_ts(ts, data_format, resample = False, time_resolution = None):
    
    #get a formatted version of the time series
    ts_f = get_catchment_ts(ts, data_format)
    sum_orig = ts_f['SSL (t event-1)'].sum()
    
    if resample == True:
        ts_f = ts_f.resample(time_resolution, origin = 'epoch').sum()
        sum_resampled = ts_f['SSL (t event-1)'].sum()
        diff = sum_resampled - sum_orig
        print('Resampling time resolution of the sediment yield data to: ', time_resolution)
        if diff > 1:
            print(diff)
            sys.exit('resampling has caused change in total loads: check resampling')
        
        
    del(ts_f['Month'])
    return ts_f

def modify_channel_extent(file_paths, fa_threshold): 

    lc_in = file_paths['lc_paths']['f_out_lc2'] 
    lc_out = file_paths['lc_paths']['ws_lc']
    fa_path = file_paths['fa_paths']['f_out_fa2']
    wc_to_cfactor = file_paths['input_parameters']['LC_reclassification']
    parameter_inputs = pd.read_csv(wc_to_cfactor)
    paths_shp_path = file_paths['lc_paths']['f_in_paths']
    c_factor_shp = file_paths['dynamic_layers_paths']['C-factor_shapefile']
    
    raster_burner(raster_base_path = lc_in, out_path = lc_out, shp = c_factor_shp, shp_col = 'WS_parcel_id', 
                  reclass = parameter_inputs, rc_source_col = 'WC_value', rc_target_col = 'Landcover', 
                  nd_value_in = -9999, nd_value_out = 0, fa_path = fa_path, fa_threshold = fa_threshold, 
                  stream_value = -1, paths_shp_path = paths_shp_path, path_value = -10, dtype = 'integer')


def control_splines(v):
    #Added line to prevent negative values
    #also prevent the ktc ratio parameter from exiting the range 2:5
    k = 1. * v[5]
    v = np.exp(v)
    #5 is the upper limit
    v[5] = 2 + (5 - 2)/(1 + np.exp(-k))
    
    return v

def control_params(v):
    #Added line to prevent negative values
    #also prevent the ktc ratio parameter from exiting the range 2:5
    k = 1. * v[2]
    v = np.exp(v)
    #5 is the upper limit
    v[2] = 2 + (5 - 2)/(1 + np.exp(-k))
    
    return v


def export_splines(df, path):

    out_file_p = path
    try:
        csv = pd.read_csv(out_file_p)
        csv = csv.append(df)
        csv.to_csv(out_file_p, index = False)
    except:
        df.to_csv(out_file_p, index = False)



#def calibrate_dist(df_sim_cal)

                       
#def validate(ts_obs, ts_ws):
                    
                    