# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:33:12 2023

This is the implementation code for the multitemporal (15-day) version of
WaTEM-SEDEM. The implementation is in an iterative format to allow the
creation of multiple scenarios from the W/S structural parameters if desired.

This implementation runs the WS_optimise class which contains two methods 
to calibrate W/S:
    
    1) a temporally static method in which a singular calibration parameter 
    set (the 2 ktc parameters). Analagous to the traditional W/S calibration.
    2) a multitemporal method in which a 6-parameter splines function is
    optimised using the scipy optimize function to describe the monthly 
    dynamics of the ktc parameters
    
This module handles the data preparation and the export of the model runs. All 
input parameters need to be proprocessed using the 'WS_preprocess_data_layers.py'
module if they are not already processed.

@author: Francis Matthews fmatthews1381@gmail.com
"""

import sys
import os
import shutil
import pandas as pd
import WS_optimise_v3
import numpy as np
import pickle

#specify the path to the W/S executable on the system
cnws_path = 'C:\Workdir\Programs\WS_20062023_depo_limited\cn_ws\cn_ws\cn_ws'


slice_to_val_ts = True
#use 'VanOost2000', 'Dynamic_v1' or 'Dynamic_v5' 
#Dynamic_v1 is maybe best for splines method
TC_model = 'Dynamic_v1'
#give a list of pre-processed catchments to run
catchments = [9]
#An option to implement a train-test split - not used
train_test_split = False
#this parameter states that the timeseries will be resmapled to 15-days
resample = True
#here it can be specified whether a colibration is run first or, if already run, it can be read only
calibrate_ws = True
#the calibration scalar window increases the search bounds (by x val) if no acceptable range in found.
cal_window_scalar = 2
read_calibration = True
#specify whther to run the multitemporal splines optimisation
calibrate_splines = True
#specify the master directory for the outputs
master_in_folder  = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_preprocessing/WS_processed_EUSEDcollab'
#specify the name of the output folder
master_out_folder  = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_implementation/EUSEDcollab_results_v4'

#specify whether to run scenarios
run_scenarios = False
filter_sdr = False
filter_quantiles = False
filter_ssl = False
ssl_lower_limit = 0.1

splines_runs = []
results = {}
#initial calibration parameters for the internal calibration process
cal_params = {'KTcHigh_lower': 0.15625, 'KTcHigh_upper': 2.5, 'KTcLow_lower': 0.078125, 'KTcLow_upper': 1.25, 'steps': 20}

splines_array_in =  None #example input: [116.44745996,  43.28984593,  36.22289995,   0.47231912, 98.93378576,   7.57999032]

if run_scenarios == True:
    n_iterations = 100
elif filter_quantiles == True:
    n_iterations = 5
    filter_sdr = False    
else:
    n_iterations = 1
    filter_sdr = False
    
for ID_eused in catchments:
    for i in np.arange(n_iterations):   
        id_ = str(ID_eused)
        catchment_name = f'Catchment_id_{id_}'
        print('iteration number: ' + str(i))
        #specify the input and output folders
        in_folder = os.path.join(master_in_folder, f'WS_inputs_id_{id_}')
        md_all_p = os.path.join(master_in_folder, 'EUSEDcollab_data_repository/ALL_METADATA.csv')
        EUSEDcollab_path = os.path.join(master_in_folder, 'EUSEDcollab_data_repository/all_timeseries.pickle')
        out_folder = os.path.join(master_out_folder, f'Catchment_id_{id_}_' + TC_model)
        cal_res_folder = master_out_folder                          
        #get dictionary with the input file paths
        file_paths_p = os.path.join(in_folder, 'ws_file_paths.pickle')
        file_paths = pd.read_pickle(file_paths_p)
        
        
        file_paths['in_folder'] = in_folder
        file_paths['out_folder'] = out_folder
        WS_preprocess_params = file_paths['input_parameters'] 
        
        os.chdir(in_folder)
        
        if filter_sdr == True:
            res_file = os.path.join(cal_res_folder, 'Posterior_analysis_'+ id_ +'.pickle')
            res_slr = pd.read_pickle(res_file)
            res_slr = res_slr[['Event_index', 'SDR']]
            sdr_mean = res_slr['SDR'].mean()
            sdr_max = res_slr['SDR'].max()
        else:
            sdr_mean = 1
            sdr_max = 1
    
            
        n_ = 10
        fa = WS_preprocess_params['River flow acc threshold']
        fa_low = fa - (0.5 * fa)
        fa_high = fa + (0.2 * fa)
    
        
        WS_params = {}
        if run_scenarios == True:
            #here a the parameters for a random scenario are established
            #here the parameter limits are specified
            forest_con = np.linspace(10, 50, n_).astype(int)
            forest_te = np.linspace(50, 90, n_).astype(int)
            cropland_con = np.linspace(80, 100, n_).astype(int)
            cropland_te = np.linspace(0, 20, n_).astype(int)
            pasture_te = np.linspace(50, 90, n_).astype(int)
            fa_thresholds = np.linspace(fa_low, fa_high, n_).astype(int)
            sdr_thresholds = np.linspace(sdr_mean, sdr_max, n_)
    
            WS_params['ktc limit'] = 0.011
            #here a random parameter is selected
            WS_params['Parcel connectivity cropland'] = np.random.choice(cropland_con) #90
            WS_params['Parcel connectivity forest'] = np.random.choice(forest_con) #30
            WS_params['Parcel trapping efficiency cropland'] = np.random.choice(cropland_te) #0
            WS_params['Parcel trapping efficiency forest'] = np.random.choice(forest_te) #75
            WS_params['Parcel trapping efficiency pasture'] = np.random.choice(pasture_te) #75
            WS_params['LS correction'] = 1 # set LS cor to 0.3
            fa_threshold = np.random.choice(fa_thresholds)
            sdr_threshold = np.random.choice(sdr_thresholds)
            WS_preprocess_params['Flow acc channel threshold'] = fa_threshold
            WS_preprocess_params['SDR ratio threshold'] = sdr_threshold
            WS_preprocess_params['SSL quantile limit'] = 1
            WS_preprocess_params['Run type'] = 'Multi-scenario'
        elif filter_quantiles == True:
            quantiles = np.linspace(0.5, 1, n_iterations)
            quant = quantiles[i]
            WS_preprocess_params['SSL quantile limit'] = quant
            WS_params['ktc limit'] = 0.011
            WS_params['Parcel connectivity cropland'] = 90
            WS_params['Parcel connectivity forest'] = 30
            WS_params['Parcel trapping efficiency cropland'] = 0
            WS_params['Parcel trapping efficiency forest'] = 75
            WS_params['Parcel trapping efficiency pasture'] = 75
            WS_params['LS correction'] = 1 # set LS cor to 0.3
            fa_threshold = fa
            sdr_threshold = 'None'
            WS_preprocess_params['Flow acc channel threshold'] = fa_threshold
            WS_preprocess_params['SDR ratio threshold'] = sdr_threshold
            WS_preprocess_params['Run type'] = 'Multi-scenario'
        else:
            WS_params['ktc limit'] = 0.011
            WS_params['Parcel connectivity cropland'] = 95 #90
            WS_params['Parcel connectivity forest'] = 27 #30
            WS_params['Parcel trapping efficiency cropland'] = 0 #0
            WS_params['Parcel trapping efficiency forest'] = 90 #75
            WS_params['Parcel trapping efficiency pasture'] = 90 #75
            WS_params['LS correction'] = 1 # set LS cor to 0.3
            fa_threshold = 7500 #fa
            sdr_threshold = 'None'
            WS_preprocess_params['Flow acc channel threshold'] = fa_threshold
            WS_preprocess_params['SDR ratio threshold'] = sdr_threshold
            WS_preprocess_params['SSL quantile limit'] = 1
            WS_preprocess_params['Run type'] = 'Single-scenario'

        
        if filter_ssl == True:
            WS_preprocess_params['SSL lower limit'] = ssl_lower_limit
        else:
            WS_preprocess_params['SSL lower limit'] = 'None'
            
        #create a dictionary with pre-processing parameters that are relevant
        md_all = pd.read_csv(md_all_p)
        data_format = str(md_all[md_all['Catchment ID'] == ID_eused]['Data type'].values[0])
        
        #open EUSEDcollab and get the timeseries info
        EUSEDcollab_all = pd.read_pickle(EUSEDcollab_path)
        ts_val = EUSEDcollab_all[f'ID_{ID_eused}']
        
        #return a harmonised version of the catchment time series
        if resample == True:
            ts_val_h = format_catchment_ts(ts_val, data_format, resample = True, time_resolution= '15d')
            ts_val_h = ts_val_h[ts_val_h['SSL (kg event-1)'] > 0]
        else:
            ts_val_h = format_catchment_ts(ts_val, data_format, resample = False)
        
        ts_val_h['Event_datetime'] = ts_val_h.index
        #get the r-factor timeseries
        r_ts = file_paths['dynamic_layers_paths']['r_factor_ts']
        r_ts['Datetime'] = r_ts.index
        
        sim_vs_val_eval = {}
        #slice the simulation period to the validation period
        sim_vs_val_eval['Time period sliced to validation?'] = slice_to_val_ts
        if slice_to_val_ts == True:
            r_ts = slice_period_to_validation(r_ts, ts_val_h, buffer_period = '15d')
        
        #merge to include only those events
        '''
        Here the events are matched in time between simulated (EMO-5) events and 
        measured events in the catchment validation dataset. It is important to
        ensure the the sampling period is the same if they are resampled.
        ''' 
        #find all the runoff events that can match with a rainfall event
        r_ts_events_m = pd.merge_asof(ts_val_h, r_ts, right_index = True, left_index = True, 
                                    tolerance = pd.Timedelta('15d'))
        r_ts_events_m = r_ts_events_m[r_ts_events_m['Event_index'].notna()]
        r_ts_events_m = r_ts_events_m[r_ts_events_m['SSL (kg event-1)'].notna() & r_ts_events_m['SSL (kg event-1)'] > 0]
        r_ts_events_m.to_csv(os.path.join(cal_res_folder, id_ + '_event_information.csv'))
        
        r_ts_events_nm = r_ts[~r_ts['Event_index'].isin(r_ts_events_m['Event_index'])]
        
        if filter_sdr == True:
            r_ts_events_m = r_ts_events_m.merge(res_slr, how = 'left', on = 'Event_index')
            r_ts_events_m = r_ts_events_m[r_ts_events_m['SDR'] <= sdr_threshold].copy()
        if filter_quantiles == True:
            quantile_val = r_ts_events_m['SSL (t event-1)'].quantile(quant)
            r_ts_events_m = r_ts_events_m[r_ts_events_m['SSL (t event-1)'] <= quantile_val].copy()
        if filter_ssl == True:
            r_ts_events_m = r_ts_events_m[r_ts_events_m['SSL (t event-1)'] >= ssl_lower_limit].copy()
            

        #find out if missing events are poor EMO-5 simulations or non-erosive events
        
        '''
        evaluate the number of simulated and measured events that were matched
        '''
        sim_vs_val_eval['N input rainfall events'] = len(r_ts[r_ts['Event_index'].notna()])
        sim_vs_val_eval['N measured runoff/sed events'] = len(ts_val_h[ts_val_h['SSL (kg event-1)'].notna() & ts_val_h['SSL (kg event-1)'] > 0])
        sim_vs_val_eval['N matched events'] = len(r_ts_events_m)
        sim_vs_val_eval['% match'] = sim_vs_val_eval['N matched events']/sim_vs_val_eval['N input rainfall events'] *100
        sim_vs_val_eval['SSL sum (t whole ts)'] = ts_val_h['SSL (t event-1)'].sum()
        sim_vs_val_eval['SSL sum (t matched events/periods)'] = r_ts_events_m['SSL (t event-1)'].sum()
    
        
        WS_calibrate = WS_optimise_v3.Optimise_dynamic_WS(cnws_path, file_paths, catchment_name, r_ts_events_m, 
                                                          TC_model, WS_params, WS_preprocess_params)
        
        #update the calibration ranges to those of the last model run
        #prevents the range needing to be found each time
        if cal_params is not None:
            WS_calibrate.ktc_cal_p = cal_params
        
        #run WaTEM-SEDEM in calibration mode and collect the results into a dataframe
        if calibrate_ws == True:
            #if the out_folder exists, delete it to avoid the accumulation of files
            if os.path.exists(file_paths['out_folder']):
                print('Clearing content of '+ file_paths['out_folder'] +' outfolder for re-writing')
                #sometimes file is getting stuck and not clearing
                try:
                    shutil.rmtree(file_paths['out_folder'])
                except:
                    print('Changing outfolder name to avoid error. Iteration_number = ' + str(i))
                    file_paths['out_folder'] = file_paths['out_folder'] + '_backup_' + str(i)
                    #update file paths with changed name
                    WS_calibrate.file_paths = file_paths
                    
            try:
                cal_params = WS_calibrate.find_ktc_range(capture_pcnt = 40, n_events = 20, fa_threshold = fa_threshold,
                                                         cal_window_scalar = cal_window_scalar)
                WS_calibrate.run_ws_calibration(n_cal_steps = 20, n_events = 'All', fa_threshold = fa_threshold)
            except:
                print('Model run failed to find a calibration range.')
                continue
        
            
        if read_calibration == True:
            try:
                cal_all = WS_calibrate.process_calibration(plot_ts = False)
                WS_calibrate.visualise_calibration(cal_all['All event simulations'])
            except:
                print('Error processing calibration')
                continue
            
        
        if calibrate_ws == True:
            WS_calibrate.export_calibration_results(cal_res_folder, extention = catchment_name)
        
        if train_test_split == True:    
            #define a dataframe defining the events to simulate
            #NOTE: here only merged events are run for the purpose of comparison
            ts_cal, ts_val = split_timeseries(r_ts_events_m, method = 'time') 
        else:
            ts_cal = r_ts_events_m 
            ts_val = r_ts_events_m
            
        cal_events = list(ts_cal['Event_index'])
        val_events = list(ts_val['Event_index'])
    
        
        if calibrate_splines == True:
            #run the dynamic calibration routine - this only works with a matched sim-obs series
            WS_cal_splines = WS_calibrate.run_dynamic_calibration(ts_cal, dynamic_channels = False, 
                                                                      max_fa = 20000, min_fa = 800)
            WS_calibrate.export_calibration_results(cal_res_folder, splines = True)
        
    
        
        if i == n_iterations - 1:
            WS_predict = WS_optimise_v3.run_ws_prediction(cnws_path, file_paths, catchment_name, TC_model, WS_params = WS_params)
            WS_results = WS_predict.run_ws(WS_calibrate.ktc_low_l, WS_calibrate.ktc_high_l, event_indexes = val_events)
            WS_predict.visualise_outputs(r_ts, calibration = True, calibration_events = cal_events)
        
        if calibrate_splines == True:
            splines_array = WS_calibrate.splines_parameters_transformed
        else:
            splines_array = splines_array_in
            
        try:
            WS_val_splines = WS_calibrate.run_dynamic_calibration(ts_val, dynamic_channels = False, 
                                                                  run_one = True, v_fitted = splines_array)
        except:
            splines_runs.append(TC_model + ' failed splines run')
                
        results[str(i)] = cal_all['Statistical eval_lumped']
        results['WS preprocess params ' + str(i)] =  WS_preprocess_params
        results['WS params ' + str(i)] =  WS_params
        
        if run_scenarios == True:
            pickle_p = os.path.join(cal_res_folder, f'All_WS_scenarios_{TC_model}_ID_{id_ }.pickle')
        else:
            pickle_p = os.path.join(cal_res_folder, f'WS_single_scenario_{TC_model}_ID_{id_ }.pickle')
        pickle.dump(results, open(pickle_p, 'wb'))



 

