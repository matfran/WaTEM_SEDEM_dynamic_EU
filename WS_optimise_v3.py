# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:31:17 2023

@author: u0133999
"""
import pandas as pd
import sys
import os
from WS_dynamic_functions import run_WS_dynamic, collect_WS_output, modify_channel_extent, control_splines, control_params, export_splines
from WS_post_processing import aggregate_ws_grids, plot_event_ws, compare_event_distributions, merge_sim_obs
from WS_cal_val import calibrate_lumped_catch_precip, calibrate_event_catch_precip, calibrate_monthly_mean_diff, get_metrics
from WS_cal_val import evaluate_maximum_efficiency, plot_calibration, validate_matched_events, plot_sim_obs_calibrated
from WS_cal_val import get_mean_monthly_avg
import patsy
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import shutil
import pickle
import re



class run_ws_prediction:
    """
    A class to run and visualise WaTEM-SEDEM in a predictive mode. The class needs
    to be initialised with a dictionary of all appropriately formulated file paths 
    and a transport capacity formula.
    """
    def __init__(self, cnws_path, file_paths, catchment_name, TC_model, RE_name,
                 outfolder_ext = 'run', WS_params = None, WS_preprocess_params = None):
        self.file_paths = file_paths
        self.TC_model = TC_model 
        self.WS_params = WS_params
        self.WS_preprocess_params = WS_preprocess_params
        self.cnws_path = cnws_path
        self.catchment_name = catchment_name
        self.RE_name = RE_name
        
        self.file_paths['out_folder'] = self.file_paths['out_folder'] + '_' + outfolder_ext
        
        if os.path.exists(self.file_paths['out_folder']):
            print('Clearing content of ' + self.file_paths['out_folder'] + ' for re-writing')
            try:
                shutil.rmtree(self.file_paths['out_folder'])
            except:
                #avoid system errors if files get stuck
                num = 1
                if os.path.exists(self.file_paths['out_folder'] + f'_backup_{num}'):
                    nums = re.findall(r'\d+', self.file_paths['out_folder'])
                    num = int(nums[-1]) + 1
                self.file_paths['out_folder'] = self.file_paths['out_folder'] + f'_backup_{num}'

        
        
        
    def run_ws(self, ktc_low, ktc_high, n_events = 'All', event_indexes = None, print_sdr = False): 
        '''
        Run a number of events (n_steps) in predictive mode with a defined 
        parameter set (ktc_low and ktc_high)

        Parameters
        ----------
        ktc_low : FLOAT
            WS calibration parameter.
        ktc_high : STRING
            WS calibration parameter.
        n_events : INT
            The number of events to run from the dataframe of events

        Returns
        -------
        data_exists : BOOLEAN
            A boolean stating if sheet has data
        ws_results_run : DATAFRAME
            A dataframe with the WS predictions from the desired events

        '''
            
        run_WS_dynamic(self.file_paths, self.cnws_path, calibrate = False, n_iterations = n_events, 
                       TC_model = self.TC_model, ktc_low = ktc_low, ktc_high = ktc_high,
                       event_indexes = event_indexes, WS_params = self.WS_params, RE_name = self.RE_name)
        
        
        ws_results_run = collect_WS_output(self.file_paths['out_folder'], calibration = False)
        
        if print_sdr == True:
            sdr = -1 * (ws_results_run[ 'Sediment export via river (kg)'].sum()/ws_results_run[ 'Total gross erosion (kg)'].sum())
            print('SDR = ' + str(np.around(sdr * 1,2)))
        return ws_results_run
    
    def clear_folder(self):
        try:
            if os.path.exists(self.file_paths['out_folder']):
                shutil.rmtree(self.file_paths['out_folder'])
        except:        
            if os.path.exists(self.file_paths['out_folder']):
                print('Clearing content of ' + self.file_paths['out_folder'] + ' for re-writing')
                try:
                    shutil.rmtree(self.file_paths['out_folder'])
                except:
                    #avoid system errors if files get stuck
                    num = 1
                    if os.path.exists(self.file_paths['out_folder'] + f'_backup_{num}'):
                        nums = re.findall(r'\d+', self.file_paths['out_folder'])
                        num = int(nums[-1]) + 1
                    self.file_paths['out_folder'] = self.file_paths['out_folder'] + f'_backup_{num}'
        


    def visualise_outputs(self, r_ts, calibration = False, calibration_events = None, 
                          export = True):
        '''
        Visualise the WS predictions.

        Parameters
        ----------
        r_ts : DATAFRAME
            A dataframe with the event information
        Returns
        -------
        '''
        
        if calibration == True:
            r_ts = r_ts[r_ts['Event_index'].isin(calibration_events)]
        
        plot_event_ws(self.file_paths, r_ts, raster_name = 'Capacity', 
                      log_y = True, export = export)
        plot_event_ws(self.file_paths, r_ts, raster_name = 'CN', 
                      log_y = True, export = export)
        plot_event_ws(self.file_paths, r_ts, raster_name = 'Cfactor', export = export)
        plot_event_ws(self.file_paths, r_ts, raster_name = 'RUSLE', export = export)        

        rusle_sum = aggregate_ws_grids(self.file_paths, 'RUSLE', cmap = 'GnBu', rusle = True)
        capacity_sum = aggregate_ws_grids(self.file_paths, 'Capacity')
        sed_ex_sum = aggregate_ws_grids(self.file_paths, 'SediExport_kg')
        sed_in_sum = aggregate_ws_grids(self.file_paths, 'SediIn_kg')
        uparea_sum = aggregate_ws_grids(self.file_paths, 'UPAREA')
        ws_eros_sum = aggregate_ws_grids(self.file_paths, 'WATEREROS (mm per gridcel)',
                                         export = export)
        ws_eros_binary = aggregate_ws_grids(self.file_paths, 'WATEREROS (mm per gridcel)', binary = True,
                                            export = export)

    
class Optimise_dynamic_WS:
    """
    A class to calibrate WaTEM-SEDEM starting from an unknown parameter range. 
    The class needs to initialised with a dictionary containing all file paths
    for WS and a selected transport capacity model. A dataframe also needs 
    to be provided with a merged set of rainfall erosivity events and catchment 
    sediment yield data for the catchment of interest.
    """    
    
    def __init__(self, cnws_path, file_paths, catchment_name, r_ts_events_m, TC_model, RE_name,
                 WS_params = None, WS_preprocess_params = None):
        #here a set of calibration parameter ranges is given for each TC formula
        ktc_cal_p = {}
        if TC_model in ['VanOost2000']:
            ktc_cal_p["KTcHigh_lower"] = 10.
            ktc_cal_p["KTcHigh_upper"] = 40.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10
        elif TC_model in ['VanOost_v2']:
            ktc_cal_p["KTcHigh_lower"] = 5.
            ktc_cal_p["KTcHigh_upper"] = 40.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10  
        elif TC_model in ['Dynamic_v1']:
            ktc_cal_p["KTcHigh_lower"] = 10
            ktc_cal_p["KTcHigh_upper"] = 40.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10 
        elif TC_model in ['Dynamic_v4', 'Dynamic_v5']:
            ktc_cal_p["KTcHigh_lower"] = 50.
            ktc_cal_p["KTcHigh_upper"] = 100.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10   
        elif TC_model in ['Dynamic_v2', 'Dynamic_v3']:
            sys.exit('select valid TC model - not re-scaled')
        elif TC_model in ['Verstraeten2007']:
            ktc_cal_p["KTcHigh_lower"] = 1.
            ktc_cal_p["KTcHigh_upper"] = 10.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10 
        elif TC_model in ['Dynamic_v6']:
            ktc_cal_p["KTcHigh_lower"] = 20.
            ktc_cal_p["KTcHigh_upper"] = 100.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10 
        elif TC_model in ['Verstraeten_v2']:
            ktc_cal_p["KTcHigh_lower"] = 20.
            ktc_cal_p["KTcHigh_upper"] = 100.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10  
        elif TC_model in ['VanOost_v3']:
            ktc_cal_p["KTcHigh_lower"] = 20.
            ktc_cal_p["KTcHigh_upper"] = 100.
            ktc_cal_p["KTcLow_lower"] = ktc_cal_p["KTcHigh_lower"]/2
            ktc_cal_p["KTcLow_upper"] = ktc_cal_p["KTcHigh_upper"]/2
            ktc_cal_p["steps"] = 10             
        else:
            sys.exit('select valid TC model')

            
        self.ktc_cal_p = ktc_cal_p
        self.file_paths = file_paths 
        self.r_ts_events_m = r_ts_events_m.copy()
        self.TC_model = TC_model
        self.WS_params = WS_params
        self.WS_preprocess_params = WS_preprocess_params
        self.cnws_path = cnws_path
        self.catchment_name = catchment_name
        self.RE_name = RE_name
        
    def update_cal_params(self, previous_prediction, scalar_factor = 2, n_steps = 10):
        '''
        Update the calibration parameter range to find a suitable range to 
        run a calibration procedure. The calibration parameter range
        associated with the object is updated based on this. A previous_prediction
        is required to evaluate how the parameter range should change.

        Parameters
        ----------
        previous_prediction : STRING
            'Underestimation', 'Overestimation', or 'In_range'
        Returns
        -------
        '''
        
        #if underestimation, shift window
        if previous_prediction == 'Underestimation':
            self.ktc_cal_p["KTcHigh_upper"] = self.ktc_cal_p["KTcHigh_upper"]*scalar_factor
            self.ktc_cal_p["KTcHigh_lower"] = self.ktc_cal_p["KTcHigh_lower"]*scalar_factor
        #if overestimation, shift window
        elif previous_prediction == 'Overestimation':
            self.ktc_cal_p["KTcHigh_upper"] = self.ktc_cal_p["KTcHigh_upper"]/scalar_factor
            self.ktc_cal_p["KTcHigh_lower"] = self.ktc_cal_p["KTcHigh_lower"]/scalar_factor
        #if the lump sum of sediment within the calibration matched the observed load, increase the window
        else:
            self.ktc_cal_p["KTcHigh_upper"] = self.ktc_cal_p["KTcHigh_upper"]*scalar_factor
            self.ktc_cal_p["KTcHigh_lower"] = self.ktc_cal_p["KTcHigh_lower"]/scalar_factor
            
        #update the lower boundary to always half of the upper
        self.ktc_cal_p["KTcLow_lower"] = self.ktc_cal_p["KTcHigh_lower"]/2
        self.ktc_cal_p["KTcLow_upper"] = self.ktc_cal_p["KTcHigh_upper"]/2
        self.ktc_cal_p["steps"] = self.ktc_cal_p["steps"] = n_steps
        
        
    def update_for_simulation(self, n_steps):
        '''
        Update the calibration parameters to run a full calibration procedure.

        Parameters
        ----------
        n_steps : INT
            n_steps to run per event in the calibration procedure
        Returns
        -------
        '''
        self.ktc_cal_p["steps"] = self.ktc_cal_p["steps"] = n_steps
        
    def evaluate_calibration(self, ws_obs_ts):
        '''
        Evaluate the calibration parameter range guess. This requires a merged set of 
        WS predictions and catchment sediment yield observations and an evaluation 
        is made on the % of observations captured within the calibration parameter. 
        New attributes are added to the class, describing the % of observations 
        

        Parameters
        ----------
        ws_obs_ts : DATAFRAME
            A dataframe with merged WaTEM-SEDEM simulations and catchment observations.
        Returns
        -------
        '''
        #only consider events that match with an observation - we assume that the measured 
        #data captures all of the sediment
        ws_obs_ts = ws_obs_ts[ws_obs_ts['SSL (t event-1)'].notna()]
        #group event sediment yield 
        cols = ['Event_index', 'ktc_low', 'ktc_high', 'tot_erosion', 'tot_sedimentation', 'sed_noriver', 'sed_buffer', 
                'sed_openwater', 'SSL-WS (t event-1)', 'SSL (t event-1)']
        
        #we use the 5th and 95th percentiles of the distribution of realisations
        pred_max = ws_obs_ts.groupby('Event_index', 
                                     as_index = False)[cols].quantile(.99).rename(columns = {'SSL-WS (t event-1)': 'SSL-WS max (t event-1)'})
        
        pred_min = ws_obs_ts.groupby('Event_index', 
                                     as_index = False)[cols].min(.01).rename(columns = {'SSL-WS (t event-1)': 'SSL-WS min (t event-1)'})
        #Do not consider zero predictions as valid
        pred_min['SSL-WS min (t event-1)'] = np.where(pred_min['SSL-WS min (t event-1)'] > 0, pred_min['SSL-WS min (t event-1)'], np.nan)
       
        eval_combined = pred_max.merge(pred_min[['Event_index', 'SSL-WS min (t event-1)']], on = 'Event_index', 
                                       how = 'left')
        
        mask = (eval_combined['SSL-WS max (t event-1)'] >= eval_combined['SSL (t event-1)']) & (eval_combined['SSL-WS min (t event-1)'] <= eval_combined['SSL (t event-1)'])
        eval_combined['In_envelope'] = mask.astype(int)
        
        high_sum = eval_combined['SSL-WS max (t event-1)'].sum()
        low_sum = eval_combined['SSL-WS min (t event-1)'].sum()
        obs_sum = eval_combined['SSL (t event-1)'].sum()
        
        
        #evaluate if the ktc envelope should shift up or down
        if low_sum > obs_sum:
            sum_load = 'Overestimation'
        elif high_sum < obs_sum:
            sum_load = 'Underestimation'
        else:
            sum_load = 'In_range'

        self.pcnt_in_envelope = (eval_combined['In_envelope'].sum()/len(eval_combined['In_envelope'])) * 100
        self.prev_pred = sum_load

    def find_ktc_range(self, capture_pcnt, n_events = 'All', cal_window_scalar = 2):
        #get an initial first-guess parameter set
        pcnt_in_bounds = 0
        count = 0
        
            
        #find a suitable parameter range for ktc by iteratively increasing the range
        #until a critical percentage of observations sit within the range of simulations
        while pcnt_in_bounds < capture_pcnt:
            if count >= 1:
                self.update_cal_params(self.prev_pred, scalar_factor = cal_window_scalar)
                print(self.ktc_cal_p)
            
            run_WS_dynamic(self.file_paths, self.cnws_path, calibrate = True, n_iterations = n_events, 
                           TC_model = self.TC_model, ktc_cal_p = self.ktc_cal_p, 
                           WS_params = self.WS_params, RE_name = self.RE_name)
            ws_first_guess = collect_WS_output(self.file_paths['out_folder'], calibration = True)
            ws_obs_ts_guess = merge_sim_obs(ws_first_guess, self.r_ts_events_m, calibration = True)
            self.evaluate_calibration(ws_obs_ts_guess)
            pcnt_in_bounds = self.pcnt_in_envelope
            
            if count > 6:
                self.parameter_range_found = False
                sys.exit('No ktc range found')
                break
            print(pcnt_in_bounds)
            print(self.prev_pred)
            count = count + 1
            
        self.parameter_range_found = True 
        print('Parameter range found')
            
        return self.ktc_cal_p

    
    def run_ws_calibration(self, n_cal_steps, n_events = 'All'):
        self.update_for_simulation(n_steps = n_cal_steps)
        if n_events == 'All':
            n_events = None
        
            
        #pass a list of only the calibration events to run
        event_indexes = list(self.r_ts_events_m['Event_index'])
        
        run_WS_dynamic(self.file_paths, self.cnws_path, calibrate = True, event_indexes = event_indexes, 
                       TC_model = self.TC_model, ktc_cal_p = self.ktc_cal_p,
                       WS_params = self.WS_params, RE_name = self.RE_name)
        
        
        
    def process_calibration(self, plot_ts = True):
        
        if not os.path.exists(self.file_paths['out_folder']):
            print('WS needs to be run before the output can be processed')
            sys.exit()
        
        ws_results_cal = collect_WS_output(self.file_paths['out_folder'], calibration = True)
        ws_obs_ts_cal = merge_sim_obs(ws_results_cal, self.r_ts_events_m, calibration = True)
        
        cal_all = {}
        cal_all['All event simulations'] = ws_obs_ts_cal
        cal_all['WS results'] = ws_results_cal
        
        
        lumped_calibration = calibrate_lumped_catch_precip(ws_obs_ts_cal, plot_sdr = plot_ts,
                                                           out_path = self.file_paths['out_folder'])
        self.lumped_calibration = lumped_calibration
        event_calibration = calibrate_event_catch_precip(ws_obs_ts_cal, objective_function = 'KGE efficiency_events')
        self.event_calibration = event_calibration
        monthly_mean_calibration = calibrate_monthly_mean_diff(ws_obs_ts_cal)
        self.monthly_mean_calibration = monthly_mean_calibration
        
        best_poss_pred = evaluate_maximum_efficiency(ws_obs_ts_cal)
        min_abs_error_total = best_poss_pred['Min pred-obs_abs (t event-1)'].sum()
        mean_pcnt_error_best = best_poss_pred['% error'].mean()
        print('The lowest absolute difference (tonnes) for all events, considering all ktc combinations, is: ', str(min_abs_error_total))
        print('The lowest mean % error for all events, considering all ktc combinations, is: ', str(mean_pcnt_error_best))
        

        self.ktc_low_l = lumped_calibration['ktc_low'].iloc[0]
        self.ktc_high_l = lumped_calibration['ktc_high'].iloc[0]
        #self.min_diff_l = lumped_calibration[''].iloc[0]


        self.ktc_low_e = event_calibration['ktc_low'].iloc[0]
        self.ktc_high_e = event_calibration['ktc_high'].iloc[0]
        #self.min_diff_e = event_calibration[''].iloc[0]
        
        self.ktc_low_m = monthly_mean_calibration['ktc_low'].iloc[0]
        self.ktc_high_m = monthly_mean_calibration['ktc_high'].iloc[0]        
        
        if plot_ts == True:
            plot_sim_obs_calibrated(ws_obs_ts_cal, self.ktc_low_l, self.ktc_high_l, 
                                    title = 'Lumped calibration', out_path = self.file_paths['out_folder'])
            plot_sim_obs_calibrated(ws_obs_ts_cal, self.ktc_low_e, self.ktc_high_e, 
                                    title = 'Event-wise calibration (absolute error)', out_path = self.file_paths['out_folder'])
            plot_sim_obs_calibrated(ws_obs_ts_cal, self.ktc_low_m, self.ktc_high_m, 
                                    title = 'Mean monthly calibration (absolute error)', out_path = self.file_paths['out_folder'])
            

        cal_all['Lumped calibration'] = lumped_calibration
        cal_all['Event-wise calibration'] = event_calibration
        cal_all['Monthly mean calibration'] = monthly_mean_calibration
        
        
        stat_eval_l = validate_matched_events(ws_obs_ts_cal, self.catchment_name + '_lumped', self.ktc_low_l, self.ktc_high_l)
        stat_eval_e = validate_matched_events(ws_obs_ts_cal, self.catchment_name + '_event', self.ktc_low_e, self.ktc_high_e)
        stat_eval_m = validate_matched_events(ws_obs_ts_cal, self.catchment_name + '_mean_monthly', self.ktc_low_m, self.ktc_high_m)

        
        self.statistical_eval = stat_eval_l
        self.statistical_eval_e = stat_eval_e
        self.statistical_eval_m = stat_eval_m
        
        cal_all['Statistical eval_lumped'] = stat_eval_l
        cal_all['Statistical eval_event rmse'] = stat_eval_e
        cal_all['Statistical eval_mean monthly rmse'] = stat_eval_m
        
        pickle_p = os.path.join(self.file_paths['out_folder'], 'All_calibration_output.pickle')
        pickle.dump(cal_all, open(pickle_p, 'wb'))
                
        return cal_all  

    
    def run_dynamic_calibration(self, r_ts_events_m, run_one = False, v_fitted = None):
        

            
        #copy so that changes aren't implemented on original df
        r_ts_events_m_copy = r_ts_events_m.copy()
        if run_one == True:
            extention = 'dynamic_splines_run'            
        else:
            extention = 'dynamic_splines_calibration'
            
        ws_predictor = run_ws_prediction(self.cnws_path, self.file_paths, self.catchment_name, self.TC_model, 
                                         self.RE_name, extention, self.WS_params, self.WS_preprocess_params)
        

        months = range(1, 13)
        dsm_mo = patsy.dmatrix("bs(x, df=5, degree=2, include_intercept=True) - 1", {"x": months})
        
        #THIS IS THE INPUT FROM THE WATEM-SEDEM CALIBRATION - WE CAN USE IT TO 
        #SUGGEST AN INITAL CALIBRATION RANGE IF IT'S USEFUL

        self.dynam_it_count = 0
        self.mape_record = []
        self.total_diff_record = []
        self.rmse_record = []
        
        
        def fit_splines(v):
            
            '''
            #if no net increase is has been made, don't continue the full number of iterations
            if self.dynam_it_count >100:
                diff_50 = sum(np.diff(self.mape_record[-50:]))
                if diff_50 < 0.5:
                    return np.nan
            '''
            #clear the output folder if it exists
            ws_predictor.clear_folder()
            
            v = control_splines(v)
            

            #b_low = np.array([v[0], v[1], v[2], v[3], v[4]])
            #b_low is a calibrated ration of b_high
            b_high = np.array([v[0], v[1], v[2], v[3], v[4]])
            b_low = b_high /v[5]
            
            
            y_obs = r_ts_events_m_copy['SSL (t event-1)'].values
            y_pred = []
            
            #Loop through each event index value
            for i in r_ts_events_m_copy['Event_index'].values:
                idx = i
                line = r_ts_events_m_copy[r_ts_events_m_copy['Event_index'] == i]
                rf_id = int(idx)
                
                nd = {'x' : line['Month']}
                #evaluating sline basis functions
                new_dsm_mo = patsy.build_design_matrices([dsm_mo.design_info], nd)[0]
                
                #there are currently nan values in the matrix
                try:
                    ktc_low = np.dot(new_dsm_mo, b_low)[0]
                    ktc_high = np.dot(new_dsm_mo, b_high)[0]
                except:
                    return np.nan
                
                try:
                    ws_results = ws_predictor.run_ws(ktc_low, ktc_high, event_indexes = [idx])
                except:
                    ws_results = ws_predictor.run_ws(ktc_low, ktc_high, event_indexes = [idx])
                    
                #the results will will iteratively include each new model run in the output folder
                ws_results_e = ws_results[ws_results['Event_index'] == idx]
                
                y_pred.append(float(ws_results_e['SSL-WS (t event-1)']))
            
            r_ts_events_m_copy['SSL-WS (t event-1)'] = y_pred
            #self.splines_ws_obs_ts = r_ts_events_m_copy
            
            monthly_mean = get_mean_monthly_avg(r_ts_events_m_copy)
            
            #MONTHLY MEAN IS THE SAME AS CALIBRATION SERIES
            
            y_obs = monthly_mean['SSL_avg (t month-1)'].values
            y_sim = monthly_mean['SSL-WS_avg (t month-1)'].values
            
            global eval_splines
            eval_splines = get_metrics(y_obs, y_sim, name = 'Splines monthly mean')
            
            if run_one == False:
                if self.dynam_it_count == 0 or self.dynam_it_count % 10 == 0:
                    f_name = 'Splines_nochannels__' + self.catchment_name + '_' + self.TC_model + '.csv'
                        
                    folder = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_implementation/EUSEDcollab_results'
                    path = os.path.join(folder, f_name)
                    df = pd.DataFrame({'Catchment': [self.catchment_name], 'TC_model': [self.TC_model], 
                                       'Iteration n':[self.dynam_it_count], 'v_transformed': [v],
                                       'eval_monthly': [eval_splines]})
                    export_splines(df, path)
            
            mae = eval_splines['MAE']
            mape = eval_splines['MAPE']
            rmse = eval_splines['RMSE']
            total_diff = eval_splines['Total diff']
            
            print('v = ' + str(np.around(v,2)))
            print('rmse= ' + str(np.around(rmse, 2)))
            print('mae= ' + str(np.around(mae, 2)))
            print('iteration n: ' + str(self.dynam_it_count))
            print('total diff (t): ' + str(total_diff))
            self.dynam_it_count = self.dynam_it_count + 1
            self.mape_record.append(mape)
            self.total_diff_record.append(total_diff)
            self.rmse_record.append(rmse)
            self.eval_splines_monthly = eval_splines
            self.splines_parameters_transformed = v
            
            
            
            return rmse
        
        
        #THE MATRIX NEEDS TO BE INITIALISED BASED ON THE KNOWN KTC BOUNDARIES
        #HERE THE LUMPED CALIBRATION IS USED TO SET THE FIRST GUESS
        vl_1 = self.ktc_low_m
        vl_5 = self.ktc_low_m
        step = (vl_5 - vl_1)/5 
        vl_2 = vl_1 + step
        vl_3 = vl_2 + step
        vl_4 = vl_3 + step 
        
        vh_1 = self.ktc_high_m
        vh_5 = self.ktc_high_m
        step = (vh_5 - vh_1)/5 
        vh_2 = vh_1 + step
        vh_3 = vh_2 + step
        vh_4 = vh_3 + step 
        
        vr_0 = vh_1/vl_1
        
        vc = self.file_paths['input_parameters']['River flow acc threshold']

        v0 = [vh_1, vh_1, vh_1, vh_1, vh_1,
              vr_0]
        
        
        
        if run_one == True:
            print('Beginning pre-defined splines run')

            v = v_fitted
            v = np.log(v)
            res = fit_splines(v)
        else:
            print('Beginning splines optimisation')
            print(v0)
            #added line to prevent negative values
            v0 = np.log(v0)
            res = scipy.optimize.minimize(fit_splines, v0, method='Nelder-Mead', options={'maxiter': 100}) # This can not handle bounds
            # Below, plot
            v = res['x']
            self.splines_parameters = v

            
        v_f = self.splines_parameters_transformed
        nd = {"x": np.linspace(1.0, 12.0, num = 100)}
        new_dsm_mo = patsy.build_design_matrices([dsm_mo.design_info], nd)[0]
        b_high = np.array([v_f[0], v_f[1], v_f[2], v_f[3], v_f[4]])
        ktc_high = np.dot(new_dsm_mo, b_high)
        
            


        
        fig, ax = plt.subplots(figsize = (12,8))
        plt.plot(nd['x'], ktc_high, color = 'blue') # ktc_high
        ax.set_xlabel('Month')
        ax.set_ylabel('Ktc_high')
        plt.savefig(os.path.join(ws_predictor.file_paths['out_folder'], 'ktc_high.png'))
        plt.show()
        
        df_plot_data = pd.DataFrame({'Month':nd['x'], 'ktc_high':ktc_high})
        self.df_splines_plot_data = df_plot_data

        
        ws_results_run = collect_WS_output(ws_predictor.file_paths['out_folder'], calibration = False)
        
        sdr = -1 * (ws_results_run[ 'Sediment export via river (kg)'].sum()/ws_results_run[ 'Total gross erosion (kg)'].sum())
        print('SDR = ' + str(np.around(sdr * 1,2)))
        
        ws_obs_ts = merge_sim_obs(ws_results_run, r_ts_events_m, calibration = False)
        
        
        y_obs = ws_obs_ts['SSL (t event-1)'].values
        y_sim = ws_obs_ts['SSL-WS (t event-1)'].values
        self.eval_splines_events = get_metrics(y_obs, y_sim, name = 'Splines event evaluation')
        
        plot_sim_obs_calibrated(ws_obs_ts, title = 'Splines calibration')
        
            
        #WS_predictor.visualise_outputs(r_ts, r_ts_events_m)
        
        all_results = {}
        all_results['ws ts'] = ws_results_run
        all_results['ws obs ts'] = ws_obs_ts
        all_results['eval monthly mean'] = eval_splines
        all_results['eval events'] = self.eval_splines_events
        all_results['spline params original'] = v
        all_results['spline params transformed'] = self.splines_parameters_transformed
        all_results['spline figure data'] = self.df_splines_plot_data
        all_results['SDR'] = sdr
        all_results['x'] = res.x
        
        cal_events = list(r_ts_events_m['Event_index'])
        ws_predictor.visualise_outputs(r_ts_events_m, calibration = True, calibration_events = cal_events)
        
        if run_one == True:
            pickle_p = os.path.join(ws_predictor.file_paths['out_folder'], 'Spline_run.pickle')
            pickle.dump(all_results, open(pickle_p, 'wb'))
        else:
            pickle_p = os.path.join(ws_predictor.file_paths['out_folder'], 'Spline_calibration.pickle')
            pickle.dump(all_results, open(pickle_p, 'wb'))
            
        return all_results
    
    def run_basic_calibration(self, v0, r_ts_events_m):
            
        #copy so that changes aren't implemented on original df
        r_ts_events_m_copy = r_ts_events_m.copy()
        
        extention = 'basic_calibration'
            
        ws_predictor = run_ws_prediction(self.cnws_path, self.file_paths, self.catchment_name, self.TC_model, 
                                         self.RE_name, extention, self.WS_params, self.WS_preprocess_params)
        
        
        self.dynam_it_count = 0
        self.mape_record = []
        self.total_diff_record = []
        self.rmse_record = []
        self.kge_record = []
        
        
        def fit(v):
            
            '''
            #if no net increase is has been made, don't continue the full number of iterations
            if self.dynam_it_count >100:
                diff_50 = sum(np.diff(self.mape_record[-50:]))
                if diff_50 < 0.5:
                    return np.nan
            '''
            #clear the output folder if it exists
            ws_predictor.clear_folder()
            
            
            v = control_params(v)
            
            
            ktc_high = v[0]
            ktc_low = v[0] / v[2]
            fa = v[1]
            
            modify_channel_extent(ws_predictor.file_paths, fa)
            
            
            y_obs = r_ts_events_m_copy['SSL (t event-1)'].values
            y_pred = []
            
            #Loop through each event index value
            for i in r_ts_events_m_copy['Event_index'].values:
                idx = i
                line = r_ts_events_m_copy[r_ts_events_m_copy['Event_index'] == i]
                rf_id = int(idx)
                
                try:
                    ws_results = ws_predictor.run_ws(ktc_low, ktc_high, event_indexes = [idx])
                except:
                    ws_results = ws_predictor.run_ws(ktc_low, ktc_high, event_indexes = [idx])
                    
                #the results will will iteratively include each new model run in the output folder
                ws_results_e = ws_results[ws_results['Event_index'] == idx]
                
                y_pred.append(float(ws_results_e['SSL-WS (t event-1)']))
            
            r_ts_events_m_copy['SSL-WS (t event-1)'] = y_pred
            #self.splines_ws_obs_ts = r_ts_events_m_copy
            
            global eval_
            eval_ = get_metrics(y_obs, y_pred, name = 'Splines monthly mean')
            
            
            mae = eval_['MAE']
            mape = eval_['MAPE']
            rmse = eval_['RMSE']
            total_diff = eval_['Total diff']
            kge = eval_['KGE efficiency']
            
            print('v = ' + str(np.around(v,2)))
            print('rmse= ' + str(np.around(rmse, 2)))
            print('mae= ' + str(np.around(mae, 2)))
            print('iteration n: ' + str(self.dynam_it_count))
            print('total diff (t): ' + str(total_diff))
            print('KGE: ' + str(kge))
            print('FA: ' + str(fa))
            self.dynam_it_count = self.dynam_it_count + 1
            self.mape_record.append(mape)
            self.total_diff_record.append(total_diff)
            self.rmse_record.append(rmse)
            self.kge_record.append(kge)     
            
            
            return kge * -1
        
                
        v0 = np.log(v0)

        res = scipy.optimize.minimize(fit, v0, method='Nelder-Mead', options={'maxiter': 50}) # This can not handle bounds

        
        ws_results_run = collect_WS_output(ws_predictor.file_paths['out_folder'], calibration = False)
        
        sdr = -1 * (ws_results_run[ 'Sediment export via river (kg)'].sum()/ws_results_run[ 'Total gross erosion (kg)'].sum())
        print('SDR = ' + str(np.around(sdr * 1,2)))
        
        ws_obs_ts = merge_sim_obs(ws_results_run, r_ts_events_m, calibration = False)
        
        #WS_predictor.visualise_outputs(r_ts, r_ts_events_m)
        
        all_results = {}
        all_results['ws ts'] = ws_results_run
        all_results['ws obs ts'] = ws_obs_ts
        all_results['SDR'] = sdr
        all_results['eval'] = eval_
        all_results['x'] = res.x
        
        cal_events = list(r_ts_events_m['Event_index'])
        ws_predictor.visualise_outputs(r_ts_events_m, calibration = True, calibration_events = cal_events)

        pickle_p = os.path.join(ws_predictor.file_paths['out_folder'], 'Basic_calibration.pickle')
        pickle.dump(all_results, open(pickle_p, 'wb'))
            
        return all_results
    
    def run_basic_calibration_optuna(self, v0, r_ts_events_m):
        import optuna
        from optuna.visualization import plot_param_importances
        # Copy so that changes aren't implemented on the original df
        r_ts_events_m_copy = r_ts_events_m.copy()
        
        extention = 'basic_calibration'
        
        ws_predictor = run_ws_prediction(self.cnws_path, self.file_paths, self.catchment_name, self.TC_model, 
                                         self.RE_name, extention, self.WS_params, self.WS_preprocess_params)
        
        self.dynam_it_count = 0
        self.mape_record = []
        self.total_diff_record = []
        self.rmse_record = []
        self.kge_record = []
        
        def objective(trial):
            # Suggest values for the parameters
            ktc_high = trial.suggest_float('ktc_high', 1, 50)  # Log scale for wide range
            fa = trial.suggest_float('fa', 100, 1000)  # Example range for fa
            ktc_ratio = trial.suggest_float('ktc_ratio', 2.0, 5.0)  # Example range for ktc_high / ktc_low
            
            # Calculate ktc_low based on the ratio
            ktc_low = ktc_high / ktc_ratio
            
            # Modify channel extent
            modify_channel_extent(ws_predictor.file_paths, fa)
            
            # Predict SSL for each event
            y_obs = r_ts_events_m_copy['SSL (t event-1)'].values
            y_pred = []
            
            for i in r_ts_events_m_copy['Event_index'].values:
                idx = i
                line = r_ts_events_m_copy[r_ts_events_m_copy['Event_index'] == i]
                rf_id = int(idx)
                
                try:
                    ws_results = ws_predictor.run_ws(ktc_low, ktc_high, event_indexes=[idx])
                except:
                    ws_results = ws_predictor.run_ws(ktc_low, ktc_high, event_indexes=[idx])
                
                ws_results_e = ws_results[ws_results['Event_index'] == idx]
                y_pred.append(float(ws_results_e['SSL-WS (t event-1)']))
            
            r_ts_events_m_copy['SSL-WS (t event-1)'] = y_pred
            
            # Evaluate metrics
            global eval_
            eval_ = get_metrics(y_obs, y_pred, name='OPTUNA')
            
            mae = eval_['MAE']
            mape = eval_['MAPE']
            rmse = eval_['RMSE']
            total_diff = eval_['Total diff']
            kge = eval_['KGE efficiency']
            
            # Log metrics
            trial.set_user_attr('mae', mae)
            trial.set_user_attr('mape', mape)
            trial.set_user_attr('rmse', rmse)
            trial.set_user_attr('total_diff', total_diff)
            
            # Print progress
            print(f"Trial {trial.number}: ktc_high={ktc_high:.2f}, fa={fa:.2f}, ktc_ratio={ktc_ratio:.2f}")
            print(f"KGE: {kge:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            # Return the objective value to minimize
            return kge 
        
        # Create an Optuna study
        study = optuna.create_study(direction='maximize') 
        study.optimize(objective, n_trials=30)  # Run 50 trials (adjust as needed)
        
        # Retrieve feature importance values
        param_importances = optuna.importance.get_param_importances(study)
        
        # Plot feature importance using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        params = list(param_importances.keys())
        importance_values = list(param_importances.values())
        
        ax.barh(params, importance_values, color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Parameters')
        ax.set_title('Feature Importance from Optuna Optimization')
        plt.tight_layout()

        plt.show()

        # Get the best parameters
        best_params = study.best_params
        print("Best parameters:", best_params)
        
        # Run the model with the best parameters
        ktc_high = best_params['ktc_high']
        fa = best_params['fa']
        ktc_ratio = best_params['ktc_ratio']
        ktc_low = ktc_high / ktc_ratio
        
        modify_channel_extent(ws_predictor.file_paths, fa)
        
        ws_results_run = collect_WS_output(ws_predictor.file_paths['out_folder'], calibration=False)
        
        sdr = -1 * (ws_results_run['Sediment export via river (kg)'].sum() / ws_results_run['Total gross erosion (kg)'].sum())
        print('SDR = ' + str(np.around(sdr * 1, 2)))
        
        ws_obs_ts = merge_sim_obs(ws_results_run, r_ts_events_m, calibration=False)
        
        all_results = {
            'ws ts': ws_results_run,
            'ws obs ts': ws_obs_ts,
            'SDR': sdr,
            'eval': eval_,
            'best_params': best_params
        }
        
        cal_events = list(r_ts_events_m['Event_index'])
        ws_predictor.visualise_outputs(r_ts_events_m, calibration=True, calibration_events=cal_events)
        
        pickle_p = os.path.join(ws_predictor.file_paths['out_folder'], 'Basic_calibration.pickle')
        pickle.dump(all_results, open(pickle_p, 'wb'))
        
        return all_results
    
    def visualise_calibration(self, ws_obs_ts_cal):
        plot_calibration(ws_obs_ts_cal, out_path = self.file_paths['out_folder'])

        compare_event_distributions(ws_obs_ts_cal, self.ktc_low_l, self.ktc_high_l, 
                                    catchment_forcing = True, out_path = self.file_paths['out_folder'])
        compare_event_distributions(ws_obs_ts_cal, self.ktc_low_e, self.ktc_high_e, 
                                    catchment_forcing = True, out_path = self.file_paths['out_folder'])
        
        
    def export_calibration_results(self, out_folder, splines = False, extention = 'all'):
        if splines == True:
            f_name = 'calibration_results_splines.csv'
            df = pd.DataFrame({'Catchment': [self.catchment_name], 'TC_model': [self.TC_model], 'Calibration range':[self.ktc_cal_p],
                               'ktc_low_l':[self.ktc_low_l], 'ktc_high_l':[self.ktc_high_l], 'WS_params':[self.WS_params], 
                               'WS_preprocess_params':[self.WS_preprocess_params],
                               'stat_eval monthly':[self.eval_splines_monthly], 'stat_eval events':[self.eval_splines_events],
                               'splines_params':[self.splines_parameters], 'splines_params_tranformed':[self.splines_parameters_transformed]})
        else:
            
            f_name = f'calibration_results_{extention}.csv'
            
            df = pd.DataFrame({'Catchment': [self.catchment_name], 'TC_model': [self.TC_model], 'Calibration range':[self.ktc_cal_p],
                               'pcnt_in_envelope':[self.pcnt_in_envelope], 'ktc_low_l':[self.ktc_low_l],
                               'ktc_high_l':[self.ktc_high_l], 'WS_params':[self.WS_params], 'WS_preprocess_params':[self.WS_preprocess_params],
                               'stat_eval':[self.statistical_eval]})
        
        out_file_p = os.path.join(out_folder, f_name)
        try:
            csv = pd.read_csv(out_file_p)
            csv = csv.append(df)
            csv.to_csv(out_file_p, index = False)
        except:
            df.to_csv(out_file_p, index = False)
    
        

        


