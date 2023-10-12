# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:37:13 2022

@author: u0133999
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


def get_catchment_ts(df_ts, data_format):
    '''
    Generate a selection of statistical overview plots from the time series 
    record of an individual catchment in EUSEDcollab

    Parameters
    ----------
    df_ts : DATAFRAME
        The dataframe with an individual time series from a catchment
    data_format : STRING
        The time series format of the data

    '''
    #find the relevant column keys
    
    for col in df_ts.columns:
        if 'SSL' in col:
            sed_key = col
        elif 'Q' in col:
            Q_key = col
        elif 'SSC' in col:
            ssc_key = col
    
    #check if an ssc column was found. Not all datasets have it (e.g. monthly data)
    if 'ssc_key' in locals():
        ssc_exists = True
    else:
        ssc_exists = False
        

    #create a dataframe with the annual sum by resampling time series
    #aggregate to annual and plot bar chart 
    ts_y = df_ts.resample('Y').sum()
    ts_y['Year'] = ts_y.index.year
    #convert to tonnes
    ts_y['SSL (t yr-1)'] =  ts_y[sed_key] / 1000
    

    #format dataframes to get relevant fields for plotting
    #this is done in a specific way for each time series type
    if data_format == 'Daily data - fixed timestep':
        
        sed_key_t = 'SSL (t d-1)'
        #count days in annual average 
        ts_y['Count'] = df_ts[sed_key].resample('Y').count()
        #remove values under 1 kg d-1 - insignificant
        df_ts_h = df_ts[df_ts[sed_key] >= 1].copy(deep=True)
        #get ssl in tonnes
        df_ts_h[sed_key_t] = df_ts_h[sed_key] / 1000
        #log ssl
        df_ts_h['log ' +  sed_key_t] = np.log(df_ts_h[sed_key])
        #define the labels for plotting
        ly1 = 'Number of days in data record'
        ly2 = 'Suspended sediment load ($\mathregular{t \ d^{-1}}$)'
        t2 = 'Daily sediment yield'
        ly3 = "Sediment load ($\mathregular{kg \ d^{-1}}$)"
        lx3 = "Water discharge ($\mathregular{m^{3} \ day^{-1}}$)" 
        ly4 = "Sediment load ($\mathregular{t \ d^{-1}}$)"
        t4 = 'Daily sediment yield distribution'
        n_ts = 366

    elif data_format == 'Event data - variable timestep' or data_format == 'Event data - fixed timestep':
        
        sed_key_t = 'SSL (t event-1)'
        #count the number of events per year. needs an event index column
        df_ts['Year'] = df_ts.index.year
        #count the number of events
        ts_y['Count'] = df_ts.groupby('Year')['Event_index'].nunique().values
        #delete the year key
        del(df_ts['Year'])
        #if data is not daily, give an event count 
        #aggregate the timeseries into events
        df_ts_h = df_ts.groupby('Event_index').sum()
        #replace the ssc with the mean instead of sum
        df_ts_h[ssc_key] = df_ts.groupby('Event_index').mean()[ssc_key]
        #add the first date of the event as the index 
        df_ts_h.index = df_ts.groupby('Event_index').first()['Date (DD/MM/YYYY)']
        #get ssl in tonnes
        df_ts_h[sed_key_t] = df_ts_h[sed_key]/1000
        #log ssl
        df_ts_h['log ' +  sed_key_t] = np.log(df_ts_h[sed_key_t])
        
        new_cols = []
        for col in list(df_ts_h.columns):
            new_cols.append(col.replace('ts', 'event'))
        df_ts_h.columns = new_cols
        
        #define the labels for plotting
        ly1 = 'Number of events in data record'
        ly2 = 'Suspended sediment load ($\mathregular{t \ event^{-1}}$)'
        t2 = 'Event sediment yield'
        ly3 = "Sediment load ($\mathregular{t \ event^{-1}}$)"  
        lx3 = "Water discharge ($\mathregular{m^{3} \ event^{-1}}$)" 
        ly4 = "Sediment load ($\mathregular{t \ event^{-1}}$)"
        t4 = 'Event sediment yield distribution'
        

    elif data_format == 'Event data - aggregated':
        
        sed_key_t = 'SSL (t event-1)'
        #count the number of events per year. needs an event index column
        df_ts['Year'] = df_ts.index.year
        #get the count per year
        ts_y['Count'] = df_ts[sed_key].resample('Y').count()
        #remove insignificant events
        df_ts_h = df_ts[df_ts[sed_key] >= 1].copy(deep=True)
        df_ts_h[sed_key_t] = df_ts_h[sed_key]/1000
        df_ts_h['log ' +  sed_key_t] = np.log(df_ts_h[sed_key])
        
        new_cols = []
        for col in list(df_ts_h.columns):
            new_cols.append(col.replace('ts', 'event'))
        df_ts_h.columns = new_cols
        
        #define the labels for plotting
        ly1 = 'Number of events in data record'
        ly2 = 'Suspended sediment load (t event-1)'
        t2 = 'Event seidment yield'
        ly3 = "Sediment load ($\mathregular{t \ event^{-1}}$)"
        lx3 = "Water discharge ($\mathregular{m^{3} \ event^{-1}}$)" 
        ly4 = "Sediment load ($\mathregular{t \ event^{-1}}$)"
        t4 = 'Event sediment yield distribution'

    elif data_format == 'Monthly data':
        sed_key_t = 'SSL (t month-1)'
        #count the months in the annual sum
        ts_y['Count'] = df_ts[sed_key].resample('Y').count()
        
        #remove values under 1 kg d-1 - insignificant
        df_ts_h = df_ts[df_ts[sed_key] >= 1].copy(deep=True)
        df_ts_h[sed_key_t] = df_ts_h[sed_key]/1000
        df_ts_h['log ' +  sed_key_t] = np.log(df_ts_h[sed_key])
        
        #define the labels for plotting
        ly1 = 'Number of months in data record'
        ly2 = 'Suspended sediment load ($\mathregular{t \ month^{-1}}$)'
        t2 = 'Monthly sediment yield'
        ly3 = "Sediment load ($\mathregular{t \ month^{-1}}$)"
        lx3 = Q_key
        ly4 = "Sediment load ($\mathregular{t \ month^{-1}}$)"
        t4 = 'Monthly sediment yield distribution'
        n_ts = 13
        
    else:
        sys.exit('specify a compatible data format')
    
    #set the year as the dataframe index
    ts_y = ts_y.set_index('Year')
    
    #add a column with the datetime
    df_ts_h['Datetime'] = df_ts_h.index
    #add a column with the month 
    df_ts_h['Month'] = df_ts_h.index.month
    df_ts_h[sed_key_t] = df_ts_h[sed_key_t]
                
    return df_ts_h


     

def format_catchment_p(precip_catch, event_delineation_h = 6, precip_event_limit_mm = None):
    precip_catch['Timestamp'] = pd.to_datetime(precip_catch['Timestamp (DD/MM/YYYY hh:mm) '], dayfirst = True)
    #remove the zeros for the method to work
    precip_catch = precip_catch[precip_catch['Precipitation Depth (mm)'] > 0].copy()
    
    precip_ts = precip_catch.copy()
    precip_ts.index = precip_ts['Timestamp']
    precip_daily = precip_ts.resample('D').sum()
    
    precip_catch['T-1'] = precip_catch['Timestamp'].shift()
    #determine the time period between each timestamp - large differences are events
    precip_catch['delta t (h)'] = (precip_catch['Timestamp'] - precip_catch['T-1']).dt.total_seconds()/3600
    precip_catch['New event'] = (precip_catch['delta t (h)'] > event_delineation_h).astype(int)
    precip_catch['Event_index'] = np.cumsum(precip_catch['New event'])
    start_t = precip_catch.groupby('Event_index').first()['Timestamp']
    end_t = precip_catch.groupby('Event_index').last()['Timestamp']
    precip_events = precip_catch.groupby('Event_index', as_index = False).sum()
    precip_events['Start timestamp'] = start_t
    precip_events['End timestamp'] = end_t    
    #get the event duration in days
    precip_events['Event dur (d)'] = (precip_events['End timestamp'] - precip_events['Start timestamp']).astype('timedelta64[m]')/(24 * 60)
    precip_events.index = start_t
    precip_events = precip_events.drop(columns = 'delta t (h)')
    if precip_event_limit_mm is not None:
        precip_events = [precip_events['Precipitation Depth (mm)'] > precip_event_limit_mm].copy()
    else:
        print('No event limit (mm) set. Considering all events.')
    
    output = {}
    output['Precip events'] = precip_events
    output['Precip daily'] = precip_daily
    return output

def add_rfactor(precip_catch, EnS):
    alpha_p = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/R_Factor/ref_files_for_EMO5/alpha_params_v2.shp'
    beta_p = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/R_Factor/ref_files_for_EMO5/beta_params_v2.shp'
    
    a_b_ens = pd.DataFrame()
    
    alpha_m = gpd.read_file(alpha_p)
    beta_m = gpd.read_file(beta_p)
    
    columns = []
    for i in list(alpha_m.columns):
        if 'Month' in i:
            columns.append(i)
    
    a_b_ens['Month'] = np.arange(1,13)
    a_b_ens['Alpha'] = alpha_m[alpha_m['EnS_name'] == EnS][columns].T.values
    a_b_ens['Beta'] = beta_m[alpha_m['EnS_name'] == EnS][columns].T.values
    
    precip_catch['Month'] = precip_catch['Start timestamp'].dt.month
    
    precip_catch_m = precip_catch.merge(a_b_ens, how = 'left', on = 'Month')
    precip_catch_m['RE'] =  precip_catch_m['Alpha'] * precip_catch_m['Precipitation Depth (mm)'] ** precip_catch_m['Beta']

    precip_catch_m.index = precip_catch['Start timestamp']
    
    return precip_catch_m
    
    
    