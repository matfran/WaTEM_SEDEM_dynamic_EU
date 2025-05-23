# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:52:16 2023

@author: u0133999
"""
import pandas as pd 
import numpy as np
import re


def ei30_from_ts(emo5_pr_st, EnS, col_name, alpha_m, beta_m, time_resolution = 6):
    '''
    To do:
        
        Automate the EnS selection
        Create a routine for nans in alpha and beta
        document code
    
    '''
    name = int(re.findall("\d+", col_name)[0])

    #define first the timesteps which potentially belong to an event
    emo5_pr_st['Timestamp'] = emo5_pr_st.index
    emo5_pr_st['potential event'] = (emo5_pr_st[col_name] > 1.27).astype(int)
    emo5_pr_sliced = emo5_pr_st[emo5_pr_st['potential event'] == 1].copy(deep = True)
    #when only potential event timesteps are sliced, get delta t
    emo5_pr_sliced['T-1'] = emo5_pr_sliced['Timestamp'].shift()
    emo5_pr_sliced['delta t (h)'] = (emo5_pr_sliced['Timestamp'] - emo5_pr_sliced['T-1']).astype('timedelta64[m]')/(60)
    #a new event is present if the time distance exceeds one timestep
    emo5_pr_sliced['New event'] = (emo5_pr_sliced['delta t (h)'] > time_resolution).astype(int)
    #index the events
    emo5_pr_sliced['Event_index'] = np.cumsum(emo5_pr_sliced['New event'])
    #group by the event index (sum)
    emo5_events = emo5_pr_sliced.groupby('Event_index', as_index = False).sum()
    
    #get a start and end timestamp. EMO5 accumulates precip and the timestamp represents the 
    #end of the accumulation period
    #minus the time resolution. Given that precipitation is accumulated from n hours to timestamp
    start_t = emo5_pr_sliced.groupby('Event_index').first()['Timestamp'] - pd.Timedelta(hours = time_resolution)
    end_t = emo5_pr_sliced.groupby('Event_index').last()['Timestamp']
    emo5_events['Start timestamp'] = start_t
    emo5_events['End timestamp'] = end_t    
    #get the event duration in hours
    emo5_events['Event dur (h)'] = (emo5_events['End timestamp'] - emo5_events['Start timestamp']).astype('timedelta64[m]')/(60)
    emo5_events.index = start_t
    
    #get 2 masks: 1) the total precip in an event > 12.7, or 2) the precip in one timestep > 6.35 mm
    mask_1 = (emo5_events[col_name] >= 12.7).values
    mask_2 = (emo5_events[col_name] >= 6.35) & (emo5_events[col_name] < 12.7) & (emo5_events['Event dur (h)'] == time_resolution)
    #combine the masks and slice the potential events
    emo5_ei = emo5_events[mask_1 | mask_2].copy(deep = True)
    
    #get a dataframe of the relevant alpha and beta parameters for the EnS
    a_b_ens = pd.DataFrame()
    
    columns = []
    for i in list(alpha_m.columns):
        if 'Month' in i:
            columns.append(i)
    
    a_b_ens['Month'] = np.arange(1,13)
    a_b_ens['Alpha'] = alpha_m[alpha_m['EnS_name'] == EnS][columns].T.values
    a_b_ens['Beta'] = beta_m[alpha_m['EnS_name'] == EnS][columns].T.values
    
    #prioritise backfill and fill remaining months in a forward 
    a_b_ens = a_b_ens.fillna(method = 'bfill')
    a_b_ens = a_b_ens.fillna(method = 'ffill')
    
    '''
    After this, events occuring during cold periods should be removed. This 
    can be done in the workflow based on the maximum daily temperature.
    '''
    #get the month of the event
    emo5_ei['Month'] = emo5_ei['Start timestamp'].dt.month
    #merge based on the month
    emo5_ei_m = emo5_ei.merge(a_b_ens, how = 'left', on = 'Month')
    emo5_ei_m['RE EMO'] =  emo5_ei_m['Alpha'] * emo5_ei_m[col_name] ** emo5_ei_m['Beta']
    emo5_ei_m['Station_Id'] = name
    emo5_ei_m['EnS_name'] = EnS
    emo5_ei_m['EnZ'] = emo5_ei_m['EnS_name'].str[:3]
    emo5_ei_m = emo5_ei_m.rename(columns = {col_name: 'Rainfall depth (mm)'})
    emo5_ei_m = emo5_ei_m[['Station_Id', 'EnS_name', 'EnZ', 'Start timestamp', 'End timestamp', 
                           'Event dur (h)','Rainfall depth (mm)', 'RE EMO']]
    return emo5_ei_m
    
    