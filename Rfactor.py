# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:52:54 2023

@author: u0133999
"""
import pandas as pd 

def correct_rfactor(r_ts, col_name, r_long_term):
    
    #get length of time series in years 
    len_y = (r_ts.index[-1] - r_ts.index[0]).days / 365
    
    #sum and divide by length - gives verage RE per year
    r_mean_a = r_ts[col_name].sum()/len_y
    
    #get the linear correction factor to linearly scale each event
    cor_f = r_long_term/r_mean_a 
    
    print('The R-factor correction factor is: ', str(cor_f))
    
    #correct all of the R-factor events based on the linear factor
    r_ts[col_name + ' corr'] = r_ts[col_name].values * cor_f
    
    return r_ts 
    


    
    
    
    
    