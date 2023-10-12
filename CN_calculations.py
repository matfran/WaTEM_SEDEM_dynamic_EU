# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:57:38 2020

@author: fmatt
"""
import scipy.stats.mstats as mstats
import numpy as np
import rasterio
from WS_preprocess_functions import read_raster, write_raster, plot_image
import sys
import warnings
import os
import pandas as pd

def extract_CN(parameter_inputs, h_cond, cn_table_p, p_int = 0):
    
    cn_table_p = 'CN_parameter_values.csv'
    cn_table = pd.read_csv(cn_table_p)
    
    cn_table = cn_table[cn_table['Physical intervention'] == p_int]
    
    lc = []
    cn_a = []
    
    for lc_value in parameter_inputs['Landcover'].unique():
        row = parameter_inputs[parameter_inputs['Landcover'] == lc_value]
        cn_id_min = row['CN_table_ID_min'].iloc[0]
        cn_id_max = row['CN_table_ID_max'].iloc[0]
        
        cn_range = cn_table[(cn_table['ID'] >= cn_id_min) & (cn_table['ID'] <= cn_id_max)]
        #if more than 1 option is present, select only the specified hydrological condition
        if len(cn_range) > 1:
            cn_range = cn_range[cn_range['Hydrological condition'].isin([h_cond, np.nan])]
            cn_avg = int(cn_range['A'].mean())
        else:
            cn_avg = int(cn_range['A'].iloc[0])
        
        lc.append(lc_value)
        cn_a.append(cn_avg)
    
    #apped a value for fallow land
    lc.append(3)
    cn_a.append(int(cn_table[cn_table['Land use'] == 'Fallow']['A'].iloc[0]))
    lc_cn_reclass = dict(zip(lc, cn_a))
    
    return lc_cn_reclass


def catchment_average(cn_grid, cell_count):
    cn_sum = np.nansum(cn_grid)
    cn_catchment_avg = cn_sum/cell_count
    return cn_catchment_avg
    
    
def calc_NAPI(precip_6hr_ts):
    #A function to calculate a 5-day normalised antecedent precip index from Hong et al., 2007
    #https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006WR005739
    
    
    cum_precip = []
    k = 0.85
    NAPI_list = []
    
    '''
    #TEMPORARY DUMMY DATA ---------------------------------------------------
    dt_array = pd.date_range(start = '19900101', freq = '6h', periods = 10000)
    
    precip_6hr_ts = pd.DataFrame({'precip mm' : np.random.gamma(3., 1., len(dt_array))}, 
                                 index = dt_array)
    #------------------------------------------------------------------------
    '''
    #calculate a whole array of NAPI values for the whole time serie
    precip_daily_df = precip_6hr_ts.resample('D').sum()
    precip_daily_ts = precip_daily_df['precip mm'].values
    avg_precip = np.mean(precip_daily_ts)
    
    for i in np.arange(5,len(precip_daily_ts)):
    #sum last 5 days of precip preceding first day of the year
        
        precip_5day = precip_daily_ts[[i-1, i-2, i-3, i-4, i-5]].sum()
        cum_precip.append(precip_5day)
        
        
        NAPI_top = (precip_daily_ts[i-1]*k**-1) + (precip_daily_ts[i-2]*k**-2)
        + (precip_daily_ts[i-3]*k**-3)+ (precip_daily_ts[i-4]*k**-4) + (precip_daily_ts[i-5]*k**-5)
        
        
        NAPI_bottom = avg_precip * (k**-1 + k**-2 + k**-3 + k**-4 + k**-5)
        #assuming k is constant we can calculate NAPI as following:
        NAPI = float(NAPI_top)/NAPI_bottom
        NAPI_list.append(NAPI)
    
    #create a dataframe with the outputs
    output = pd.DataFrame({'NAPI': NAPI_list, '5 day precip (mm)': cum_precip}, 
                          index = precip_daily_df.index[5:])
    
    return output

def calc_ant_rainfall(precip_6hr_ts, n_days):
    #A function to calculate a 5-day normalised antecedent precip index from Hong et al., 2007
    #https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006WR005739
    
    
    cum_precip = []

    #calculate a whole array of NAPI values for the whole time serie
    precip_daily_df = precip_6hr_ts.resample('D').sum()
    precip_daily_ts = precip_daily_df['precip mm'].values
    
    for i in np.arange(n_days,len(precip_daily_ts)):
    #sum last 5 days of precip preceding first day of the year
        ids = []
        for j in np.arange(n_days):
            ids.append(i - (j + 1))
            
        precip_ndays = precip_daily_ts[ids].sum()
        cum_precip.append(precip_ndays)

    
    #create a dataframe with the outputs
    output = pd.DataFrame({str(n_days) + ' day cumulative  precip (mm)': cum_precip}, 
                          index = precip_daily_df.index[n_days:])
    
    return output

def reclassify_LCtoCN(lc_array, parameter_inputs, slr_array_p = None, 
                      slr_threshold = None):
    
    '''
    write a loop to reclassify all elements to curve numbers. Input can be a dictionary 
    using predefined landcover to curve number reclassifications
    
    Trat out of bounds regions as nan
    '''
    
    #first assign cn value to agricultural land. Eveything with a value of 2
    #or above is a field parcel. Eventually this assignment can consider the 
    #vegetation condition 
    
    #get 2 reclassification dictionaries for good and poor hydrological condition
    lc_cn_reclass_good = extract_CN(parameter_inputs, h_cond = 'Good', p_int = 0)
    lc_cn_reclass_poor = extract_CN(parameter_inputs, h_cond = 'Poor', p_int = 0)
    
    lc_array = np.where(lc_array >= 2, lc_cn_reclass_good[2], lc_array)
    
    #loop through all elements and reclass lancover to curve number
    for lc in lc_cn_reclass_good:
        lc_array = np.where(lc_array == float(lc), float(lc_cn_reclass_good[lc]), lc_array)
    
    if slr_array_p is not None:

        slr_data = read_raster(slr_array_p)
        slr_array = slr_data['array']
        #Create a mask with pixels having a high slr value
        high_slr = slr_array > 0.4
        v_high_slr = slr_array > 0.6
        #reclassify areas with a high slr/poor hydrological condition
        lc_array = np.where(high_slr == 1, lc_cn_reclass_poor[2], lc_array)
        lc_array = np.where(v_high_slr == 1, lc_cn_reclass_poor[3], lc_array)


    #plot_image(lc_array, show = True, export = False, vmax = 100)    
    #lc_array variable now contains CN values - eq converts CN values
    #from IA = 0.2S to IA = 0.05 S
    cn_grid = 100/((1.879*((100/lc_array)-1)**1.15 )+1)
    #get new CN for out of bounds area
    
    return cn_grid    
    
def CN_runoff_coefficient(landlab_grid, precipitation, NAPI):
    
    landcover = landlab_grid.at_node['Land__cover']
    landcover, out_of_bounds = reclassify_LCtoCN(landcover)
    
    if NAPI <= 0.33:
        landcover = landcover/(2.281 - 0.01281*landcover) 
    elif NAPI >= 3:
        landcover = landcover/(0.427 + 0.00573*landcover)
    else:
        landcover = landcover
    
    S = ((25400/landcover)-254)
    
    runoff = ((precipitation - (0.05*S))**2)/(precipitation - (0.05 * S) + S)
    for i in np.arange(len(runoff)):
        if 0.05 * S[i] >= precipitation:
            runoff[i] = 0
        else:
            runoff[i] = runoff[i]
    
    #avoid error warning in divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        #create a runoff coefficient
        runoff_coeff = runoff/precipitation
        #correct for any divide by zero
        runoff_coeff[runoff == 0] = 0
        
    return runoff_coeff

    
    
def NCRS_CN(landcover_path, parameter_inputs, rf_event_mm, Ia_coeff = 0.2, runoff_type = 'absolute', 
            duration_d = 1, slr_array_p = None, NAPI = None, HSG_raster_path = None, 
            HSG_catchment = None, export_raster = False, out_path = None, plot_grid = False, 
            event_date = None, image_folder = None):
    '''
    A module to derive the discharge through the NCRS curve number method. The 
    function relies on a landcover input grid, a dictionary to reclassify landcover
    elements to a CN value, and an event rainfall value. 
    
    It returns a catchment grid of overland runoff (m), a lumped sum of discharge 
    from the grid (m3), and a lumped discharge calculated based on the average CN 
    value in the catchment (spatially lumped - most consistent with the curve number
    conceptualisation). 

    Parameters
    ----------
    landcover_path : STRING
        Path to landcover raster layer
    lc_cn_reclass : DICTIONARY
        Dictionary with landcover values (keys) and curve number target values 
        (values). These should all be for Hydrological Soil Group (HSG) A, then
        a scaling is made to other 
    rf_event_mm : INTEGER
        Event rainfall depth (integer)
    runoff_type : STRING
        Return an absolute runoff (in m) or a runoff coefficient (0-1). Input
        'absolute' (default) or 'coefficient'.
    NAPI : FLOAT, optional
        Normalised Antecedent Precipitation Index to adjust the curve
        number value. The default is None.
    HSG_path : STRING, optional
        Pathway to a HSG raster on which to modify the curve number. It 
        needs to be of the same resolution and dimensions as the landcover grid. 
        The default is None.
    HSG_catchment : INTEGER, optional. A singular catchment HSG class value. The 
    Curve number grid is reclassified based on this. The default is None. 

    Returns
    -------
    runoff_grid_m
    NUMPY ARRAY
        An array of simulated runoff depth (m) for every grid cell.
        
    Q_dist_m3
    FLOAT
        The sum of discharge (m3) calculated based on a spatially distributed 
        CN method across all pixels.
        
    Q_lumped_m3
    FLOAT
        The sum of discharge (m3) calculated based on a spatially lumped 
        CN method (i.e. the catchment average CN value)  .      

    '''
    #get a dictionary of values and unpack values
    raster_data = read_raster(landcover_path)
    lc_array = raster_data['array']
    cell_size = raster_data['cell size']
    nd_val = raster_data['no data value']

    '''
    Read the hydrological soil group layer. This needs to a singular value 
    or an array of the same dimensions as the landcover array to make modifications 
    to the curve number. Values range from 1-4 (A-D).
    
    '''
    if HSG_raster_path is not None:
        hsg_data = read_raster(HSG_raster_path)
        #values need to be unpacked from dictionary
        hsg_scalar = hsg_data['array']
        
        if not sorted(lc_array.shape) == sorted(hsg_scalar.shape):
            sys.exit('landcover and hydrological soil group arrays have different dimensions')
        hsg_type = 'array'
    elif HSG_catchment is not None:
        hsg_scalar = int(HSG_catchment)
        hsg_type = 'integer value'
    else:
        hsg_scalar = None 
    
    
    #get a count of all cells within the catchment boundaries 
    cell_count = np.where(lc_array == nd_val, False, True).sum()
    #convert nodata values to nans
    lc_array = np.where(lc_array == nd_val, np.nan, lc_array)

    #use the reclassify function to get initial curve number values 
    #initial
    cn_grid = reclassify_LCtoCN(lc_array, parameter_inputs, slr_array_p)

    '''
    Hydrological soil groups routine: 
    relationships taken from the Hawkins (2009) handbook to scale CN between 
    hydrological soil groups
    '''
    #define the scalar relationship values
    scalar_vals = {}
    scalar_vals[2] = [37.8, 0.622]
    scalar_vals[3] = [58.9, 0.411]
    scalar_vals[4] = [67.2, 0.328]
    
    #scale the CN values based on the array of HSGs 
    if hsg_scalar is not None and hsg_type == 'array':
        for i in np.arange(2,5):
            vals = scalar_vals[i]
            a = vals[0]
            b = vals[1]
            j = float(i)
            cn_grid = np.where(hsg_scalar == j, a + cn_grid * b, cn_grid)
    #scale the whole array based on the singular catchment HSG value        
    elif hsg_scalar is not None and hsg_type == 'integer value':
        vals = scalar_vals[hsg_scalar]
        a = vals[0]
        b = vals[1]  
        cn_grid = a + cn_grid * b
    
    
    '''
    Antecedent soil moisture routine. At the moment a Normalised Antecedent 
    Precipitation Index value is required to move between CN values. 
    '''
        
    #modify the curve number based on the antecedent soil moisture 
    if NAPI is not None:
        #make an adjustment based on the NAPI value
        #conditional criteria for scaling CN for AMC according to NAPI (Hong et al., 2007)
        if NAPI <= 0.33:
            cn_grid = cn_grid/(2.281 - 0.01281*cn_grid) 

        elif NAPI >= 3:
            cn_grid = cn_grid/(0.427 + 0.00573*cn_grid)
         
    else:
        cn_grid = cn_grid
    
    #get the catchment average CN value
    catchment_avg_CN = catchment_average(cn_grid, cell_count)
    
    '''
    Curve number calculation routine to simulate a spatially distributed runoff
    and lumped discharge value. 
    '''

    #define the S value as a grid 
    #calculate potential retention (S) - Hong et al., (2007) and Hawkins (2008) - metric units
    #THRESHOLD IS VERY IMPORTANT - IT CAN BE LOWERED OR REMOVED (DON'T MINUS 254)
    S = ((25400/cn_grid)-254)
    S_avg = ((25400/catchment_avg_CN)-254)
    
    #approximate the cell-wise runoff
    runoff_mm = ((rf_event_mm - (Ia_coeff*S))**2)/(rf_event_mm + (1-Ia_coeff) * S)

        
    #set areas not producing runoff to 0 (CN method criteria)
    runoff_mm = np.where(Ia_coeff * S >= rf_event_mm, 0.001, runoff_mm)
    
    #get the spatially lumped CN value
    if Ia_coeff * S_avg >= rf_event_mm:
        runoff_avg_mm = 0
    else:
        runoff_avg_mm = ((rf_event_mm - (Ia_coeff *S_avg))**2)/(rf_event_mm + (1-Ia_coeff) * S)
    
    #convert mm to m        
    runoff_lumped_m = runoff_avg_mm/1000
    runoff_grid_m = runoff_mm/1000

    #calculate the total discharge in m3
    Q_dist_m3 = np.nansum(runoff_grid_m * (cell_size * cell_size))
    Q_lumped_m3 = runoff_lumped_m * (cell_size * cell_size) * cell_count
    
    #if a runoff coefficient is required, convert to a 0-1 coefficient
    if runoff_type == 'coefficient':
        #get runoff as a percentage of rainfall event depth
        runoff_grid_out = (runoff_grid_m/(rf_event_mm/1000)) * 100
        #limit minimum duration to avoid very large runoff coefficients
        
        if duration_d < 0.25:
            duration_d = 0.25
        
        #normalise the runoff coefficient by the event duration in days
        runoff_grid_out = 1 + (runoff_grid_out/duration_d)
        
        #runoff_grid_out = np.where(runoff_grid_out > 99, 99, runoff_grid_out)
        
        
        print('Runoff coefficient (mean):', str(np.nanmean(runoff_grid_out)))

    else:
        runoff_grid_out = runoff_grid_m
        
    '''
    Optionally export the runoff array as a raster
    '''
    if export_raster == True:
        #fill nans with zero
        runoff_grid_out[np.isnan(runoff_grid_out)] = 0
        if out_path.endswith('.rst'):
            output_type = 'RST'
        else:
            output_type = 'GTiff'
            
        write_raster(runoff_grid_out, raster_data['all metadata'], out_path, 
                     output_type = output_type)

    if plot_grid == True:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        out_name = os.path.join(image_folder, event_date)
        plot_image(runoff_grid_out, event_date + ', ' + str(rf_event_mm) + ' mm', out_name, vmax = np.nanmax(runoff_grid_out))
        
    return runoff_grid_out, Q_dist_m3, Q_lumped_m3

