# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:58:46 2023

@author: u0133999
"""
import sys
from WS_preprocess_functions import read_raster, plot_image, write_raster
import numpy as np 
import os
import pandas as pd
import seaborn as sns
from fnmatch import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import re



def plot_ts(df, df2, out_path = None):
    
    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(15, 10))

    # Add x-axis and y-axis
    ax.scatter(df.index.values,
            df['RE corr'],
            color='purple', 
            s = 80, label = 'Rainfall erosivity events n = ' + str(len(df)))
    
    ax.scatter(df2.index.values,
            df2['SSL (t event-1)'],
            color='brown', 
            s = 80, label = 'Kinderveld sediment yield events n = ' + str(len(df2)))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05))
    ax.set_ylabel('Rainfall erosivity / Sediment load t/event')
    
    if out_path is not None:
        out_path = os.path.join(out_path, 'RAINFALL_EROSIVITY.png')
        plt.savefig(out_path) 
        
    
def aggregate_ws_grids(file_paths, raster_name, cmap = 'Blues', rusle = False, 
                       log = False, binary = False, export = False):
    
    events_directory = file_paths['out_folder']
    pattern = raster_name + ".rst"
    i = 0
    for path, subdirs, files in os.walk(events_directory):
        for name in files:
            if fnmatch(name, pattern):
                #file must contain target string in the name to be read 
                grid_info = read_raster(os.path.join(path, name))
                grid = grid_info['array']
                grid[grid == grid_info['no data value']] = np.nan
                if binary == True:
                    grid[grid == grid_info['no data value']] = np.nan
                    grid = np.where(grid > 0, 1., -1.)
                    
                if i == 0:
                    grid_all = grid
                    if raster_name in ['WATEREROS (mm per gridcel)']:
                        grid_max = grid
                        grid_min = grid
                else:
                    grid_all = grid_all + grid
                    if raster_name in ['WATEREROS (mm per gridcel)']:
                        grid_max = np.where(grid > grid_max, grid, grid_max)
                        grid_min = np.where(grid < grid_min, grid, grid_min)
                i = i + 1
        
    
    raster_name_max = None
    raster_name_min = None
    
    if raster_name in ['WATEREROS (mm per gridcel)']:
        grid_max = np.where(np.isnan(grid_info['array']), np.nan, grid_max)
        grid_min = np.where(np.isnan(grid_info['array']), np.nan, grid_min)
        raster_name_max = raster_name + '_max'
        raster_name_min = raster_name + '_min'
        
        if binary == True:
            raster_name = raster_name + '_binary'
            grid_all = np.where(np.isnan(grid_info['array']), np.nan, grid_all)

        plot_image(grid_all, name_str = raster_name, vmin = np.nanmin(grid_all), vmax = np.nanmax(grid_all), 
                   export = False, show = True, centre_cmap = True)
        grid_all_2 = np.where(grid_all > 5, 5, grid_all)
        grid_all_2 = np.where(grid_all_2 < -5, -5, grid_all_2)
        
        plot_image(grid_all_2, name_str = raster_name, vmin = -5, vmax = 5, 
                   export = False, show = True, centre_cmap = True)
    elif raster_name in ['UPAREA', 'SediExport_kg', 'SediIn_kg', 'SediOut_kg', 'RUSLE']:          
        #rusle_all = np.log10(rusle_all)
        plot_image(grid_all, name_str = raster_name, vmin = 0, vmax = np.nanmax(grid_all), 
                   export = False, show = True, cmap = cmap)
    elif raster_name in ['Capacity']:
        grid_all = np.log10(grid_all)
        plot_image(grid_all, name_str = raster_name, vmin = 0, vmax = np.nanmax(grid_all), 
                   export = False, show = True, cmap = cmap)        
        
    if rusle == True:
        #output is in kg m^2. Convert to t cell so it can be aggregated.
        grid_all = (grid_all * grid_info['cell size'])/1000
        grid_all = grid_all
        grid_sum = np.nansum(grid_all)
        grid_mean = np.nanmean(grid_all)
        grid_median = np.nanmedian(grid_all)
        grid_max = np.nanmax(grid_all)
        
        print('The grid (t cell-1) sum is: ' + str(grid_sum))
        print('The mean per (t cell-1) cell is: ' + str(grid_mean))
        print('The median per (t cell-1) cell is: ' + str(grid_median))
        print('The max per cell (t cell-1) is: ' + str(grid_max))
        
    if export == True:
        if binary == True:
            raster_name = raster_name + '_neg_pos'
        out_path = os.path.join(events_directory, raster_name + '_aggregated.tiff')
        
        #use the geotiff metadata in order to write geotiff
        lc_grid_p = os.path.join(file_paths['in_folder'], file_paths['lc_paths']['ws_lc tif'])
        lc_info = read_raster(lc_grid_p)
        metadata = lc_info['all metadata']
        write_raster(grid_all, metadata, out_path, output_type = 'GTIFF')
        
        if raster_name_max is not None and binary == False:
            out_path_max = os.path.join(events_directory, raster_name + '_max.tiff')
            write_raster(grid_max, metadata, out_path_max, output_type = 'GTIFF')
            
        if raster_name_min is not None and binary == False:
            out_path_min = os.path.join(events_directory, raster_name + '_min.tiff')
            write_raster(grid_min, metadata, out_path_min, output_type = 'GTIFF')
            
    return grid_all
        
        
def sample_ws_grids(events_directory, raster_name, secondary_grid_path = None, secondary_grid_name = None):
    
    pattern = ".rst"
    i = 0
    for path, subdirs, files in os.walk(events_directory):
        for name in files:
            if name.endswith(pattern) and raster_name in name:
                event_n = float(re.findall(r'\d+', os.path.join(path, name))[-1])
                #file must contain 'RUSLE.rst' in the name to be read 
                grid_info = read_raster(os.path.join(path, name))
                grid = grid_info['array']
                grid[grid == grid_info['no data value']] = np.nan
                grid_flat = grid.flatten()
                grid_flat_m = grid_flat[~np.isnan(grid_flat)]
                
                
                event_index = np.full(grid_flat_m.shape, event_n)
                grid_flat_m = np.c_[grid_flat_m, event_index]
                
                if secondary_grid_path is not None:
                    array_info = read_raster(secondary_grid_path)
                    array = array_info['array']
                    array_flat = array.flatten()
                    array_flat = array_flat[~np.isnan(grid_flat)]
                    array_flat = np.where(array_flat >= 1, 1, array_flat)
                    grid_flat_m = np.c_[grid_flat_m, array_flat]

                if i == 0:
                    all_vals = grid_flat_m
                else:
                    all_vals = np.concatenate((all_vals, grid_flat_m), axis = 0)

                i = i + 1
    if secondary_grid_path is not None:
        df_vals = pd.DataFrame(all_vals, columns = [raster_name, 'Event_index', secondary_grid_name])    
    else:
        df_vals = pd.DataFrame(all_vals, columns = [raster_name, 'Event_index'])
    df_vals = df_vals.sort_values('Event_index')
    return df_vals

#def check_outputs():

def plot_event_ws(file_paths, r_ts, raster_name = 'RUSLE', 
                  strip_plot = False, log_y = False, export = True, 
                  out_path = None):
    
    if raster_name in ['CN', 'Cfactor']:
        events_directory = file_paths['in_folder']
    else:
        events_directory = file_paths['out_folder']
        

    out_path = file_paths['out_folder']
    lc_grid_p = os.path.join(file_paths['in_folder'], file_paths['lc_paths']['ws_lc tif'])
    df_ws_event = sample_ws_grids(events_directory, raster_name = raster_name,
                                        secondary_grid_path = lc_grid_p, secondary_grid_name = 'Landcover')
    
    lc_code = [-10., -4.0, -3.0, -2.0, -1., 0., 1.] 
    lc_name = ['Path', 'Grassland', 'Forest', 'Built-up', 'Stream', 'Other', 'Arable']
    
    
    reclass = dict(zip(lc_code, lc_name))
    df_ws_event['Landcover'] = df_ws_event['Landcover'].replace(reclass)
    
    df_ws_event = df_ws_event[~df_ws_event['Landcover'].isin(['Stream'])]
    
    df_ws_event = df_ws_event.merge(r_ts, on = 'Event_index', how = 'left')
    
    if strip_plot == True:
        f, ax = plt.subplots(figsize=(25, 15))
        sns.despine(f, left=True, bottom=True)
        
        if raster_name == 'Cfactor':
            df_ws_event = df_ws_event[df_ws_event['Landcover'].isin(['Arable'])]
            sns.violinplot(data = df_ws_event, x = 'Date', y = raster_name, 
                          size = 20, orient = 'v', linewidth=1)
       
        else:
            sns.stripplot(data = df_ws_event, x = 'Date', y = raster_name, hue = 'Landcover', 
                          size = 20, alpha = 0.05, edgecolor = 'black', 
                          orient = 'v', linewidth=1) 
        
        if log_y == True:
            ax.set_yscale('log')
        ax.set_title('') 
        ax.set_xlabel('Event date')
        ax.set_ylabel(raster_name)
        ax.tick_params(axis='x', labelrotation=90)
        
        if out_path is not None:
            out_p = os.path.join(out_path, raster_name + '.png')
            plt.savefig(out_p)
            
    if export == True:
        df_ws_event.to_pickle(os.path.join(out_path, raster_name + '_sampled_grids.pickle'))

        
def compare_event_distributions(ws_obs_ts, ktc_low, ktc_high, catchment_forcing = True, 
                                out_path = None):
    
    #only consider events that match with an observation - we assume that the measured 
    #data captures all of the sediment
    if catchment_forcing == True:
        ws_obs_ts_cal = ws_obs_ts[ws_obs_ts['SSL (t event-1)'].notna()]
    
    ws_obs_ts_cal = ws_obs_ts_cal.loc[(ws_obs_ts['ktc_low'] == ktc_low) & (ws_obs_ts['ktc_high'] == ktc_high)]
    
    sns.set(font_scale = 3)
    f, ax = plt.subplots(figsize=(25, 15))
    sns.despine(f, left=True, bottom=True)

    max_sim = ws_obs_ts['SSL-WS (t event-1)'].max()
    max_obs = ws_obs_ts['SSL (t event-1)'].max()
    if max_sim >= max_obs:
        bins = np.histogram_bin_edges(ws_obs_ts_cal['SSL-WS (t event-1)'], bins = 15)        
    else:
        bins = np.histogram_bin_edges(ws_obs_ts_cal['SSL (t event-1)'], bins = 15)
        
    sns.histplot(data = ws_obs_ts_cal, x= 'SSL-WS (t event-1)', ax=ax, color="blue", bins = bins, alpha = 0.5)
    sns.histplot(data = ws_obs_ts_cal, x= 'SSL (t event-1)', ax=ax, color="grey", bins = bins, alpha = 0.5)    
    ax.set_title('') 
    ax.set_xlabel('Event sediment yield (t event-1)')
    ax.set_ylabel('n events')   
    
    if out_path is not None:
        out_p = os.path.join(out_path, 'HISTOGRAM_DISTRIBUTIONS.png')
        plt.savefig(out_p)
    
    #make a bivariate plot and include the month
    f, ax = plt.subplots(figsize=(25, 15))
    sns.despine(f, left=True, bottom=True)

        
    sns.histplot(data = ws_obs_ts_cal, y= 'SSL-WS (t event-1)', x = 'Month', 
                 ax=ax, color="blue", discrete=(True, False), alpha = 0.5, cbar = True, common_bins= (True, True))
    sns.histplot(data = ws_obs_ts_cal, y= 'SSL (t event-1)', x = 'Month',
                 ax=ax, color="grey", discrete=(True, False), alpha = 0.5, cbar = True, common_bins= (True, True))    
    ax.set_title('') 
    ax.set_xlabel('Month')
    ax.set_ylabel('SSL (t event-1)')
    
    if out_path is not None:
        out_p = os.path.join(out_path, 'MONTHLY_DISTRIBTIONS.png')
        plt.savefig(out_p)
    
def merge_sim_obs(ws_results, r_ts_events_m, calibration = False):
    
    #merge with the rainfall erosivity information
    ws_obs_ts = ws_results.merge(r_ts_events_m, on = 'Event_index', 
                                how = 'left').sort_values('Event_index').sort_index()
    
    #ensure that all simulations have a matching observation
    ws_obs_ts = ws_obs_ts[~(ws_obs_ts['SSL (t event-1)'].isna()) & ~(ws_obs_ts['SSL-WS (t event-1)'].isna())].copy()
    
    
    if calibration == True:
        cols_all_poss = ['Event_index', 'ktc_low', 'ktc_high', 'tot_erosion', 'tot_sedimentation', 'sed_river',
               'sed_noriver', 'sed_buffer', 'sed_openwater', 'outlet_1',
                'Precipitation Depth (mm)','Start timestamp', 'End timestamp', 'Event dur (d)',
               'Month', 'Alpha', 'Beta', 'RE', 'RE corr', 'Q (m3 s-1)',
               'Q (m3 event-1)', 'SSC (kg m-3)', 'SSL (t event-1)', 'SSL-WS (t event-1)']
        cols = []
        for col in cols_all_poss:
            if col in ws_obs_ts.columns:
                cols.append(col)
                
        ws_obs_ts = ws_obs_ts[cols]

        ws_obs_ts['pred-obs_abs (t event-1)'] = abs(ws_obs_ts['SSL (t event-1)'] - ws_obs_ts['SSL-WS (t event-1)'])
  
    ws_obs_ts['Date'] = ws_obs_ts['Start timestamp'].dt.date
    
    return ws_obs_ts
    
def export_df(ws_obs_ts, path_out):
    
    df_out = ws_obs_ts[['Event_index','Start timestamp', 'End timestamp', 'Event dur (d)', 'ktc_low', 'ktc_high',
                           'Month', 'SSL-WS (t event-1)', 'SSL (t event-1)']]
    df_out.to_csv(path_out)
    
    
    
    