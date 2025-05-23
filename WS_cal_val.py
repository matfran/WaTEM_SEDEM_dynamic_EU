# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:59:54 2023

@author: u0133999
"""
import sys
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import metrics
import numpy as np
from scipy import stats
import hydroeval as he




def calibrate_lumped_emo_precip(ts_val_all, ts_sim_all):
    """
    Calibrate the model in a lumped way. This takes the whole time series 
    of catchment runoff/sediment export events and the timeseries of WaTEM-SEDEM
    realisations. The optimal 2 ktc parameters are found to match the TOTAL AGGREGATED 
    sediment yield between the two. The reference time periods of the observations 
    and simulations should match in order for the calibration to be meaningfull. 
    Additonally, it is important that the observation timeseries is COMPLETE.
    

    Parameters
    ----------
    ts_catchment_all : PANDAS DATAFRAME
        All catchment measurement realisations over the reference period
    ts_ws_all : PANDAS DATAFRAME
        All WaTEM-SEDEM model realisations over the reference period.

    Returns
    -------
    None.

    """
    
    sum_sl_t = ts_val_all['SSL (t event-1)'].sum()
    #group event sediment yield 
    cols = ['tot_erosion', 'tot_sedimentation', 'sed_river',
           'sed_noriver', 'sed_buffer', 'sed_openwater']
    sim_per_ktc_group = ts_sim_all.groupby(['ktc_low', 'ktc_high'], as_index = False)[cols].sum()
    sim_per_ktc_group['sim - obs (t)'] = (sim_per_ktc_group['sed_river']/1000) - sum_sl_t
    sim_per_ktc_group['abs(sim - obs (t))'] = abs(sim_per_ktc_group['sim - obs (t)']) 
    sim_per_ktc_group = sim_per_ktc_group.sort_values('abs(sim - obs (t))')
    return sim_per_ktc_group

def calibrate_lumped_catch_precip(ws_obs_ts, plot_sdr = False, out_path = None):
    """
    Calibrate the model in a lumped way. This takes the whole time series 
    of catchment runoff/sediment export events and the timeseries of WaTEM-SEDEM
    realisations. The optimal 2 ktc parameters are found to match the TOTAL AGGREGATED 
    sediment yield between the two. The reference time periods of the observations 
    and simulations should match in order for the calibration to be meaningfull. 
    Additonally, it is important that the observation timeseries is COMPLETE.
    

    Parameters
    ----------
    ts_catchment_all : PANDAS DATAFRAME
        All catchment measurement realisations over the reference period
    ts_ws_all : PANDAS DATAFRAME
        All WaTEM-SEDEM model realisations over the reference period.

    Returns
    -------
    None.

    """
    #only consider events that match with an observation - we assume that the measured 
    #data captures all of the sediment
    ws_obs_ts = ws_obs_ts[ws_obs_ts['SSL (t event-1)'].notna()]
    sum_sl_t = ws_obs_ts['SSL (t event-1)'].sum()
    #group event sediment yield 
    cols = ['tot_erosion', 'tot_sedimentation', 'sed_river', 'sed_noriver', 'sed_buffer', 
            'sed_openwater', 'SSL-WS (t event-1)', 'SSL (t event-1)']
    
    
    ktc_grouped = ws_obs_ts.groupby(['ktc_low', 'ktc_high'], as_index = False)[cols].sum()
    #ignore 0 predictions
    ktc_grouped = ktc_grouped[ktc_grouped['SSL-WS (t event-1)'] > 1]
    ktc_grouped['sim - obs (t)'] = (ktc_grouped['SSL-WS (t event-1)']) - ktc_grouped['SSL (t event-1)']
    ktc_grouped['abs(sim - obs (t))'] = abs(ktc_grouped['sim - obs (t)']) 
    ktc_grouped['SDR'] = ktc_grouped['SSL-WS (t event-1)'] / ((ktc_grouped['tot_erosion'] / -1000))
    ktc_grouped = ktc_grouped.sort_values('abs(sim - obs (t))')
    
    if plot_sdr == True:
        fig, ax = plt.subplots(figsize = (20,7))
        sns.histplot(data = ktc_grouped, x = 'SDR', y = 'ktc_high', ax = ax)
        out_path = os.path.join(out_path, 'SDR_KTC_lumped_calibration.png')
        plt.savefig(out_path)    

    return ktc_grouped

def calibrate_event_catch_precip(ws_obs_ts, objective_function = 'RSME_events', ascending = False):
    """
    Calibrate the model on a per-event. This takes the whole time series 
    of catchment runoff/sediment export events and the timeseries of WaTEM-SEDEM
    realisations. The optimal 2 ktc parameters are found to match the average ABSOLUTE ERROR of the 
    sediment yield between the two. The reference time periods of the observations 
    and simulations should match in order for the calibration to be meaningfull. 
    Additonally, it is important that the observation timeseries is COMPLETE.
    

    Parameters
    ----------
    ts_catchment_all : PANDAS DATAFRAME
        All catchment measurement realisations over the reference period
    ts_ws_all : PANDAS DATAFRAME
        All WaTEM-SEDEM model realisations over the reference period.

    Returns
    -------
    None.

    """
    #only consider events that match with an observation - we assume that the measured 
    #data captures all of the sediment
    ws_obs_ts = ws_obs_ts[ws_obs_ts['SSL (t event-1)'].notna()]
    #group event sediment yield 
    cols = ['tot_erosion', 'tot_sedimentation', 'sed_noriver', 'sed_buffer', 
            'sed_openwater', 'SSL-WS (t event-1)', 'SSL (t event-1)', 
            'pred-obs_abs (t event-1)']
    
    ktc_pairs = ws_obs_ts.groupby(['ktc_low', 'ktc_high'], as_index = False).first()[['ktc_low', 'ktc_high']]
    
    mae_events = []
    rmse_events = []
    total_diff_events = []
    kge_events = []
    
    #iterate through ktc pairs and evaluate
    for i in np.arange(len(ktc_pairs)):
        row = ktc_pairs.iloc[i]
        ktc_low = row['ktc_low']
        ktc_high = row['ktc_high']
        
        ktc_pair_sim = ws_obs_ts[(ws_obs_ts['ktc_low'] == ktc_low) & (ws_obs_ts['ktc_high'] == ktc_high)]
        
        y_obs = ktc_pair_sim['SSL (t event-1)'].values
        y_sim = ktc_pair_sim['SSL-WS (t event-1)'].values
        
        eval_ = get_metrics(y_obs, y_sim)
            
        mae = eval_['MAE']
        rmse = eval_['RMSE']
        total_diff = eval_['Total diff']
        kge = eval_['KGE efficiency']
        
        mae_events.append(mae)
        rmse_events.append(rmse)
        total_diff_events.append(total_diff)
        kge_events.append(kge)
    
    ktc_pairs['MAE_events'] = mae_events
    ktc_pairs['RMSE_events'] = rmse_events
    ktc_pairs['Total diff'] = total_diff_events
    ktc_pairs['KGE efficiency_events'] = kge_events
    #sort by objective function
    ktc_pairs = ktc_pairs.sort_values(objective_function, ascending = ascending)
    
    return ktc_pairs

def calibrate_monthly_mean_diff(ws_obs_ts):
    
    ktc_pairs = ws_obs_ts.groupby(['ktc_low', 'ktc_high'], as_index = False).first()[['ktc_low', 'ktc_high']]
    mae_monthly = []
    rmse_monthly = []
    total_diff_monthly = []
    
    #iterate through ktc pairs and evaluate
    for i in np.arange(len(ktc_pairs)):
        row = ktc_pairs.iloc[i]
        ktc_low = row['ktc_low']
        ktc_high = row['ktc_high']
        
        ktc_pair_sim = ws_obs_ts[(ws_obs_ts['ktc_low'] == ktc_low) & (ws_obs_ts['ktc_high'] == ktc_high)]
        
        monthly_mean = get_mean_monthly_avg(ktc_pair_sim)
        
        y_obs = monthly_mean['SSL (t event-1)'].values
        y_sim = monthly_mean['SSL-WS (t event-1)'].values
        
        eval_ = get_metrics(y_obs, y_sim)
           
        mae = eval_['MAE']
        rmse = eval_['RMSE']
        total_diff = eval_['Total diff']
        
        mae_monthly.append(mae)
        rmse_monthly.append(rmse)
        total_diff_monthly.append(total_diff)

                       
    ktc_pairs['MAE_monthly_avg'] = mae_monthly
    ktc_pairs['RMSE_monthly_avg'] = rmse_monthly
    ktc_pairs['Total diff'] = total_diff_monthly
    
    ktc_pairs = ktc_pairs.sort_values('RMSE_monthly_avg')
    
    return ktc_pairs
        

def evaluate_maximum_efficiency(ws_obs_ts):
    cols = ['SSL-WS (t event-1)', 'SSL (t event-1)', 'pred-obs_abs (t event-1)']
    #calculate the minimum achievable difference between sim and obs for the calibration routine
                                                                                          
    event_predictions = ws_obs_ts.loc[ws_obs_ts.groupby('Event_index')['pred-obs_abs (t event-1)'].idxmin()].rename(columns = {'pred-obs_abs (t event-1)': 'Min pred-obs_abs (t event-1)'})  
    event_predictions['SSL-WS_max (t event-1)'] = ws_obs_ts.groupby(['Event_index'], as_index = False)[cols].max()['SSL-WS (t event-1)'].values
    event_predictions['SSL-WS_min (t event-1)'] = ws_obs_ts.groupby(['Event_index'], as_index = False)[cols].min()['SSL-WS (t event-1)'].values
                               

    #calculate the % error for each event                                        
    event_predictions['% error'] = event_predictions['Min pred-obs_abs (t event-1)'] / event_predictions['SSL (t event-1)'] * 100                                                                          
    min_abs_error_total = event_predictions['Min pred-obs_abs (t event-1)'].sum()
    mean_pcnt_error_best = event_predictions['% error'].mean()
    
    return event_predictions
    
def plot_sim_obs_calibrated(ws_obs_ts, ktc_low = None, ktc_high = None, title = None, out_path = None):
    
    if ktc_low is not None and ktc_high is not None:
        df = ws_obs_ts[(ws_obs_ts['ktc_low'] == ktc_low) & (ws_obs_ts['ktc_high'] == ktc_high)].copy()
    else:
        df = ws_obs_ts
        
    df['pred-obs (t event-1)'] = df['SSL-WS (t event-1)'] - df['SSL (t event-1)']

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize = (30,15), gridspec_kw={'height_ratios': [2, 1]})
    sns.scatterplot(data = df, x ='Date', y = 'SSL-WS (t event-1)', color = 'blue', s = 80, alpha = 0.7, ax = ax0)
    sns.scatterplot(data = df, x ='Date', y = 'SSL (t event-1)', color = 'black', alpha = 0.7, s = 80, ax = ax0)
    ax0.set_yscale('log')
    ax0.set_ylabel('Sediment load (t/event)')
    
    sns.histplot(data = df, x = 'pred-obs (t event-1)', alpha = 0.5, binwidth = 20,
                 color = 'red', stat = 'probability', ax = ax1)
    ax1.set_xlabel('Predicted - Observed SSL (t/event)')
    
    if title is not None:
        plt.suptitle(title)
        
    if out_path is not None:
        out_path = os.path.join(out_path, 'CALIBRATION_TIMESERIES_PLOT' + title + '.png')
        plt.savefig(out_path)    

    
def plot_calibration(df, out_path = None):
    
    sns.set(font_scale = 2.5)
    fig, ax = plt.subplots(figsize = (20,10))
    sns.stripplot(data = df, x = 'Date', y = 'SSL (t event-1)', 
                  size = 20, alpha = 0.005, edgecolor = 'black', color = 'white', 
                  orient = 'v', linewidth=1)
    sns.boxplot(data = df, x = 'Date', y = 'SSL-WS (t event-1)', color = 'grey', ax = ax)
    ax.set_yscale('log')
    ax.set_ylabel('Sediment load t/event')
    ax.tick_params(axis='x', labelrotation=90)
    
    if out_path is not None:
        out_path = os.path.join(out_path, 'CALIBRATION_PLOT.png')
        plt.savefig(out_path)
        
        
def plot_sim_realisations(df):
    
    fig = plt.figure(figsize=(14,10))
    ax = Axes3D(fig)
    ax.scatter(df['ktc_high'], df['ktc_low'], df['pred-obs_abs'])
    ax.set_xlabel('ktc_high')
    ax.set_ylabel('ktc_low')
    ax.set_zlabel('pred-obs_abs (kg event-1)')
    plt.show()


def get_metrics(y_obs, y_sim, name = 'Catchment', print_ = False):
    #https://stackoverflow.com/questions/63903016/calculate-nash-sutcliff-efficiency
    def nse(y_sim, y_obs):
        nse = (1-(np.sum((y_obs-y_sim)**2)/np.sum((y_obs-np.mean(y_obs))**2)))
        return nse
    
    n = len(y_sim)
    try:
        mae = metrics.mean_absolute_error(y_obs, y_sim)
    except:
        mae = None
        
    try:    
        mse = metrics.mean_squared_error(y_obs, y_sim)
    except:
        mse = None
        
    try:
        rmse = np.sqrt(metrics.mean_squared_error(y_obs, y_sim))
    except:
        rmse = None
        
    try:
        mape = np.mean(np.abs((y_obs, y_sim) / np.abs(y_obs))) * 100
    except:
        mape = None
    
    try:
        SR_scipy = stats.spearmanr(y_obs, y_sim)
    except:
        SR_scipy = None 
    
    try:
        r_all = stats.linregress(y_obs, y_sim)
        r_scipy = r_all[2]
        r2_scipy = r_scipy ** 2
    except:
        r_scipy = None
        r2_scipy = None
        
    try:
        r_log_all = stats.linregress(np.log10(y_obs), np.log10(y_sim))
        r_scipy_log = r_log_all[2]
        r2_scipy_log = r_scipy_log ** 2
    except:
        r_scipy_log = None
        r2_scipy_log = None
    
    
    try:
        nse_ = nse(y_sim, y_obs)
    except:
        nse_ = None
        
    try:
        nse_log = nse(np.log10(y_sim), np.log10(y_obs))
    except:
        nse_log = None
        
    try:
        skew_obs = stats.skew(y_obs)
    except:
        skew_obs = None
    
    try:
        skew_sim = stats.skew(y_sim)
    except:
        skew_sim = None
    
    try:
        ws_total = sum(y_sim)
    except:
        ws_total = None 
    
    try:
        obs_total = sum(y_obs)
    except:
        obs_total = None
    
    try:
        total_diff = abs(ws_total - obs_total)
    except:
        total_diff = None
        
    try:
        kge, r, alpha, beta = he.evaluator(he.kge, y_sim, y_obs)
        kge = kge[0]
    except:
        kge = None
    
    keys = ['Catchment name', 'n', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SR scipy', 
            'R2 scipy', 'NSE', 'NSE_log', 'Skew observation', 'Skew simulation',
            'WS_total_load', 'Obs_total_load', 'Total diff', 'R2_log scipy',
            'Pearson_all', 'Pearson_all_log', 'KGE efficiency']
    dic = dict(zip(keys, [name, n, mae, mse, rmse, mape, SR_scipy, 
                          r2_scipy, nse_, nse_log, skew_obs, skew_sim,
                          ws_total, obs_total, total_diff, r2_scipy_log,
                          r_all, r_log_all, kge]))
    
    return dic
    
def validate_matched_events(ws_obs_ts, name, ktc_low, ktc_high):
    
    df = ws_obs_ts[(ws_obs_ts['ktc_low'] == ktc_low) & (ws_obs_ts['ktc_high'] == ktc_high)].copy()
    #only take non nan values
    df = df[~(df['SSL-WS (t event-1)'].isna()) & ~ (df['SSL (t event-1)'].isna())] 
    y_obs = df['SSL (t event-1)'].values
    y_sim = df['SSL-WS (t event-1)'].values
    

    dic = get_metrics(y_obs, y_sim, name = name)
    
    return dic


def analyse_residuals(df, plot = True):
    
    df['pred-obs (t event-1)'] = df['SSL-WS (t event-1)'] - df['SSL (t event-1)']
    
    
    # Extract season from datetime
    df['season'] = df['Start timestamp'].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Autumn
    
    # Compute seasonal % bias
    seasonal_bias = (
        df.groupby("season").apply(
            lambda g: (g["pred-obs (t event-1)"].sum() / g["SSL (t event-1)"].sum()) * 100
        )
    ).rename("Seasonal % Bias")
    
    if plot:
        seasonal_bias.plot(kind="bar", ylabel="% Bias", title="Average % Seasonal Bias", rot=0)
        plt.xticks(ticks=range(4), labels=["Winter", "Spring", "Summer", "Autumn"], rotation=45)
        plt.show()
    
    try:
        df['tot_erosion'] = df['tot_erosion'] * -1
    except:
        df['tot_erosion'] = df['Total gross erosion (kg)'] * -1
    
    try:
        df['pred-obs_ abs (t event-1)/RE'] = df['pred-obs_abs (t event-1)']/df['RE corr']
    except:
        df['pred-obs_abs (t event-1)'] = abs(df['pred-obs (t event-1)'])
        df['pred-obs_ abs (t event-1)/RE'] = df['pred-obs_abs (t event-1)']/df['RE corr']

        
    df['RC'] = df['Cfactor_mean'] * df['RE corr']
    df['log RC'] = np.log10(df['RC'])
    df['SDR'] = df['SSL (t event-1)']/(df['RUSLE_sum'])
    df['SDR_WS'] = df['SSL-WS (t event-1)']/(df['RUSLE_sum'])
    sdr_avg = (df['SSL (t event-1)'].sum())/(df['RUSLE_sum'].sum())
    df['SSL_SDR (t event-1)'] = df['RUSLE_sum'] * sdr_avg
    df['Deposition prev'] = df['Total gross deposition (kg)'].shift(1)
    df['Erosion prev'] = df['RUSLE_sum'].shift(1)
    
    if plot == True:
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'SSL-WS (t event-1)', y = 'SSL (t event-1)', s = 200, hue = 'log RC', 
                        edgecolor = 'black', palette = 'PuBu', alpha = 0.5, linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        sns.set(font_scale = 2.5)
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'RE corr', y = 'pred-obs_ abs (t event-1)/RE', size = 'Cfactor_mean', hue = 'Cfactor_mean', 
                        edgecolor = 'black', color = 'white', linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xscale('log')
    
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'Event dur (d)', y = 'pred-obs_ abs (t event-1)/RE', 
                        edgecolor = 'black', color = 'black', alpha = 0.5, linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
    
    
    
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'CN_mean', y = 'pred-obs_ abs (t event-1)/RE', 
                        edgecolor = 'black', color = 'black', alpha = 0.5, linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xscale('log')
    
        pal = sns.diverging_palette(200, 200, center='light', as_cmap=True)
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'RUSLE_max', y = 'SSL (t event-1)', hue = 'Month',  
                        edgecolor = 'black', color = 'black', alpha = 0.5, linewidth=1, s = 200,
                        palette = pal)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'SDR_WS', y = 'SDR', s = 200, size = 'Month', hue = 'Month',
                        palette = pal, edgecolor = 'black', alpha = 0.5, linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
        
        
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'RC', y = 'pred-obs (t event-1)',
                        edgecolor = 'black', alpha = 0.5, linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
    
        
        
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'SSL_SDR (t event-1)', y = 'SSL (t event-1)', 
                        size = 'Month', hue = 'Month', palette = pal,
                        edgecolor = 'black', s = 200, alpha = 0.5, linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        fig, ax = plt.subplots(figsize = (20,10))
        sns.scatterplot(data = df, x = 'Deposition prev', y = 'SDR',
                        edgecolor = 'black', alpha = 0.5, linewidth=1)
        ax.tick_params(axis='x', labelrotation=90)
        
    
        fig, ax = plt.subplots(figsize = (20,5))
        sns.histplot(data = df, x = 'SDR', alpha = 0.5, linewidth=1, bins = 20)
        ax.tick_params(axis='x', labelrotation=90)    
     
    eval_all = {}
    cols = ['tot_erosion', 'RE corr', 'Q (m3 event-1)',
    'SSL-WS (t event-1)', 'pred-obs_abs (t event-1)',
    'Cfactor_mean', 'Cfactor_max', 'Precipitation Depth (mm)',
    'CN_mean', 'CN_max', 'RUSLE_mean', 'RUSLE_max', 'RUSLE_sum',
    'SSL_SDR (t event-1)']
    
    for col in cols:
        name = col + ' vs SSL'

        eval_ = get_metrics(df['SSL (t event-1)'].values, df[col].values, print_ = False, name =  name)

        eval_all[name] = eval_
        
        name2 = col + ' vs SDR'
        #eval_ = get_metrics(df['SDR'].values, df[col].values, print_ = False, name =  name)
        #eval_all[name2] = eval_
        
        
    #analyse variables against error
    #eval_ = get_metrics(df['SDR_WS'].values, df['SDR'].values, print_ = False, name =  name)
    #eval_all['SDR_WS vs SDR'] = eval_
  
    results_posterior = {}
    results_posterior['eval_all'] = eval_all
    results_posterior['event data'] = df
    results_posterior['seasonal bias'] = seasonal_bias
   
    
    return results_posterior


def cumulative_plots(df):
    df['Sim_cumulative (t)'] = df['SSL-WS (t event-1)'].cumsum()
    df['Sim_cumulative_splines (t)'] = df['SSL-WS_splines (t event-1)'].cumsum()
    df['Obs_cumulative (t)'] = df['SSL (t event-1)'].cumsum()
    sns.set(font_scale = 1.5)
    fig, ax = plt.subplots(figsize = (10,5))
    sns.lineplot(data = df, x = 'Date', y = 'Sim_cumulative (t)', marker = 'o', markersize = 6, alpha = 0.5, 
                 linewidth=1, color = 'grey', ax = ax, label = 'W/S static ktc simulation')
    sns.lineplot(data = df, x = 'Date', y = 'Sim_cumulative_splines (t)', marker = 'o', markersize = 6, alpha = 0.5, 
                 linewidth=1, color = 'purple', ax = ax, label = 'W/S dynamic ktc simulation')
    sns.lineplot(data = df, x = 'Date', y = 'Obs_cumulative (t)', marker = 'o', markersize = 6, alpha = 0.5, 
                 linewidth=1, color = 'red', ax = ax, label = 'Observation')
    ax.set_ylabel('Cumulative SSL (t)')
    ax.tick_params(axis='x', labelrotation=90)
    
def cumulative_plots_multiple(df_all):
    
    fig2, axs2 = plt.subplots(2,2, figsize = (20,12))
    plt.tight_layout()
    i = 0
    
    colours = ['royalblue', 'darkorange', 'limegreen', 'firebrick']
    
    for name in df_all['Catchment name'].unique():
        df = df_all[df_all['Catchment name'] == name]
    
        #cumulative_plots(df_rf)
        
        df['Sim_cumulative (t)'] = df['SSL-WS (t event-1)'].cumsum()
        df['Sim_cumulative_splines (t)'] = df['SSL-WS_splines (t event-1)'].cumsum()
        df['Obs_cumulative (t)'] = df['SSL (t event-1)'].cumsum()
    
        if i == 0:
            ax_i = axs2[0,0]
        elif i == 1:
            ax_i = axs2[0,1]
        elif i == 2:
            ax_i = axs2[1,0]
        elif i == 3:
            ax_i = axs2[1,1]
            
        sns.lineplot(data = df, x = 'Date', y = 'Sim_cumulative (t)', marker = 'o', markersize = 6, alpha = 0.5, 
                     linewidth=2, color = 'grey', label = 'W/S static ktc simulation', ax = ax_i)
        sns.lineplot(data = df, x = 'Date', y = 'Sim_cumulative_splines (t)', marker = 'o', markersize = 6, alpha = 0.5, 
                     linewidth=2, color = 'purple', label = 'W/S dynamic ktc simulation', ax = ax_i)
        sns.lineplot(data = df, x = 'Date', y = 'Obs_cumulative (t)', marker = 'o', markersize = 6, alpha = 0.5,
                     linewidth=4, color = colours[i], label = 'Observation: '+name, ax = ax_i)
        ax_i.set_ylabel('Cumulative SSL (t)')
        
        ax_i.legend(frameon = False, framealpha = 0.9)

        i = i+1

def model_intercomparison(df):
    sns.set(font_scale = 2.5)
    fig = plt.figure(figsize=(30, 35))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    i = 0
    hue_order = ['TC1 (VanOost et al., (2000))', 'TC2 (area-slope)', 'TC3 (cell-wise runoff)']
    for col in df['Catchment'].unique():
        legend = False
        if i == 0 or i == 2:
            legend = True
        df_c = df[df['Catchment'] == col]
        name = df_c['Catchment name'].values[0]
        
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=outer[i], wspace=0.1, hspace=0.4)
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            if j == 0:
                sns.scatterplot(data = df_c, x = 'sdr_max', y = 'NSE', hue = 'TC model', palette = 'Accent', 
                                edgecolor = 'black', sizes = (1, 200), s = 250, alpha = 0.5, linewidth=1, ax = ax,
                                legend = legend, hue_order = hue_order)
                ax.tick_params(axis='x', labelrotation=90)
                #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                ax.set_ylim([-3.0, 1.0])
                ax.set_xlabel('SDR upper limit')
                if legend == True:
                    ax.legend(frameon = False)
            else:
                sns.scatterplot(data = df_c, x = 'Connectivity index', y = 'NSE', hue = 'TC model', palette = 'Accent',
                                edgecolor = 'black', s = 200, alpha = 0.5, linewidth=1, ax = ax, legend = False, hue_order = hue_order)    
                ax.set_ylim([-3.0, 1.0])
            ax.title.set_text(name)
            fig.add_subplot(ax)


        i = i + 1        


    
def get_mean_monthly_avg(df):
    #only consider months with events. This does not count months without erosion.
    
    df = df[['Start timestamp', 'Month', 'SSL (t event-1)', 'SSL-WS (t event-1)']].copy()
    
    df['Year'] = df['Start timestamp'].dt.year
    
    #get unique year and month pairs
    df_rs = df.groupby(['Year', 'Month'], as_index = False).sum()
    df_m = df_rs.groupby('Month', as_index = False).sum()
    #count months to average the monthly value
    month_count = df_rs[['Year', 'Month']].groupby('Month', as_index = False).count().rename(columns = {'Year': 'Month count'})
    df_m['Month count'] = month_count['Month count'].values
    df_m['SSL_avg (t month-1)'] = df_m['SSL (t event-1)'] / df_m['Month count']
    df_m['SSL-WS_avg (t month-1)'] = df_m['SSL-WS (t event-1)'] / df_m['Month count']
    
    df_m = df_m.drop(columns = 'Year')
    
    return df_m
    