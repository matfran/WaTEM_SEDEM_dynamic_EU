# WaTEM_SEDEM_dynamic_EU
The code respository for a multitemporal implementation of the WaTEM/SEDEM soil erosion and sediment delivery model. The relevant modules are contained to: 1) run the preprocessing of all model input layers from EU data sources to undertake catchment-scale modelling, 2) run WaTEM/SEDEM in a multitemporal format and perform several types (time-lumped and time-dynamic) calibration using different optimisation routines. For the time-dynamic calibration routine the model is linked within the scipy.optimise function to undertake an efficient external optimisation scheme, independent of the internal WaTEM/SEDEM calibration routine. 

The repository contains two main scripts:
1) WaTEM_SEDEM_PREPROCESSING_1: the script to undertake the (geo)processing for the multitemporal and static input layers.
2) WaTEM_SEDEM_IMPLEMENTATION_2: the script to launch multitemporal simulations using the WaTEM/SEDEM model as well as run the calibration routine and merge the simulations with catchment observations from the EUSEDcollab repository (https://www.nature.com/articles/s41597-023-02393-8).

The relevant data to run examples of each routine can be found at:
