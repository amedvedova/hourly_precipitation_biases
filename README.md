# README for the code of the manuscript: Temperature-dependent Hourly Precipitation Biases in Convection-parameterizing Regional Climate Models: Insights from the Kilometer-scale

Code and data to reproduce analysis and the figures are also available on Zenodo.

We investigated the characteristics of hourly precipitation in two regional climate model (RCM) ensembles covering the greater Alpine region: the km-scale CORDEX-FPS Convection ensemble (deep convection resolving), and a coarser (convection-parameterizing) RCM ensemble from which the km-scale ensemble was downscaled. 
We compared the model data with observations from 277 weather stations in Austria.

Due to storage limitations, we provide climate model data in the form of time series of precipitation and temperature at the 277 station locations. 
However, some of the raw climate model output is publicly available through the Earth System Grid Federation (ESGF; https://esgf-metagrid.cloud.dkrz.de/). 
Additionally, we provide the observational data from the weather stations obtained from Geosphere Austria (https://data.hub.geosphere.at/dataset/klima-v2-1h), and compiled into a single file for ease of use.
Running the processing scripts creates additional 185 GB of data.

All provided datasets have been pre-processed to remove obvious outliers. 
Using the shortest Euclidean distance, we selected the nearest neighbor grid cell to each weather station from all models of the used ensembles.


## Overview of data processing and figure notebooks

### 1. Data processing: process_raw_timeseries.ipynb

As **input**, this script takes raw timeseries from the stations and models, saved in .nc files:
- Separately for each model
- Exactly 10 years for models, up to 30 years for observations
- Precipitation and temperature
- Hourly resolution
- Only station locations, i.e. for the models it's not gridded data, but timeseries from 277 grid cells that are nearest neighbors of the stations
- The files were processed to exclude obivous outliers, e.g. timesteps with more than 100mm/h averaged over all stations (i.e., large scale). Similar outliers were found and removed from station data (precipitation over ~200 mm/h). It is possible that some outliers remain.

As **output**, this script produces two types of auxiliary .nc files where intermediate results are stored:
- **extended hourly timeseries**: dataset with hourly frequency which contains:
    - hourly precipitation timeseries
    - hourly temperature timeseries
    - mean daily temperature timeseries broadcasted to an hourly frequency
    - total daily precipitation timeseries broadcasted to hourly frequency
- **extended daily timeseries**: dataset which contains the following timeseries with daily frequency:
    - 'mean_daily_tas' (average temperature over 24 hours)
    - 'pr' (total daily precipitation sum over 24 hours)
    - 'wet_hour_count' (number of wet hours within the day, used to determine wet hour freqnency later in the analysis)
    - 'wet_hour_mean_intensity' (how much precipitation falls within an hour if the hour is wet?
        Note: this is calculated separately for each day here; nan if there are zero wet hours in a day)
    - 'wet_hour_max_intensity' (what's the most intense precipitation that falls in that day?
        Note: this in NOT equivalent to high percentiles as this is calculated separately for each day;
        nan if there are zero wet hours in a day)
    - 'pr_onset_time' (time of precipitation onset, i.e. time in UTC when the first wet hour occurs.
        This is calculated only for days with at least one wet hour)

The following .nc files are produced from the extended timeseries and serve as the main output of this script, used in later processing:
- **quantiles_DatasetName_TimePeriod_hdmean_Season.nc**: "hdmean" stands for "hourly precipitation, daily mean tempearture". 
    The files contain hourly precipitation quantiles, count of wet hours, and count of all hours in each temperature bin.
- **quantiles_DatasetName_TimePeriod_ddmean_Season.nc**: "ddmean" stands for "daily precipitation, daily mean tempearture". 
    The files contain daily precipitation quantiles, count of wet days, and count of all days in each temperature bin.
- **various_daily_stats_DatasetName_TimePeriod_dmean_Season.nc**: "dmean" stands for "daily mean tempearture". 
    The files contain mean daily precipitation, count of days at each temperature, mean and maximum of wet hour intensities,
        time of onset of precipitation, as well as all these aforementioned characteristics separated by 
        various ranges of mean daily precipitation (0.1-1, 1-2.5, 25.-5, 5-10, 10+ mm/d).
        
One file is produced per time period (evaluation/historical/rcp), per season (DJF/MAM/JJA/SON/all year), per model.
Subsequently, ensemble files are created, containing all the models per time period and per season.


### 2. Plotting of figures: plot_data.ipynb

This notebook creates figures based on the datasets produced by the process_raw_timeseries.ipynb script.

**Figure 1**: Daily and hourly precipitation statistics as a function of temperature: evaluation, historical, and future (RCP8.5) periods.

**Figure 2**: Precipitation characteristics similar to Fig. 1, but separated by the ranges of total daily precipitation sums, evaluation period only.

**Figure 3**: Scaling of hourly precipitation percentiles with mean daily temperature (Clausius-Clapeyron scaling). 

**Figure S1**: Spatial distribution of weather stations in Austria, and the corresponding nearest neighbor grid cells. Scatter plots of model vs actual station elevation.

**Figure S2**: Average number of days/hours per year that fall within each temperature bin: : evaluation, historical, and future (RCP8.5) periods.

**Figure S3**: As Fig. S1, but for summer (June, July, August): evaluation, historical, and future (RCP8.5) periods.

**Figure S4**: As Fig. 1, but for summer (June, July, August): evaluation, historical, and future (RCP8.5) periods.

**Figure S5**: As Fig. 2, but for the historical and future periods.

**Figure S6**: Average count of days per year with a given daily precipitation sum: evaluation, historical, and future (RCP8.5) periods.
