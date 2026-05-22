import os
import cftime
import numpy as np
import pandas as pd
import xarray as xr
from flox.xarray import xarray_reduce


# bins for grouping data: calculate overlapping binned series to get smoother data
# it is necessary to have two arrays with step size 1 ("bin width") for some intermediate processing, the final array serves as an index
T_BINS_1 = np.arange(-15, 36, 1)
T_BINS_2 = np.arange(-14.5, 36.5, 1)
T_BINS = np.ravel([T_BINS_1[:-1], T_BINS_2[:-1]], order="F")

# define thresholds for wet/dry days
WET_HOUR_INTENSITY = 0.1  # mm/h
WET_DAY_INTENSITY = 1  # mm/d

# naming used for various ranges of mean daily precipitation
dict_pr_ranges = {
    "all": (0.1, 999),      # all days with at least one wet hour
    "wet": (1, 999),        # all wet days (at least 1mm/day)
    "0p1_1p0": (0.1, 1),
    "1p0_2p5": (1, 2.5),
    "2p5_5p0": (2.5, 5),
    "05_10": (5, 10),
    "10_plus": (10, 999),
}


def bin_values(ds, var_grouped, groupby_var, func, quantiles=None):
    """
    This function is where the grouping by tmeperature happens. 
    The input data array contains 'time' as a dimension; the output contains 'temperature bin'.
    The data is grouped by intervals, not by discrete values

    Args:
        ds (xr.Dataset): A dataset with hourly or daily time resolution. Contains 'time' as a dimension.
        var_grouped (str): The variable to be grouped. In our case this is mostly some characteristic of precipitation, e.g. hourly intensity.
        groupby_var (str): Variable to group by. In our case this is mean daily temperature ('mean_daiy_tas').
        func (str): Function to apply to the groups. This function is applied to each bin. One of:
            'mean': takes an average over a given bin. Used to calculate mean daily precipitation, mean and maximum hourly intensities
            'count': counts the number of items in a given bin. Used to determine the frequency of all/wet hours/days.
            'nanquantile': calculates the specified quantiles of a given bin. Used to calculated all-hour percentiles.
        quantiles (list, optional): List of floats between 0-1. These represent the quantiles to be calculated. 
            If None, no quantiles are calculated. Defaults to None.

    Returns:
        groups_concat (xr.DataArray): a matrix of a statistic as a function of temperature. The dimensions are temperature_bin, station_name, and for model data also model
    """

    # prepare dataset for grouping: set NaNs at all places where one of the variables is NaN
    ds_to_group = ds.where((ds[var_grouped].notnull() & ds[groupby_var].notnull()))

    # same arguments for all grouping
    kwargs = {
        "dim": "time",         # grouping is done along this dimension (it does not appear in the resulting dataset)
        "isbin": True,         # the data is grouped by intervals, not by discrete values
        "fill_value": np.nan,  # fill value for groups with no members
        "func": func,          # function to be performed on each group: mean, count, or nanquantile
    }

    # add another kwarg when calculating quantiles: not applicable for mean, max, count, etc
    if quantiles is not None:
        kwargs["q"] = quantiles

    # group the first variable (mostly something to do with precipitation) by the second one (mean daily temperature)
    # group by first array of temperature bins: 0-1, 1-2, 2-3, ...
    group_1 = xarray_reduce(
        ds_to_group[var_grouped],
        ds_to_group[groupby_var],
        expected_groups=pd.IntervalIndex.from_breaks(T_BINS_1),
        **kwargs,
    )
    # group by second array of temperature bins: 0.5-1.5, 1.5-2.5, 2.5-3.5, ...
    group_2 = xarray_reduce(
        ds_to_group[var_grouped],
        ds_to_group[groupby_var],
        expected_groups=pd.IntervalIndex.from_breaks(T_BINS_2),
        **kwargs,
    )

    # concatenate the two groups, sort by increasing temperature (so that the two groups alternate: 0-1, 0.5-1.5, 1-2, 1.5-2.5, ...)
    groups_concat = xr.concat([group_1, group_2], dim=f"{groupby_var}_bins").sortby(f"{groupby_var}_bins")

    # rename the grouping variables so that it's consistent among all variables
    groups_concat = groups_concat.rename({f"{groupby_var}_bins": "temperature_bin"})
    
    return groups_concat


def get_quantiles_and_counts(ds, groupby_var="mean_daily_tas", wet_intensity=WET_HOUR_INTENSITY):
    """
    This function calculates the all-hour/all-day quantiles, and the count of all/wet days and hours.
    These are calculated for each temperature bin separately.

    Args:
        ds (xr.Dataset): A dataset with hourly or daily time resolution. 
        groupby_var (str, optional): Variable that the array is grouped by. Defaults to 'mean_daily_tas'.
        wet_intensity (float, optional): A threshold above which the hour/day is considered to be wet. Defaults to WET_HOUR_INTENSITY (global constant).

    Returns:
        quantiles_all (xr.DataArray): high percentiles of hourly/daily precipitation vs. mean daily temperature
        count_all (xr.DataArray): count of all hours/days
        count_wet (xr.DataArray): count of wet hours/days
    """

    # get all/wet hours/days
    ds_all = ds
    ds_wet = ds.where(ds.pr >= wet_intensity)

    # calculate quantiles for each bin (all hours/days, not only wet ones), 1 is basically max
    quantiles = [0.9, 0.95, 0.99, 0.999, 1]
    quantiles_all = bin_values(ds_all, "pr", groupby_var, "nanquantile", quantiles=quantiles)

    # get count of all values, wet values, dry values in each bin
    count_all = bin_values(ds_all, "pr", groupby_var, "count")
    count_wet = bin_values(ds_wet, "pr", groupby_var, "count")

    return quantiles_all, count_all, count_wet


def get_daily_stats(ds_hourly):
    """
    This function calculates certain precipitation stats for EVERY SINGLE DAY, before any grouping occurs.

    We also take the hourly timeseries and extend them by two mean daily temperature and daily precipitation sum,
    which are reindexed from daily to hourly frequency:

    All these measures serve for later grouping by temperature and daily precipitation sums

    Args:
        ds_hourly (xr.Dataset): input dataset which contains hourly precipitation and temperature timeseries plus some metadata.

    Returns:
        ds_hourly_extended (xr.Dataset): dataset with hourly frequency which contains:
            - hourly precipitation timeseries
            - hourly temperature timeseries
            - mean daily temperature timeseries broadcasted to an hourly frequency
            - total daily precipitation timeseries broadcasted to hourly frequency
        ds_daily_extended (xr.Dataset): dataset which daily frequency which contains the following timeseries:
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
    """

    # if needed, convert 360 day calendar to 365 day calendar: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.convert_calendar.html
    # this is relevant for only a few models
    if type(ds_hourly.time[0].item()) is cftime._cftime.Datetime360Day:
        ds_hourly = ds_hourly.convert_calendar(calendar="gregorian", align_on="year")
    else:
        pass

    if "station_name" not in ds_hourly.dims:
        ds_hourly = ds_hourly.rename({"stations": "station_name"})

    # get mean daily temperature: skipna False ensures that days which have at least one NaN are filled with zeros: 
    # otherwise e.g. a sum of 24 NaNs is 0 which I don't want
    mean_daily_tas = ds_hourly.tas.groupby(ds_hourly.time.dt.date).mean(skipna=False)
    # get mean daily dewpoint temperature if present in the dataset
    if "td" in ds_hourly.data_vars:
        mean_daily_td = ds_hourly.td.groupby(ds_hourly.time.dt.date).mean(skipna=False)

    # get total daily precipitation
    total_daily_pr = ds_hourly.pr.groupby(ds_hourly.time.dt.date).sum(skipna=False)

    # get a mask of wet hours (boolean)
    ds_hourly_wet = (ds_hourly.pr >= WET_HOUR_INTENSITY)
    # group wet hours by date
    pr_hourly_grouped = ds_hourly.pr.where(ds_hourly_wet).groupby(ds_hourly.time.dt.date)

    # get count of wet hours (sum of "True", i.e. 1, for every day)
    wet_hour_count = ds_hourly_wet.groupby(ds_hourly.time.dt.date).sum(skipna=True)

    # get intensity on days with wet hours: mean and maximum of EVERY DAY
    wet_hour_mean_intensity = pr_hourly_grouped.mean(skipna=True)
    wet_hour_max_intensity = pr_hourly_grouped.max(skipna=True)

    # get the first hour when precipitation occurs, add that to the daily data
    pr_onset_time = (ds_hourly.time.dt.hour).where(ds_hourly_wet).groupby(ds_hourly.time.dt.date).first()

    # create a dataset from individual variables
    ds_daily_extended = xr.Dataset(
        data_vars={
            "mean_daily_tas": mean_daily_tas,
            "pr": total_daily_pr,
            "wet_hour_count": wet_hour_count,
            "wet_hour_mean_intensity": wet_hour_mean_intensity,
            "wet_hour_max_intensity": wet_hour_max_intensity,
            "pr_onset_time": pr_onset_time,
        }
    )
    if "td" in ds_hourly.data_vars:
        ds_daily_extended["mean_daily_td"] = mean_daily_td

    # change the name "date" to "time" in the daily timeseires for late processing
    try:
        ds_daily_extended["date"] = pd.to_datetime(ds_daily_extended.date)
        ds_daily_extended = ds_daily_extended.rename({"date": "time"})
    except AttributeError:
        ds_daily_extended["floor"] = pd.to_datetime(ds_daily_extended.floor)
        ds_daily_extended = ds_daily_extended.rename({"floor": "time"})

    # extend hourly statistics: add the required daily values to hourly data, this will serve for later grouping of hourly data by daily values
    ds_hourly_extended = ds_hourly
    ds_hourly_extended["mean_daily_tas"] = ds_daily_extended["mean_daily_tas"].reindex_like(ds_hourly_extended, method="ffill")
    ds_hourly_extended["pr_daily"] = ds_daily_extended["pr"].reindex_like(ds_hourly_extended, method="ffill")
    if "td" in ds_hourly.data_vars:
        ds_hourly_extended["mean_daily_td"] = ds_daily_extended["mean_daily_td"].reindex_like(ds_hourly_extended, method="ffill")
    
    return ds_hourly_extended, ds_daily_extended


def make_quantiles_file(ds_timeseries, metadata, groupby_var="mean_daily_tas", wet_intensity=WET_HOUR_INTENSITY, description=None, dataset_name=None):
    """
    The function takes hourly/daily timeseries as an input and creates a file with hourly/daily precipitation quantiles and a count of all/wet days/hours

    Args:
        ds_timeseries (xr.Dataset): input dataset which contains hourly or daily precipitation and temperature timeseries.
        metadata (xr.Dataset): metadata about a location: elevation, latitude, longitude, station ID
        groupby_var (str, optional): Variable that the array is grouped by. Defaults to 'mean_daily_tas'.
        wet_intensity (float, optional): A threshold above which the hour/day is considered to be wet. Defaults to WET_HOUR_INTENSITY (global constant).
        description (str, optional): Metadata to be added to dataset attributes. Defaults to None.
        quantiles (list, optional): List of floats between 0-1. These represent the quantiles to be calculated. If None, no quantiles are calculated. Defaults to None.
        dataset_name (str): One of 'kmscale', 'driving', 'GeoSphere'. Explains what type of dataset is being processed and is used in the output file name.

    Returns:
        ds_groupby_temp (xr.Dataset): resulting dataset with quantities grouped by mean daily (dewpoint) temperature, by default mean daily temperature
    """

    # calculate quantiles, mean and max precipitation, counts, etc
    quantiles_all, count_all, count_wet = get_quantiles_and_counts(ds_timeseries, groupby_var=groupby_var, wet_intensity=wet_intensity)

    # make a dataset with all the calculated variables
    ds_groupby_temp = xr.Dataset(
        dict(
            prcp_percentiles_all=quantiles_all,
            count_all=count_all,
            count_wet=count_wet,
            elevation=(["station_name"], metadata['elevation'].values),
            station_id=(["station_name"], metadata['station_id'].values),
            lon=(["station_name"], metadata['lon'].values),
            lat=(["station_name"], metadata['lat'].values)
        ),
        # add attributes in file info
        attrs=dict(
            desctiption=description,
            dataset_name=dataset_name,
            wet_intensity=wet_intensity,
        )
    )

    # change temperature bins to values instead of intervals
    ds_groupby_temp["temperature_bin"] = [interval.left for interval in ds_groupby_temp.temperature_bin.values]

    return ds_groupby_temp


def make_complete_daily_file(ds_daily_timeseries, metadata, groupby_var="mean_daily_tas", dataset_name=None):
    """
    This function takes daily timeseries of various precipitation statistics as an input and produces a dataset where these statistics
    are stored as a function of temperature

    Args:
        ds_daily_timeseries (xr.Dataset): daily timeseries extended by derived quantities
        metadata (xr.Dataset): metadata about a location: elevation, latitude, longitude, station ID
        groupby_var (str, optional): Variable that the array is grouped by. Defaults to 'mean_daily_tas'.
        dataset_name (str): One of 'kmscale', 'driving', 'GeoSphere'. Explains what type of dataset is being processed and is used in the output file name.

    Returns:
        ds_groupby_temp (xr.Dataset): resulting dataset with ~40 quantities grouped by mean daily temperature
    """

    # create a dataset to hold the results
    ds_groupby_temp = xr.Dataset(
        dict(
            # what's the mean daily precipitation? - average over all days (even dry) at a given temperature
            mean_daily_precipitation=bin_values(ds_daily_timeseries, "pr", groupby_var, "mean"),
            # how many times does this bin occur? - get bin counts, including dry days
            daily_count=bin_values(ds_daily_timeseries, "pr", groupby_var, "count"),
            # the usual stats present in each dataset
            elevation=(["station_name"], metadata["elevation"].values),
            station_id=(["station_name"], metadata["station_id"].values),
            lon=(["station_name"], metadata["lon"].values),
            lat=(["station_name"], metadata["lat"].values),
        ),
        # add attributes in file info
        attrs=dict(
            desctiption="Various daily stats grouped by temperature bins. Bins of 1°C, overlapping by 0.5°C.",
            dataset_name=dataset_name,
            wet_hour_intensity=WET_HOUR_INTENSITY,
            wet_day_intensity=WET_DAY_INTENSITY,
        ),
    )

    for range_name, (range_min, range_max) in zip(dict_pr_ranges.keys(), dict_pr_ranges.values()):
        # identify days which fall within a given daily precipitation sum (e.g. 0.1-1 mm/day, 1-2.5mm/day, ...)
        days_within_range = ds_daily_timeseries.where((ds_daily_timeseries.pr >= range_min) & (ds_daily_timeseries.pr < range_max))

        # calculate statistics for that precipitation range and add then to the complete dataset
        ds_groupby_temp[f"wet_hour_mean_intensity_{range_name}"] = bin_values(days_within_range, "wet_hour_mean_intensity", groupby_var, "mean")
        ds_groupby_temp[f"wet_hour_max_intensity_{range_name}"]  = bin_values(days_within_range, "wet_hour_max_intensity",  groupby_var, "mean")
        ds_groupby_temp[f"wet_hour_count_{range_name}"]          = bin_values(days_within_range, "wet_hour_count",          groupby_var, "mean")
        ds_groupby_temp[f"pr_range_count_{range_name}"]          = bin_values(days_within_range, "pr",                      groupby_var, "count")
        ds_groupby_temp[f"pr_onset_time_{range_name}"]           = bin_values(days_within_range, "pr_onset_time",           groupby_var, "mean")

    # change temperature bins to values instead of intervals
    ds_groupby_temp["temperature_bin"] = [interval.left for interval in ds_groupby_temp.temperature_bin.values]

    return ds_groupby_temp


def make_all_datasets_from_hourly_data(ds_hourly, dataset_name, metadata, folder_processed_data=None, savefiles=True, groupby_var="mean_daily_tas", period='eval', season_months=None, season='JJA'):
    """
    This function first takes simple hourly timeseries of temperature and precipitation, and calculates derived quantities on daily timescales.
    If files with these derived quantities don't exist yet, they are created and saved.
    These are then used to create three datasets:
        - high percentiles of hourly precipitation vs. mean daily temperature, count of all hours, count of wet hours
        - high percentiles of daily precipitation vs. mean daily temperature, count of all days, count of wet days
        - various precipitation characteristics: mean daily precipitation, count of days in each category, mean and maximum of wet hour intensities,
            time of onset of precipitation, and all these characteristics separated by various ranges of mean daily precipitation
    These three datasets are saved as three separate files.

    Args:
        ds_hourly (xr.Dataset): input dataset which contains hourly precipitation and temperature timeseries plus some metadata.
        dataset_name (str): One of 'kmscale', 'driving', 'GeoSphere'. Explains what type of dataset is being processed and is used in the output file name.
        metadata (xr.Dataset): metadata about a location: elevation, latitude, longitude, station ID.
        folder_processed_data (str): folder to save the output data
        savefiles (bool, optional): A flag determining whether the created datasets should be saved. Defaults to True.
        groupby_var (str, optional): variable to group by. Either mean daily temperature or dewpoint temperature is used. Defaults to "mean_daily_tas'.
        period (str, optional): Time period, either evaluation ('eval'), historical ('hist'), or future RCP8.5 ('rcp'). Defaults to 'eval'.
        season_months (list or None): one of None, [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]. Used for subsetting months if complete raw 
            yearly files exist.
        season (str, optional): one of 'DJF', 'MAM', 'JJA', 'SON', 'allseasons'. The last one means we consider the whole year. Defaults to 'JJA'.

    Returns:
        N/A. It saves the produced results in files.
    """

    # create full names (paths) for extended timeseries which will hold quantities derived from hourly precipitation and temperature data
    extended_daily_file_out = f"{folder_processed_data}/extended_daily_measures_{dataset_name}_{period}_{season}.nc"
    extended_hourly_file_out = f"{folder_processed_data}/extended_hourly_measures_{dataset_name}_{period}_{season}.nc"

    # Check if the extended files exist: if not, create them
    if not (os.path.exists(extended_daily_file_out) and os.path.exists(extended_hourly_file_out)):
        # If "allseasons" file exist, crop that one to a relevant season
        if (os.path.exists(extended_daily_file_out.replace(season, "allseasons")) and os.path.exists(extended_hourly_file_out.replace(season, "allseasons"))):
            ds_hourly_extended = xr.open_dataset(extended_hourly_file_out.replace(season, "allseasons")) 
            ds_daily_extended = xr.open_dataset(extended_daily_file_out.replace(season, "allseasons"))
            if season_months is not None:
                ds_hourly_extended = ds_hourly_extended.sel(time=ds_hourly_extended.time.dt.month.isin(season_months), drop=True)
                ds_daily_extended = ds_daily_extended.sel(time=ds_daily_extended.time.dt.month.isin(season_months), drop=True)
        else:
            # If "allseasons" file doesn't exist, get extended hourly and daily data from simple hourly data
            ds_hourly_extended, ds_daily_extended = get_daily_stats(ds_hourly)

        ds_daily_extended["time"] = pd.to_datetime(ds_daily_extended["time"])
        print(f"Daily data obtained: {dataset_name} {period} {season}")

        # make datasets with extended daily statistics grouped by temperature bins
        if savefiles:
            ds_daily_extended.to_netcdf(extended_daily_file_out)
            ds_hourly_extended.to_netcdf(extended_hourly_file_out)
    # if files already exist, open them instead
    else:
        # print("Daily data exists")
        ds_daily_extended = xr.open_dataset(extended_daily_file_out, engine='h5netcdf')
        ds_hourly_extended = xr.open_dataset(extended_hourly_file_out, engine='h5netcdf')

    # define dataset desctiptions
    description_dict = {
        "hdmean": "High percentiles of hourly precipitation vs. mean daily temperature, count of all hours, count of wet hours. Bins of 1°C, overlapping by 0.5°C.",
        "ddmean": "High percentiles of daily precipitation vs. mean daily temperature, count of all days, count of wet days. Bins of 1°C, overlapping by 0.5°C.",
    }

    # determine string for saving based on grouping variable
    if groupby_var == "mean_daily_td":
        var_str = "_td"
    else:
        var_str = ""

    # make datasets with hourly/daily quantiles and all/wet hour/day counts based on hourly/daily data
    hdmean_path_out = f"{folder_processed_data}/quantiles_{dataset_name}_{period}_hdmean_{season}{var_str}.nc"
    ddmean_path_out = f"{folder_processed_data}/quantiles_{dataset_name}_{period}_ddmean_{season}{var_str}.nc"
    daily_stats_dmean_path_out = f"{folder_processed_data}/various_daily_stats_{dataset_name}_{period}_dmean_{season}{var_str}.nc"
    
    if not (os.path.exists(hdmean_path_out) and os.path.exists(ddmean_path_out) and os.path.exists(daily_stats_dmean_path_out)):
        ds_hdmean = make_quantiles_file(ds_hourly_extended, metadata, groupby_var=groupby_var, wet_intensity=WET_HOUR_INTENSITY, description=description_dict["hdmean"], dataset_name=dataset_name)
        ds_ddmean = make_quantiles_file(ds_daily_extended, metadata, groupby_var=groupby_var, wet_intensity=WET_DAY_INTENSITY, description=description_dict["ddmean"], dataset_name=dataset_name)

        # make datasets with various daily stats based on extended daily data
        ds_daily_stats_dmean = make_complete_daily_file(ds_daily_extended, metadata, groupby_var=groupby_var, dataset_name=dataset_name)

        # save files only if the flag is True
        if savefiles:
            # fill in the blanks in the names based on which dataset is processed
            ds_hdmean.to_netcdf(hdmean_path_out)
            ds_ddmean.to_netcdf(ddmean_path_out)
            ds_daily_stats_dmean.to_netcdf(daily_stats_dmean_path_out)

    return


def make_all_datasets_from_hourly_data_parallel(file_path, season_months, season, metadata, folder_processed_data=None, savefiles=True, period='eval', groupby_var="mean_daily_tas"):
    """
    This functions serves as a parallelization wrapper of the 'make_all_datasets_from_hourly_data' function.
    It automatically determines the type of dataset (km-scale, driving, station data).
    If specified, this function also selects one season from the dataset before further processing.

    Args:
        file_path (str): path to the input file which contains precipitation and temperature timeseries plus some metadata.
            All the input files are in the .nc format.
        season_months (list or None): one of [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], None. If None, no selection is made, i.e. we take the whole year.
        season (str): one of 'DJF', 'MAM', 'JJA', 'SON', 'allseasons'. The last one means we consider the whole year.
        folder_processed_data (str): folder to save the output data
        metadata (xr.Dataset): metadata about a location: elevation, latitude, longitude, station ID.
        savefiles (bool, optional): A flag determining whether the created datasets should be saved. Defaults to True.
        period (str, optional): Time period, either evaluation ('eval'), historical ('hist'), or future RCP8.5 ('rcp'). Defaults to 'eval'.
        groupby_var (str, optional): variable to group by. Either mean daily temperature or dewpoint temperature is used. Defaults to "mean_daily_tas'.

    Returns:
        N/A. It saves the produced results in files.
    """

    # determine which model we're dealing with based on the filepath
    model = file_path.split("/")[-2]

    # determine resolution based on the filepath
    if "ALP-3" in file_path:
        res = "kmscale"
        dataset_name = f"{res}_{model}"
    elif "ALP-12" in file_path:
        res = "upscale"
        dataset_name = f"{res}_{model}"
    elif "EUR-12" in file_path:
        res = "driving"
        dataset_name = f"{res}_{model}"
    elif "Geo" in file_path or "geosphere" in file_path:
        dataset_name = "GeoSphere"
    else:
        raise ValueError("No model found in path")
    
    # determine string for saving based on grouping variable
    if groupby_var == "mean_daily_td":
        var_str = "_td"
    else:
        var_str = ""
    
    # make datasets with hourly/daily quantiles and all/wet hour/day counts based on hourly/daily data
    hdmean_path_out = f"{folder_processed_data}/quantiles_{dataset_name}_{period}_hdmean_{season}{var_str}.nc"
    ddmean_path_out = f"{folder_processed_data}/quantiles_{dataset_name}_{period}_ddmean_{season}{var_str}.nc"
    daily_stats_dmean_path_out = f"{folder_processed_data}/various_daily_stats_{dataset_name}_{period}_dmean_{season}{var_str}.nc"
    
    if not (os.path.exists(hdmean_path_out) and os.path.exists(ddmean_path_out) and os.path.exists(daily_stats_dmean_path_out)):
        # print what's data is being processed now
        print(f'{season} {dataset_name} {period}')
        print(file_path)

        # open file
        ds_model = xr.open_dataset(file_path, engine='h5netcdf')

        # subselect months if we don't do the whole year
        # It would have been smarter to make the extended timeseries only once and select seasons afterwards, but oh well...
        if season_months is not None:
            ds_model = ds_model.sel(time=ds_model.time.dt.month.isin(season_months), drop=True)
        
        # pass all the determined arguments further and create resulting datasets
        make_all_datasets_from_hourly_data(
            ds_model,
            dataset_name,
            metadata,
            folder_processed_data=folder_processed_data,
            savefiles=savefiles,
            period=period,
            season_months=season_months,
            season=season,
            groupby_var=groupby_var,
        )

    return
