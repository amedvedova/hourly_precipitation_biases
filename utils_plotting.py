import os
import string

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def apply_requirements(
    ds,
    count_var="count_all",
    req_count=2400,
    req_station_fraction=0.2,
    req_model_fraction=0.2,
    requirements=True,
):
    """
    We only want to use bins where there's enough stations and models - we set others to NaN
    We filter the precipitation data according to the following rules:
    1. First, for any given station and model, we exclude temperature bins with fewer than 2400 recorded 
        hours (equivalent to 100 days). Only bins meeting this threshold are retained for further analysis.
    2. Second, we calculate the mean value across all stations that meet the first criterion (i.e., bins with 
        at least 2400 hours). If fewer than 20% of the stations (< 56) have valid data for a given temperature bin, 
        the bin is considered not robust and excluded from further analysis. This is done separately for each model.
    3. Finally, for the model ensembles, we calculate the mean across all models that have valid averages 
        from step 2. If fewer than 20% of the models have valid values for a given temperature bin, the 
        bin is considered invalid

    Args:
        ds (xr.Dataset): any of the processed hourly/daily datasets with statistics as a funcion of temperature.
        count_var (str, optional): Variable with the count of hours/days in a given temperature bin.
            It is used to determine whether the dataset fulfills the count requirements. Defaults to "count_all".
        req_count (int, optional): Used in Step 1: required number of values in each temperature bin
            at each station location and in each model. Bins with fewer valid values than this threshold
            are set to NaN. Defaults to 2400.
        req_station_fraction (float, optional): Used in Step 2: required fraction of stations that have to have 
            enough valid values. Defaults to 0.2.
        req_model_fraction (float, optional): Used in Step 3: required fraction of models that have to have 
            enough valid values at enough stations. Defaults to 0.2.
        requirements (bool, optional): Boolean flag deciding whether the requirements are applied to the input dataset.
            If it is set to false, the original input dataset is returned unchanged. Defaults to True.

    Returns:
        ds_threshold or ds (xr.Dataset): One of:
            ds_threshold: a dataset to which the requirements were applied, with NaNs in bins where the 
                requirements were not met
            ds: the original, unchanged dataset
    """

    if requirements is True:
        # Step 1: set bins that don't have the required count of values in them to nan
        ds_threshold = ds.where(ds[count_var] >= req_count, drop=False)

        # Step 2: keep only bins where enough stations have the required amount of values, set others to zero
        # example: for a given model, only 5% of stations exceed the required threshold. even the bins with enough data are set to nan.
        ds_threshold = ds_threshold.where(ds_threshold.notnull().sum(dim='station_name') >= (len(ds_threshold.station_name) * req_station_fraction), drop=False)

        # Step 3: keep only bins where enough models have enough data from enough stations
        if 'model' in ds.dims:
            ds_threshold = ds_threshold.where(ds_threshold.notnull().sum(dim='model') >= (len(ds_threshold.model) * req_model_fraction), drop=False)
        return ds_threshold

    else:
        return ds


def open_and_prepare_dataset(file_path, apply_req=False, elev_band=None, select_vars=False, get_mean=False, get_mean_weighted=False, **req_dict):
    # open dataset
    ds = xr.open_dataset(file_path, engine="h5netcdf")

    # unify quantile/quantiles names
    if "quantile" in ds.dims:
        ds = ds.rename({"quantile": "quantiles"})

    # get the percentage of wet hours/days if files where it's relevant (only quantiles files, not in daystats files)
    if "count_wet" in ds.data_vars:
        ds["wet_percentage"] = ds.count_wet / ds.count_all * 100

    # get elevation bands 
    if elev_band is not None:
        ds = ds.where((ds.elevation >= elev_band[0]) & (ds.elevation <= elev_band[1]), drop=True)
        ds.attrs["num_stations"] = len(ds.station_name)

    # subset the variables in the dataset to save time when applying the requirements and calculating means etc
    # relevant for the sensitivity analysis
    if select_vars:
        vars_keep = [
            "mean_daily_precipitation",
            "wet_hour_mean_intensity_all",
            "wet_hour_max_intensity_all",
            "pr_onset_time_all",
            "count_all",
            "daily_count",
            "wet_percentage",
        ]
        vars_keep = [v for v in vars_keep if v in ds.data_vars]
        ds = ds[vars_keep]

    # apply requirements for robustness
    if apply_req:
        ds = apply_requirements(ds, **req_dict)
    else:
        pass   

    dims = [dim for dim in ["model", "station_name"] if dim in ds.dims]
    # calculate mean over stations (and models where relevant)
    if get_mean:
        # get simple mean
        ds = ds.mean(dim=dims, skipna=True)

    elif get_mean_weighted:
        # get mean weighted by the number of occurrences at each station and model
        count_var = [v for v in ds.data_vars if v in ["count_all", "daily_count"]][0]
        ds = (ds * ds[count_var]).sum(dim=dims) / ds[count_var].sum(dim=dims)

    return ds


def get_mean_temperature(ds):
    """
    Calculate the mean seasonal/yearly temperature of a given dataset: averaged over all stations/models.
    Since we don't do this based on raw timeseries, we have to used a weighted mean of temperature:
    Multiply each temperature by the number of times it occurs, and divide by the total number of events.
    This does not necessarily correspond to the REGIONAL mean seasonal temperature since the station
    locations are nor fully representative of the region.

    Args:
        ds (xr.Dataset): dataset containing the count of days in each temperature bin

    Returns:
        float: mean seasonal/yearly temperature of the dataset.
    """
    # get dimensions of the dataset
    dims = [dim for dim in ['model', 'station_name'] if dim in ds.dims]
    # get mean count in the temperature bin
    # fill nans with zeros to have an equal amount of values along all dimensions
    count_total = ds.count_all.fillna(0).mean(dim=dims)
    # get a mean of temperature weighted by the average count of days in each bin
    T_mean = (count_total * count_total.temperature_bin).sum() / count_total.sum()

    return T_mean


def decorate_axis(ax, axis_grid="both", alpha_grid=0.1, fontsize_small=8):
    """
    Change the tick settings of each subplot and add a grid

    Args:
        ax (mpl.axes.Axes): axis to change the settings of
        axis_grid (str, optional): Which grid lines to draw: either both, or only
        vertical in the CC plot. Defaults to "both".
    """
    # set x-axis (temperature) ticks same for all plots
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=10))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=5))

    # label all major ticks on both x- and y-axis with small labels
    ax.tick_params(axis="x", labelbottom=True, labelsize=fontsize_small)
    ax.tick_params(axis="y", labelleft=True, labelsize=fontsize_small)
    
    # add grid to all plots
    ax.grid(which="major", axis=axis_grid, lw=1, c="k", ls="-", alpha=alpha_grid)

    ax.axvspan(-15, 4, color=[0, 0, 0, 0.05], ec='none')
    return


def add_CC_lines(ax, t_bins, CC_alpha=0.1, lw=1, y0=0.125, plot_2CC=True):
    """
    Add thin lines to guide the eye to each subplot: at 7%/K and 14%/K increase.

    Args:
        ax (mpl.axes.Axes): axis to plot the lines on.
        t_bins (xr.DataArray): Temperature bins, used as the x-axis coordinate
        CC_alpha (float, optional): Transparency of the lines. Defaults to 0.5.
        lw (float, optional): Linewidth of the lines. Defaults to 0.5.
    """
    # CC scaling axes: add 7% and 14% increase lines
    x0 = t_bins[0]
    x1 = t_bins[-1]

    for y in [2*y0, 4*y0, 8*y0, 16*y0, 32*y0, 64*y0]:
        ax.plot([x0, x1], [y, y*1.07**(x1-x0)], color='k', ls=(5, (10, 3)), lw=lw, alpha=CC_alpha)
    # only add 14% increase lines if this is True, don't plot otherwise
    if plot_2CC:
        for y in [y0, 2*y0, 4*y0, 8*y0, 16*y0, 32*y0]:
            ax.plot([x0, x1], [y, y*1.14**(x1-x0)], color='k', ls='-.', lw=lw, alpha=CC_alpha)
    return


def plot_scaling(ds, ax, variable_prcp='prcp_percentiles_all', colors=None, lw=2, q_min=0.99):
    """
    Plot the scaling of high hourly precipitation percentiles against mean daily temperature.

    Args:
        ds (xr.Dataset): dataset containing the precipitation percentiles as a function of temperature
            (the "qunatiles" files in our case)
        ax (mpl.axes.Axes): axis to plot the scaling on
        variable_prcp (str, optional): Variable where the percentiles are stored. Defaults to 'prcp_percentiles_all'.
        colors (list, optional): List of strings: color hex codes. Defaults to None.
        lw (int, optional): Linewidth for lines to be plotted. The evaluation lines are supposed to be a bit 
            thicker in the plot than historical/rcp lines. Defaults to 2.
        q_min (optional): Lowest quantile to plot, although even lower quantiles might have been calculated
    """

    # select variable
    ds_var = ds[variable_prcp]

    # start at a given quantile, higher or equal to q_min
    ds_var = ds_var.sel(quantiles=ds_var.quantiles[ds_var.quantiles >= q_min])       

    # loop through percentiles, only 0.9 and higher           
    for q_idx, (q ,c) in enumerate(zip(ds_var.quantiles, colors)):  
        # add means to the plot, label those with the corresponding q
        ax.plot(ds_var.temperature_bin, ds_var[:, q_idx], lw=lw, color=c, label=f'p{q*100:.1f}')

    return


def get_nearest_neighbor_point(ds, station_lon, station_lat, lon_coord='lon', lat_coord='lat'):
    """Given a longitude/latitude coordinate pair, find the gridcell with the lowest
    Euclidean distance to this point. Extract the longitude and latitude of this grid
    cell and the elevation of this point in the orography file

    Args:
        ds (xr.Dataset): Orography file from which we want to extract the staiton locations
        station_lon (float): Longitude of the actual weather station
        station_lat (float): Latitude of the actual weather station
        lon_coord (str, optional): Name of the longitude coordinate in the orography file. Defaults to 'lon'.
        lat_coord (str, optional): Name of the latitude coordinate in the orography file. Defaults to 'lat'.

    Returns:
        point_coords (dict): dictionary holding the latitude, longitude, and elevation of 
            the station in the orog file based on the nearest neighbor grid cell
    """

    # get euclidean distance from all grid points to the desired station lon/lat
    centered_lon = ds[lon_coord] - station_lon
    centered_lat = ds[lat_coord] - station_lat
    euclidean_distance = centered_lon**2 + centered_lat**2

    # select the point with the lowest euclidean distance
    nearest_point = euclidean_distance.where(euclidean_distance == np.min(euclidean_distance), drop=True).squeeze()

    # get elevation at that point (from orog)
    elevation = ds['orog'].where(euclidean_distance == np.min(euclidean_distance), drop=True).squeeze()

    # get the position of the nearest neighbor of the station in all available coordinates
    coords = list(nearest_point.coords)
    coords = [c for c in coords if c not in ['time', 'bin']]
    
    # get lat/lon or whatever the coordinates are called
    point_coords = {}
    for c in coords:
        point_coords[c] = nearest_point[c].item() 

    # print(point_coords)
    point_coords['elevation'] = elevation

    return point_coords


def plot_normalized_frequency_step(ds, ax, ls='-', color_primary='k', color_secondary='gray', normalization_factor=2):
    """
    Plot of the temperature distributions, and of wet hour/day distributions.
    For simplicity of reading, only every second temperature bin is plotted - the temperature bins
    are overlapping anyway, which would lead to double-counting. This way each value is shown
    only once instead of twice.

    There is a different total number of days in each dataset: some stations have up to 30
    years of data, but all models have exactly 10 years of data. We therefore need to normalize
    the temperature distributions for comparability.
    Since the bins are overlapping, the spacing of the x-axis is 0.5 degrees C - thus, the default
    normalization factor is 2 to account for this spacing: naively integrating the area under the 
    curve with dx=0.5 would yield an area of 0.5 instead of 1. With this normalization factor, all 
    days/hours datasets would be correctly normalized to 1, which would be equivalent to a probability 
    density function (PDF) of temperature in the dataset. The wet/days hours are normalized by 
    the total number of days, which yields a "joint PDF" with an area smaller than the PDF of all
    days.

    In the publication figures we chose a normalization factor of 2 * 365 for days (and 2*365*24 
    for hours), which then shows the average number of days *per year* in each temperature bin.
    The "2" again accounts for the overlapping bins, and the 365 for the number of days in a year.
    For seasons, we divide these by 4, neglecting the slightly different number of days in each season.

    This normalization is correct, I checked multiple times.

    Args:
        ds (xr.Dataset): Dataset with the count of all/wet hours/days. In this notebook it's the 
        "quantile" files.
        ax (mpl.axes.Axes): axis to plot the lines on.
        ls (str, optional): Linestyle. Defaults to '-'.
        color_primary (str, optional): Color to be used for the like representing all
            days/hours. Defaults to 'k'.
        color_secondary (str, optional): Color to be used for the like representing only wet
            days/hours. Defaults to 'gray'.
        normalization_factor (int, optional): Normalization factor ensuring that all plotted
            datasets are comparable, i.e. that the area under the curve is the same. Defaults to 2.
    """
    # average over certain dimensions: stations for station data, models and stations for model data
    dims = [dim for dim in ['model', 'station_name'] if dim in ds.dims]


    # Normalize the datasets by the count of all hours/days in a temperature bin
    # Here we still use overlapping bins
    normalized_count_all = ds.count_all / ds.count_all.sum()
    normalized_count_wet = ds.count_wet / ds.count_all.sum()

    # We normalize all bins, therefore we still need the 2 in the normalization factor, 
    #   but we plot only every second bin
    (normalized_count_all.sum(dim=dims, skipna=True) * normalization_factor)[::2].plot.step(ax=ax, color=color_primary, ls=ls)
    (normalized_count_wet.sum(dim=dims, skipna=True) * normalization_factor)[::2].plot.step(ax=ax, color=color_secondary, ls=ls)

    return


def plot_pr_range_count_step(ds, dims, var_full, ax, ls, color, normalization_factor=2, lw=None, label=None, alpha=1):
    """
    Plot the absolute number of days per year that occur in each of the precipitation ranges in a given
    temperature bin.

    There is a different total number of days in each dataset: some stations have up to 30
    years of data, but all models have exactly 10 years of data. We therefore need to normalize
    the temperature distributions for comparability.
    Since the bins are overlapping, the spacing of the x-axis is 0.5 degrees C - thus, the default
    normalization factor is 2 to account for this spacing: naively integrating the area under the 
    curve with dx=0.5 would yield an area of 0.5 instead of 1. With this normalization factor, all 
    days/hours datasets would be correctly normalized to 1, which would be equivalent to a probability 
    density function (PDF) of temperature in the dataset. The wet/days hours are normalized by 
    the total number of days, which yields a "joint PDF" with an area smaller than the PDF of all
    days.

    In the publication figures we chose a normalization factor of 2 * 365 for days (and 2*365*24 
    for hours), which then shows the average number of days *per year* in each temperature bin.
    The "2" again accounts for the overlapping bins, and the 365 for the number of days in a year.
    For seasons, we divide these by 4, neglecting the slightly different number of days in each season.


    Args:
        ds (xr.Dataset): Dataset with the precipitation statistics separated my total daily precipitation
            sums. In this notebook it's the "various daily stats" files.
        dims (list): Dimensions to take the sums over.
        var_full (str): Variable to plot of the form 'pr_range_count_Range', where Range is one of:
            '0p1_1p0', '1p0_2p5', '2p5_5p0', '05_10', '10_plus'
        ax (mpl.axes.Axes): axis to plot the lines on.
        ls (str): Linestyle.
        color (str): Color of the plotted line.
        normalization_factor (int, optional): Normalization factor ensuring that all plotted
            datasets are comparable, i.e. that the area under the curve is the same. Defaults to 2.
        lw (int/float, optional): Linewidth of the plotted line. Defaults to None.
        label (str, optional): Label of the dataset, used in legends. Defaults to None.
        alpha (float, optional): Transparency of the plotted line. Defaults to 1.
    """
    data_to_plot = (normalization_factor * ds[var_full] / ds['daily_count'].sum()).sum(dim=dims)[::2]
    data_to_plot.plot.step(ax=ax, ls=ls, lw=lw, c=color, label=label, alpha=alpha)
    return




